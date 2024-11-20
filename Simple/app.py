import os
import time
import requests
import base64
import moviepy.editor as mp
from flask import Flask, request, jsonify, render_template
from transformers import pipeline, PegasusTokenizer, PegasusForConditionalGeneration
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Enable CORS for /api/* routes

# Rev.ai API token (ensure this is kept secure, preferably in environment variables)
REV_AI_ACCESS_TOKEN = os.getenv('REV_AI_ACCESS_TOKEN', '02HRtEMEIzsAzBQLdg6g38YAzR9_l8CtANu8G6fQBrT1Y5Weyh5kuPsl_49zbB8srXfVrXZQ-rZRiqijh23TLWX6bcTB0')

# Initialize the summarization pipeline once
summarizer = pipeline("summarization", model="google/pegasus-cnn_dailymail", device=0)  # Use GPU if available

# Set Flask to allow file uploads up to 2GB
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB limit

def convert_video_to_audio(video_buffer):
    """Convert a video file (in memory) to an audio file."""
    video_file_path = 'temp_video.mp4'
    audio_file_path = 'temp_audio.wav'

    try:
        # Save the video buffer to a temporary file
        with open(video_file_path, 'wb') as video_file:
            video_file.write(video_buffer)

        # Convert video to audio directly without compression
        video_clip = mp.VideoFileClip(video_file_path)
        video_clip.audio.write_audiofile(audio_file_path, codec='pcm_s16le')
        video_clip.close()

        # Read the audio file into memory
        with open(audio_file_path, 'rb') as audio_file:
            audio_data = audio_file.read()

    except Exception as e:
        print(f"Error during video to audio conversion: {e}")
        raise e
    finally:
        # Clean up temporary files
        for path in [video_file_path, audio_file_path]:
            if os.path.exists(path):
                os.remove(path)

    return audio_data


# Initialize the Pegasus tokenizer and model
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-cnn_dailymail")

def summarize_text(text):
    """Summarize the given text using a pre-trained Pegasus model."""
    # Tokenize input text into manageable chunks
    max_chunk_size = 1024  # Pegasus also has a maximum input size of 1024 tokens
    tokens = tokenizer.encode(text, return_tensors="pt", truncation=True)

    summaries = []
    for i in range(0, len(tokens[0]), max_chunk_size):
        chunk = tokens[0][i:i + max_chunk_size].unsqueeze(0)
        
        try:
            # Dynamically set max_length based on input length
            input_length = chunk.size(1)
            max_length = max(10, min(50, input_length // 2))  # Set max_length to half the input length, but not less than 10
            
            summary_ids = model.generate(chunk, max_length=max_length, min_length=10, length_penalty=2.0, num_beams=4, early_stopping=True)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
    
    # Combine all summaries into a final summary
    final_summary = ' '.join(summaries)
    return final_summary


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/transcribe', methods=['POST'])
def transcribe_media():
    """Handles the transcription of audio and video files."""
    try:
        # Get the base64 encoded media (audio or video) from the request
        base64_media = request.json.get('media')
        media_type = request.json.get('type')  # type could be 'audio' or 'video'

        if not base64_media or not media_type:
            return jsonify({"error": "No media data or type provided."}), 400

        # Convert the base64 media data into a binary buffer
        media_buffer = base64.b64decode(base64_media)

        if media_type == 'audio':
            # If the media is audio, we can use it directly
            audio_buffer = media_buffer
        elif media_type == 'video':
            # If the media is video, convert it to audio
            audio_buffer = convert_video_to_audio(media_buffer)
        else:
            return jsonify({"error": "Invalid media type."}), 400

        # Create form data to send to Rev.ai
        files = {
            'media': ('audio.wav', audio_buffer, 'audio/wav')
        }

        # Upload the media file to Rev.ai
        response = requests.post(
            'https://api.rev.ai/speechtotext/v1/jobs',
            files=files,
            headers={
                'Authorization': f'Bearer {REV_AI_ACCESS_TOKEN}'
            }
        )

        if response.status_code != 200:
            print(f"Rev.ai upload failed: {response.text}")
            return jsonify({"error": "Failed to upload media to Rev.ai"}), 500

        # Extract job ID from the response
        job_id = response.json().get('id')
        print(f"Job ID: {job_id}")

        # Poll for the job status
        job_status = poll_rev_ai_job_status(job_id)

        if job_status == 'transcribed':
            # Retrieve the transcript
            transcript_response = requests.get(
                f'https://api.rev.ai/speechtotext/v1/jobs/{job_id}/transcript',
                headers={
                    'Authorization': f'Bearer {REV_AI_ACCESS_TOKEN}',
                    'Accept': 'application/vnd.rev.transcript.v1.0+json',
                }
            )
            if transcript_response.status_code != 200:
                print(f"Failed to retrieve transcript: {transcript_response.text}")
                return jsonify({"error": "Failed to retrieve transcript."}), 500

            transcript_data = transcript_response.json()

            # Extract the transcription text
            if 'monologues' not in transcript_data:
                return jsonify({"error": "Transcript format invalid."}), 500

            transcript_text = ' '.join(
                [element['value'] for monologue in transcript_data['monologues'] for element in monologue['elements']]
            )

            if not transcript_text or transcript_text.strip() == '':
                return jsonify({"error": "Transcription is empty."}), 500

            # Summarize the transcription text
            summary = summarize_text(transcript_text)

            # Respond with the transcription and summary
            return jsonify({
                'transcription': transcript_text,
                'summary': summary
            })

        else:
            return jsonify({"error": f"Failed to transcribe media. Job status: {job_status}"}), 500

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Failed to process media."}), 500

def poll_rev_ai_job_status(job_id):
    """Poll the Rev.ai API until the job is finished or the attempt limit is reached."""
    job_status = 'in_progress'
    attempts = 0
    max_attempts = 1200  # Allow for up to 10 minutes of polling (5-second intervals)
    while job_status == 'in_progress' and attempts < max_attempts:
        status_response = requests.get(
            f'https://api.rev.ai/speechtotext/v1/jobs/{job_id}',
            headers={
                'Authorization': f'Bearer {REV_AI_ACCESS_TOKEN}'
            }
        )
        if status_response.status_code != 200:
            print(f"Failed to get job status: {status_response.text}")
            return 'failed'
        job_status = status_response.json().get('status', 'failed')
        print(f"Polling attempt {attempts + 1}, job status: {job_status}")
        if job_status == 'in_progress':
            time.sleep(5)  # Wait 5 seconds between each poll
        attempts += 1
    return job_status

if __name__ == '__main__':
    app.run(port=3000, debug=True)
