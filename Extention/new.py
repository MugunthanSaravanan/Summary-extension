import os
import time
import requests
import base64
import moviepy.editor as mp
from flask import Flask, request, jsonify, render_template, send_from_directory
from transformers import pipeline
from flask_cors import CORS
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Enable CORS for /api/* routes

# Rev.ai API token (ensure this is kept secure, preferably in environment variables)
REV_AI_ACCESS_TOKEN = os.getenv('REV_AI_ACCESS_TOKEN', '02HRtEMEIzsAzBQLdg6g38YAzR9_l8CtANu8G6fQBrT1Y5Weyh5kuPsl_49zbB8srXfVrXZQ-rZRiqijh23TLWX6bcTB0')

# Set Flask to allow file uploads up to 2GB
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB limit

import os
import moviepy.editor as mp
import tempfile
import subprocess

def convert_video_to_audio(video_buffer):
    """Convert a video file (in memory) to an audio file."""
    try:
        # Create temporary video file
        temp_video_path = tempfile.mktemp(suffix=".mp4")
        temp_audio_path = tempfile.mktemp(suffix=".wav")

        with open(temp_video_path, 'wb') as video_file:
            video_file.write(video_buffer)

        # Use ffmpeg directly for conversion from video to audio
        # This is to ensure ffmpeg handles the video correctly
        command = [
            'ffmpeg', '-i', temp_video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', temp_audio_path
        ]
        subprocess.run(command, check=True)  # Run the ffmpeg command and ensure it completes without error

        # Read the resulting audio file into memory
        with open(temp_audio_path, 'rb') as audio_file:
            audio_data = audio_file.read()

        # Clean up temporary files
        os.remove(temp_video_path)
        os.remove(temp_audio_path)

        return audio_data

    except Exception as e:
        print(f"Error during video to audio conversion: {e}")
        raise e

# Initialize the T5 tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

def summarize_text(text):
    """Summarize the given text using the T5 model."""
    max_chunk_size = 512  # T5 works best with chunks of 512 tokens or fewer
    text = "summarize: " + text  # T5 uses a prompt-based approach; for summarization, prepend "summarize:"

    # Tokenize the input
    inputs = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_chunk_size)

    summaries = []
    for i in range(0, len(inputs[0]), max_chunk_size):
        chunk = inputs[0][i:i + max_chunk_size].unsqueeze(0)

        try:
            # Dynamically set max_length based on input length
            input_length = chunk.size(1)
            max_length = min(200, input_length // 2)  # Set max_length to half the input length, but no more than 200 tokens

            # Generate the summary
            summary_ids = model.generate(chunk, max_length=max_length, min_length=10, length_penalty=2.0, num_beams=4, early_stopping=True)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)

        except Exception as e:
            print(f"Error summarizing text: {e}")
            return None

    return " ".join(summaries)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    try:
        data = request.get_json()

        # Extract media data and type (audio/video)
        media_data = base64.b64decode(data['media'])
        media_type = data['type']

        if media_type == 'video':
            audio_data = convert_video_to_audio(media_data)
        else:
            audio_data = media_data

        # Send audio to Rev.ai API for transcription
        transcript_response = requests.post(
            'https://api.rev.ai/revtranscribe/v1beta1/transcript',
            headers={'Authorization': f'Bearer {REV_AI_ACCESS_TOKEN}'},
            files={'media': ('audio.wav', audio_data, 'audio/wav')}
        )

        if transcript_response.status_code != 200:
            return jsonify({'error': 'Error with transcription service.'})

        # Get transcription result
        transcript = transcript_response.json()
        transcript_text = transcript.get('text', '')

        # Summarize the transcription using T5 model
        summary_text = summarize_text(transcript_text)

        return jsonify({
            'transcription': transcript_text,
            'summary': summary_text or 'Summary not available.'
        })

    except Exception as e:
        print(f"Error in transcription process: {e}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/download')
def download():
    """Provide a download link for the transcription summary as a text file."""
    try:
        transcription = 'Transcription text goes here'
        summary = 'Summary text goes here'

        with open('transcription_summary.txt', 'w') as f:
            f.write(f'Transcription:\n{transcription}\n\nSummary:\n{summary}')

        return send_from_directory(os.getcwd(), 'transcription_summary.txt', as_attachment=True)

    except Exception as e:
        return jsonify({'error': f'Error downloading transcription: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=3000)
