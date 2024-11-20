const mediaFileInput = document.getElementById('mediaFile');
const uploadBtn = document.getElementById('uploadBtn');
const transcriptionEl = document.getElementById('transcription');
const summaryEl = document.getElementById('summary');
const audioOption = document.getElementById('audioOption');
const videoOption = document.getElementById('videoOption');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const downloadBtn = document.getElementById('downloadBtn');

let mediaRecorder;
let audioChunks = [];
let isAudio = true;

// Function to check selected media type and set MIME type
function getMediaConstraints() {
    isAudio = audioOption.checked;
    return isAudio ? { audio: true, video: false } : { audio: true, video: true };
}

// Function to request microphone and camera permissions
async function requestPermissions() {
    const constraints = getMediaConstraints();
    try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        return stream;
    } catch (err) {
        console.error('Permission denied:', err);
        alert(`Could not access microphone/camera: ${err.message}`);
        return null;
    }
}

startBtn.addEventListener('click', async () => {
    const stream = await requestPermissions();
    if (stream) {
        // Set MIME type based on audio or video
        const mimeType = isAudio ? 'audio/webm' : 'video/webm';
        audioChunks = []; // Clear previous recordings

        // Initialize MediaRecorder with the correct MIME type
        mediaRecorder = new MediaRecorder(stream, { mimeType });
        mediaRecorder.ondataavailable = event => audioChunks.push(event.data);

        mediaRecorder.onstop = async () => {
            const mediaBlob = new Blob(audioChunks, { type: mimeType });
            const reader = new FileReader();
            reader.onload = function(event) {
                const base64Media = event.target.result.split(',')[1];
                const mediaType = isAudio ? 'audio' : 'video';

                fetch('http://127.0.0.1:3000/api/transcribe', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ media: base64Media, type: mediaType })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        transcriptionEl.textContent = `Error: ${data.error}`;
                    } else {
                        transcriptionEl.textContent = `Transcription: ${data.transcription}`;
                        summaryEl.textContent = `Summary: ${data.summary}`;
                        downloadBtn.disabled = false;
                    }
                })
                .catch(error => {
                    transcriptionEl.textContent = 'Error uploading media.';
                    console.error('Error:', error);
                });
            };
            reader.readAsDataURL(mediaBlob);
        };

        mediaRecorder.start();
        startBtn.disabled = true;
        stopBtn.disabled = false;
    }
});

stopBtn.addEventListener('click', () => {
    mediaRecorder.stop();
    startBtn.disabled = false;
    stopBtn.disabled = true;
});

uploadBtn.addEventListener('click', () => {
    const file = mediaFileInput.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(event) {
            const base64Media = event.target.result.split(',')[1];
            const mediaType = audioOption.checked ? 'audio' : 'video';

            fetch('http://127.0.0.1:3000/api/transcribe', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ media: base64Media, type: mediaType })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    transcriptionEl.textContent = `Error: ${data.error}`;
                } else {
                    transcriptionEl.textContent = `Transcription: ${data.transcription}`;
                    summaryEl.textContent = `Summary: ${data.summary}`;
                    downloadBtn.disabled = false; // Enable download button after success
                }
            })
            .catch(error => {
                transcriptionEl.textContent = 'Error uploading media.';
                console.error('Error:', error);
            });
        };
        reader.readAsDataURL(file);
    } else {
        alert('Please select an audio or video file.');
    }
});

downloadBtn.addEventListener('click', () => {
    fetch('http://127.0.0.1:3000/api/download')
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to download results.');
        }
        return response.blob();
    })
    .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = 'transcription_summary.txt';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    })
    .catch(error => console.error('Error:', error));
});
