document.getElementById('requestPermissionsBtn').addEventListener('click', async () => {
    const constraints = {
        audio: true,  // Request microphone access
        video: true   // Request camera access
    };

    try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        
        if (stream) {
            // Stop the media stream to release the camera/microphone
            stream.getTracks().forEach(track => track.stop());
            window.location.href = 'index.html'; // Redirect if permission is granted
        }
    } catch (err) {
        console.error('Permission denied for microphone/camera:', err);
        document.getElementById('message').textContent = `Could not access microphone/camera: ${err.message}`;
        // Add additional logging or feedback to help debug
    }    
});
