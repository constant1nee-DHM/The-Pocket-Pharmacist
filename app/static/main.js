const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const screenshotContainer = document.getElementById('screenshotContainer');
const uploadBtn = document.getElementById('uploadBtn');
const message = document.getElementById('message');

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => {
        console.error("Error accessing webcam: ", err);
        message.textContent = "Error accessing webcam.";
    });

document.getElementById('screenshotBtn').addEventListener('click', () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0);
    
    const img = document.createElement('img');
    img.src = canvas.toDataURL('image/png');
    screenshotContainer.innerHTML = ''; // Clear previous screenshot
    screenshotContainer.appendChild(img);
    
    uploadBtn.style.display = 'inline-block'; // Show upload button
});

uploadBtn.addEventListener('click', async () => {
    const dataUrl = canvas.toDataURL('image/png');
    const blob = await (await fetch(dataUrl)).blob();
    const formData = new FormData();
    formData.append('file', blob, 'screenshot.png');

    try {
        const response = await fetch('http://127.0.0.1:5000/api/upload', {
            method: 'POST',
            body: formData,
        });
        const result = await response.json();
        message.textContent = result.message || result.error;
    } catch (error) {
        console.error("Error uploading screenshot: ", error);
        message.textContent = "Error uploading screenshot.";
    }
});
