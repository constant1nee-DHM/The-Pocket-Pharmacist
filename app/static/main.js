const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const screenshotContainer = document.getElementById('screenshotContainer');
const uploadBtn = document.getElementById('uploadBtn');
const message = document.getElementById('message');
let uploadedFile = null; // Variable to store the uploaded file
let useScreenshotOverFile = 1;
let isScreenshotTaken = false; // Variable to track if a screenshot has been taken

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => {
        console.error("Error accessing webcam: ", err);
        message.textContent = "Error accessing webcam.";
    });

document.getElementById('filePicker').addEventListener('change', function() {
    const fileName = this.files[0] ? this.files[0].name : 'No file chosen';
    document.getElementById('fileName').textContent = fileName;

    // Store the uploaded file
    uploadedFile = this.files[0];

    useScreenshotOverFile = 0;

    // Hide the video feed and screenshot button
    video.style.display = 'none';
    document.getElementById('screenshotBtn').style.display = 'none';

    // Show the upload button
    uploadBtn.style.display = 'block';
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

    useScreenshotOverFile = 1;
    
    isScreenshotTaken = true; // Set the flag to true
    uploadBtn.style.display = 'inline-block'; // Show upload button
});

uploadBtn.addEventListener('click', async () => {
    let fileToUpload;

    // Determine which file to upload
    if (useScreenshotOverFile) {
        const dataUrl = canvas.toDataURL('image/png');
        const blob = await (await fetch(dataUrl)).blob();
        fileToUpload = new File([blob], 'screenshot.png', { type: 'image/png' });
    } else if (uploadedFile) {
        fileToUpload = uploadedFile;
    } else {
        message.textContent = "No file selected for upload.";
        return;
    }

    const formData = new FormData();
    formData.append('file', fileToUpload); // Use the appropriate file

    // Show loading indicator
    const loadingIndicator = document.getElementById('loadingIndicator');
    loadingIndicator.style.display = 'block'; // Show loading indicator

    try {
        const response = await fetch('http://127.0.0.1:5000/api/upload', {
            method: 'POST',
            body: formData,
        });
        const result = await response.json();
        // message.textContent = result.message;

        // Hide loading indicator
        loadingIndicator.style.display = 'none'; // Hide loading indicator

        const medication = result;

        document.getElementById('medicationLabel').innerText = medication.label;
        document.getElementById('medicationType').innerText = medication.type;
        document.getElementById('mainChemicalCompound').innerText = medication.main_chemical_compound;
        document.getElementById('countryOfManufacture').innerText = medication.country_of_manufacture;
        document.getElementById('prescriptionRequired').innerText = medication.prescription_required_uk ? 'Yes' : 'No';
        document.getElementById('contraindications').innerText = medication.contraindications.join(', ');
        document.getElementById('safeDoseAdults').innerText = medication.safe_dose_adults;
        document.getElementById('safeDoseChildren').innerText = medication.safe_dose_children;
        document.getElementById('useCases').innerText = medication.use_cases;
    } catch (error) {
        console.error("Error uploading file: ", error);
        message.textContent = "Error uploading file.";
        loadingIndicator.style.display = 'none'; // Hide loading indicator
    }
});

// Popup functionality
const popup = document.getElementById('popup');
const closePopup = document.getElementById('closePopup');

// Function to show the popup
function showPopup() {
    popup.style.display = 'block';
}

// Function to hide the popup
function hidePopup() {
    popup.style.display = 'none';
}

// Event listener for closing the popup
closePopup.addEventListener('click', hidePopup);

// Close the popup when clicking outside of it
window.addEventListener('click', (event) => {
    if (event.target === popup) {
        hidePopup();
    }
});

// Optional: Add a function to reset the application state
function resetApp() {
    uploadedFile = null;
    useScreenshotOverFile = 1;
    isScreenshotTaken = false;
    video.style.display = 'block';
    document.getElementById('screenshotBtn').style.display = 'inline-block';
    uploadBtn.style.display = 'none';
    screenshotContainer.innerHTML = '';
    message.textContent = '';
}

uploadBtn.addEventListener('click', () => {
    showPopup();
});

// Call resetApp when the popup is closed (if needed)
closePopup.addEventListener('click', () => {
    hidePopup();
    resetApp(); // Reset the application state if desired
});
