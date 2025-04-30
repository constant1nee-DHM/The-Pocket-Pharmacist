const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const screenshotContainer = document.getElementById('screenshotContainer');
const uploadBtn = document.getElementById('uploadBtn');
const message = document.getElementById('message');
let uploadedFile = null; 
let useScreenshotOverFile = 1;
let isScreenshotTaken = false; 

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

    uploadedFile = this.files[0];

    useScreenshotOverFile = 0;

    video.style.display = 'none';
    document.getElementById('screenshotBtn').style.display = 'none';

    uploadBtn.style.display = 'block';
});

document.getElementById('screenshotBtn').addEventListener('click', () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0);
    
    video.style.display = 'none';

    const img = document.createElement('img');
    img.src = canvas.toDataURL('image/png');
    screenshotContainer.innerHTML = ''; 
    screenshotContainer.appendChild(img);

    useScreenshotOverFile = 1;
    
    isScreenshotTaken = true; 
    uploadBtn.style.display = 'inline-block'; 
});

uploadBtn.addEventListener('click', async () => {
    let fileToUpload;


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
    formData.append('file', fileToUpload); 

    const loadingIndicator = document.getElementById('loadingIndicator');
    loadingIndicator.style.display = 'block'; 
    try {
        const response = await fetch('http://127.0.0.1:5000/api/upload', {
            method: 'POST',
            body: formData,
        });
        const result = await response.json();
     

    
        loadingIndicator.style.display = 'none'; 

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
        loadingIndicator.style.display = 'none'; 
    }
});


const popup = document.getElementById('popup');
const closePopup = document.getElementById('closePopup');

function showPopup() {
    popup.style.display = 'block';
}

function hidePopup() {
    popup.style.display = 'none';
}

closePopup.addEventListener('click', hidePopup);


window.addEventListener('click', (event) => {
    if (event.target === popup) {
        hidePopup();
    }
});


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


closePopup.addEventListener('click', () => {
    hidePopup();
    resetApp(); 
});
