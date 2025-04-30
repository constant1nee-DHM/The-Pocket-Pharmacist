# The Pocket Pharmacist

The Pocket Pharmacist is a web application that leverages machine learning to identify images of medications uploaded by users. The app provides relevant information about the medication presented in a user-friendly interface.

## Features

- **Image Recognition**: Utilizes machine learning algorithms to accurately identify medications from user-uploaded images.
- **Medication Information**: Provides detailed information about identified medications from an existing dataset.
- **User-Friendly Interface**: Designed with a clean and intuitive UI for easy navigation and interaction.
- **Camera Functionality**: Allows users to take and upload photos of their medication for easy identification.
- **File Upload**: Users can upload images directly from their device for identification.


## Identifiable Medications

A-ferin, Apodorm, Apronax, Arveles, Aspirin, Dikloron, Dolcontin, Dolorex, Fentanyl, Hametan, Imovane, Majezik, Metpamid, Midazolam B. Braun, Morphin, Nobligan Retard, Oxycontin, Oxynorm, Parol, Sobril, Terbisil, Ultiva, Unisom, Valium Diazepam, and Xanor.


## Technologies Used

- **Frontend**: HTML, CSS, JavaScript
- **Machine Learning**: Convolutional Neural Network, MLP, Pytorch, NumPy
- **Backend**: Flask

## Dataset

This project uses a custom medication image dataset created through a combination of our own collected images and carefully curated selections from existing sources:

- **Base Framework**: Initially organized using Roboflow.com (March 17, 2025)
- **Customization**: We significantly enhanced the dataset by:
  - Adding our own originally captured medication images
  - Carefully selecting and curating only the most relevant and high-quality images from existing sources
  - Ensuring diverse representation of medications to improve model accuracy
- **Final Size**: 750 images after curation and additions
- **Format**: Folder-based annotation structure
- **Pre-processing**: 
  - Auto-orientation of pixel data (with EXIF-orientation stripping)
  - Resize to 640x640 (Stretch)
- **Task Type**: Single-label Classification

### Dataset Development Process

Our dataset represents substantial original work beyond existing resources. We conducted a thorough selection process to identify the most useful and representative medication images, supplemented with our own photography to address gaps in coverage. This custom dataset was specifically tailored to achieve optimal performance for our use case.

### Data Privacy & Usage

All medication images used for training are intended for educational and identification purposes only. The model is designed to assist users in identifying medications but should not replace professional medical advice.

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/constant1nee-DHM/The-Pocket-Pharmacist
   cd the-pocket-pharmacist
   ```

2. **Install Dependencies**
   ```bash
   cd app
   pip install -r requirements.txt
   cd ..
   ```

2. **Start Development Server**
   ```bash
   py app/app.py
   or python app/app.py
   or python3 app/app.py
   (depending on your version)
   ```

## Get Started

1. Server should be running on http://127.0.0.1:5000
2. Take a screenshot or upload an image using the buttons in the UI.
3. Click on `Upload Image` and wait for the system to identify the medicine. (The first image usually takes 1.5 minutes as the model gets uploaded)
4. View your medication information.
