# The Pocket Pharmacist

The Pocket Pharmacist is a web application that leverages machine learning to identify images of medications uploaded by users. The app provides relevant information about the medication presented in a user-friendly interface.

## Features

- **Image Recognition**: Utilizes machine learning algorithms to accurately identify medications from user-uploaded images.
- **Medication Information**: Provides detailed information about identified medications from an existing dataset.
- **User-Friendly Interface**: Designed with a clean and intuitive UI for easy navigation and interaction.
- **Camera Functionality**: Allows users to take and upload photos of their medication for easy identification.
- **File Upload**: Users can upload images directly from their device for identification.

## Technologies Used

- **Frontend**: HTML, CSS, JavaScript
- **Machine Learning**: Convolutional Neural Network, MLP, Pytorch, NumPy
- **Backend**: Flask

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
   ```

## Get Started

1. Take a screenshot or upload an image using the buttons in the UI.
2. Click on `Upload Image` and wait for the system to identify the medicine.
3. View your medication information.
