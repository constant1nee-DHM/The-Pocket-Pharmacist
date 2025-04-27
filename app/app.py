from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  # Import CORS
import os
import magic_stuff

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# Route to serve static files (CSS, JS, etc.)
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# Route to handle the upload
@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    filename = file.filename
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # Pass image to identification function (magic stuff using ML)
    medication_name = magic_stuff.identify_medication(file)

    return magic_stuff.medication_info(medication_name)

    # return """{"medicine": {
    #                     "label": "Ultiva",
    #                     "type": "Opioid analgesic",
    #                     "main_chemical_compound": "Remifentanil hydrochloride",
    #                     "country_of_manufacture": "UK/USA",
    #                     "prescription_required_uk": true,
    #                     "contraindications": [
    #                         "Hypersensitivity to remifentanil",
    #                         "Severe respiratory depression",
    #                         "Absence of resuscitation facilities"
    #                         ],
    #                     "safe_dose_adults": "0.1\\u20130.15 \\u00b5g/kg/min IV infusion",
    #                     "safe_dose_children": "Weight-based dosing under supervision",
    #                     "use_cases": "Pain control during surgery and intensive care"
    #                     }, "message": "File uploaded successfully!"}""", 200

if __name__ == '__main__':
    # Create the upload folder if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
