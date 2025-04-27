import time
import json

# from "../../cnn_model/model.py" import infer

DB_JSON = "app/The-Pocket-Pharmacist DB.json"

with open(DB_JSON, 'r') as db_file:
    db = json.load(db_file)

# Function to search for a label in the list of dictionaries
def search_by_label(data, label):
    for item in data:
        if item.get('label') == label:
            return item  # Return the entire item if the label matches
    return None  # Return None if the label is not found

def identify_medication(img):
    # dummy delay
    time.sleep(1)

    # run classifier on image...

    # inference code called here

    return "Fentanyl" # "Oxycontin" ... etc

def medication_info(med_name):
    medication = search_by_label(db, med_name)

    if medication is not None:
        return json.dumps(medication)

    return None
