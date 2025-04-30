
import os
import json
from cnn_model import model 


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Full path to the JSON file
DB_JSON = 'app/magic_stuff/The-Pocket-Pharmacist DB.json'

# Now open it



def search_by_label(data, label):
    with open(DB_JSON, 'r') as db_file:
        data = json.load(db_file)  
        for item in data:
            if item['label'] == label:
                return item
        return None
    
def identify_medication():
    return model.main()

def medication_info(med_name):
    medication = search_by_label(DB_JSON, med_name)

    if medication is not None:
        return json.dumps(medication)

    return None

 #print(search_by_label(DB_JSON, 'Aspirin'))
