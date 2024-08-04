import requests
import json


url = 'http://localhost:5000/predict'

with open('./processed_data/test_string_json.json', 'r') as f:
    data = json.load(f)

# Effettua la richiesta POST
response = requests.post(url, json=data)

# Controlla la risposta
if response.status_code == 200:
    print('Predicted_flow', response.json())
else:
    print('Errore nella richiesta:', response.status_code, response.text)