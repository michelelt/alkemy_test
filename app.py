from flask import Flask, request, jsonify
import joblib

# Inizializza l'applicazione Flask
app = Flask(__name__)

# Carica il modello dal file
model = joblib.load('./model/best_model.pkl')
magic_number = joblib.load('./model/magic_number.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    try:
        test_data = [data[feature] for feature in model.feature_names_in_]
    except KeyError as e:
        return jsonify({'error': f'Missing feature in the input data: {str(e)}'}), 400

    prediction = model.predict([test_data])

    # Invia la risposta
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
