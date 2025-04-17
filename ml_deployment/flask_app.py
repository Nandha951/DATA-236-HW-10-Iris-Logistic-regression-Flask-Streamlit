import flask
from flask import request, jsonify
import pickle
import numpy as np

app = flask.Flask(__name__)
app.config["DEBUG"] = True

# Load the model
try:
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    print(f"Error loading the model: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        # Validate input data
        if not all(key in data for key in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']):
            return jsonify({'error': 'Invalid input: Missing features'}), 400

        sepal_length = data['sepal_length']
        sepal_width = data['sepal_width']
        petal_length = data['petal_length']
        petal_width = data['petal_width']

        if not all(isinstance(value, (int, float)) for value in [sepal_length, sepal_width, petal_length, petal_width]):
            return jsonify({'error': 'Invalid input: Features must be numbers'}), 400

        # Prepare input for the model
        input_array = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Make prediction
        prediction = model.predict(input_array)[0]

        # Map the prediction to the species name
        species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        species_name = species_mapping.get(prediction, 'unknown')

        # Return the prediction as JSON
        return jsonify({'prediction': species_name})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)