from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([
        data['CRIM'], data['ZN'], data['INDUS'], data['CHAS'], data['NOX'],
        data['RM'], data['AGE'], data['DIS'], data['RAD'], data['TAX'],
        data['PTRATIO'], data['B'], data['LSTAT']
    ]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
