"""
Flask API of the Stackoverflow Tag Prediction model.
"""
from flask import Flask, jsonify, request, render_template
from flasgger import Swagger
from joblib import load

app = Flask(__name__)
swagger = Swagger(app)

@app.route('/predict', methods=['POST'])
def predict():


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
