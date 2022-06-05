"""
Flask API for the redirection service.
"""
import datetime
import requests
from flask import Flask, jsonify, request
from flasgger import Swagger
from flask_cors import CORS

app = Flask(__name__)
swagger = Swagger(app)
cors = CORS(app, resources={r"/*": {"origins": "http://localhost:*"}})


state = {
    "active_model": "A"
}

logs = []

@app.route('/', methods=['GET'])
def running():
    """
    Test to see if running
    """
    return jsonify(success=True)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Redirect prediction call to the inference APIs
    """
    input_data = request.get_json(force=True)
    post = input_data.get('post')
    if not post:
        return jsonify(success=False)

    # Redirect request to both inference APIs
    if state['active_model'] == 'A':
        res = requests.post("http://0.0.0.0:8000/predict", json={
            "post": post
        })
    else:
        res = requests.post("http://0.0.0.0:8000/predict", json={
            "post": post
        })

    log_item = res.json()
    log_item['timestamp'] = str(datetime.datetime.now())
    logs.append(log_item)
    return res.json()

@app.route('/active-model', methods=['GET'])
def get_active_model():
    """
    Returns the current model that is active
    """
    return jsonify({
        "activeModel": state['active_model']
    })

@app.route('/logs', methods=['GET'])
def get_logs():
    """
    Returns the current model that is active
    """
    print(logs)
    return jsonify(logs)

@app.route('/set-active-model', methods=['POST'])
def set_active_model():
    """
    Sets the current active model
    """
    input_data = request.get_json(force=True)
    model = input_data.get('model')
    if model in ['A', 'B']:
        state['active_model'] = model
        return jsonify(success=True)
    return jsonify(success=False)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
