"""
Flask API for the redirection service.
"""
from flask import Flask, jsonify
from flasgger import Swagger

app = Flask(__name__)
swagger = Swagger(app)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Redirect prediction call to the inference APIs
    """
    return

@app.route('/active-model', methods=['GET'])
def get_active_model():
    """
    Returns the current model that is aactive
    """
    return jsonify({
        "activeModel": "A"
    })

@app.route('/set-active-model', methods=['POST'])
def set_active_model():
    """
    Sets the current active model
    """
    # input_data = request.get_json(force=True)
    # model = input_data.get('model')
    return jsonify(success=True)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
