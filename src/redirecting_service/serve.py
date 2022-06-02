"""
Flask API for the redirection service.
"""
from flask import Flask
from flasgger import Swagger

app = Flask(__name__)
swagger = Swagger(app)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Redirect prediction call to the inference APIs
    """
    return

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
