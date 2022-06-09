"""
Flask API for the redirection service.
"""
import datetime
import requests
from flask import Flask, jsonify, request, render_template
from flasgger import Swagger
from flask_cors import CORS

app = Flask(__name__)
swagger = Swagger(app)

cors = CORS(app, resources={r"/*": {"origins": "http://localhost:*"}})

state = {
    "active_model": "A",
}

posts = []


@app.route('/', methods=['GET'])
def index_page():
    """
    Render index page
    """
    return render_template("index.html")

@app.route('/admin', methods=['GET'])
def admin_view():
    """
    Render admin page
    """
    return render_template("admin.html")


@app.route('/deploy-image', methods=['POST'])
def deploy_image():
    """
    Deploy image to Kubernetes cluster
    """
    # TODO do something with image url
    # input_data = request.get_json(force=True)
    # image_url = input_data.get('imageUrl')
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
    resA = requests.post("http://0.0.0.0:30001/predict", json={
        "post": post
    })
    resB = requests.post("http://0.0.0.0:30002/predict", json={
        "post": post
    })

    res = {
        "A": resA.json()['result'],
        "B": resB.json()['result'],
        "active_model": state["active_model"]
    }

    res['timestamp'] = str(datetime.datetime.now())
    return res

@app.route('/metrics-active-model', methods=['GET'])
def metrics_active_model():
    """
    Returns active model metrics
    """
    return str(0.05)


@app.route('/metrics-inactive-model', methods=['GET'])
def metrics_inactive_model():
    """
    Returns inactive model metrics
    """
    return str(0.05)


@app.route('/active-model', methods=['GET'])
def get_active_model():
    """
    Returns the current model that is active
    """
    return jsonify({
        "activeModel": state['active_model']
    })


@app.route('/logs', methods=['GET'])
def get_posts():
    """
    Returns the current model that is active
    """
    return jsonify(posts)

@app.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    """
    Returns the current model that is active
    """
    input_data = request.get_json(force=True)
    user_tags = input_data.get('feedback')
    results = input_data.get('results')
    results['user_tags'] = user_tags
    posts.append(results)
    return jsonify(success=True)


@app.route('/set-active-model', methods=['POST'])
def set_active_model():
    """
    Sets the current active model
    """
    global logs
    logs = []

    input_data = request.get_json(force=True)
    model = input_data.get('model')
    if model in ['A', 'B']:
        state['active_model'] = model
        return jsonify(success=True)
    return jsonify(success=False)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
