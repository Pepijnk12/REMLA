"""
Flask API of the Stackoverflow Tag Prediction model.
"""
from flask import Flask, jsonify, request
from flasgger import Swagger
from joblib import load
from flask_cors import CORS
from sklearn.metrics import f1_score

app = Flask(__name__)
swagger = Swagger(app)
cors = CORS(app, resources={r"/*": {"origins": "http://localhost:*"}})

predictions = {}
feedbacks = {}


@app.route('/', methods=['GET'])
def running():
    """
    Test to see if running
    """
    return jsonify(success=True)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict what tags are to be used for a Stackoverflow post.
    ---
    consumes:
      - application/json
    parameters:
        - name: input_data
          in: body
          description: message to be classified.
          required: True
          schema:
            type: object
            required: post
            properties:
                post:
                    type: string
                    example: This is a post.
    responses:
      200:
        description: "The result of the classification"
    """
    input_data = request.get_json(force=True)
    post = input_data.get('post')

    # Load preprocessors
    tfidf_preprocessor = load('preprocessors/tfidf_preprocessor.joblib')
    bow_preprocessor = load('preprocessors/bow_preprocessor.joblib')

    # Transform data
    tfidf_processed_post = tfidf_preprocessor.transform([post])
    bow_preprocessed_post = bow_preprocessor.transform([post])

    # Load model
    tfidf_model = load('models/model_tfidf.joblib')
    bow_model = load('models/model_mybag.joblib')

    tfidf_prediction = tfidf_model.predict(tfidf_processed_post)
    bow_prediction = bow_model.predict(bow_preprocessed_post)

    mlb = load('models/mlb.joblib')
    tfidf_tags = mlb.inverse_transform(tfidf_prediction)
    bow_tags = mlb.inverse_transform(bow_prediction)

    res = {
        "result": tfidf_tags[0],
        "classifier": "tfifd",
        "post": post,
        "tfidf_results": tfidf_tags[0],
        "bow_results": bow_tags[0]
    }

    return jsonify(res)

@app.route('/feedback', methods=['POST'])
def calculate_f1():
    """
    Calculate how accurate the predictions were
    """
    input_data = request.get_json(force=True)
    feedback = input_data.get('feedback')
    id = input_data.get('id')

    mlb = load('models/mlb.joblib')
    feedback_mlb = mlb.transform(feedback)
    feedbacks[id] = feedback_mlb

    # Get list of predictions and ground truth
    Y = [(b, feedbacks[a]) for a, b in predictions.items() if a in feedbacks]
    predicted, groundtruth = map(list, zip(*Y))

    f1 = f1_score(groundtruth, predicted, average='weighted')

    res = {
        "score": f1,
    }

    return jsonify(res)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
