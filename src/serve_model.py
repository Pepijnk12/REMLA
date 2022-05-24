"""
Flask API of the Stackoverflow Tag Prediction model.
"""
import joblib
from flask import Flask, jsonify, request, render_template
from flasgger import Swagger

from text_preprocessing import text_prepare
app = Flask(__name__)
swagger = Swagger(app)


@app.route('/', methods = ['GET'])
def search():
    """View the main frontend web page"""
    return render_template('index.html')


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
    preprocessed_post = text_prepare(post)

    tfidf_vectorizer = joblib.load('output/tfidf_vectorizer.joblib')
    processed_post = tfidf_vectorizer.transform([preprocessed_post])

    model = joblib.load('output/model_tfidf.joblib')
    prediction = model.predict(processed_post)

    mlb = joblib.load('output/mlb.joblib')
    tags = mlb.inverse_transform(prediction)

    res = {
        "result": tags[0],
        "classifier": "tfidf",
        "post": post
    }

    return jsonify(res)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
