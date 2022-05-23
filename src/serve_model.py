"""
Flask API of the Stackoverflow Tag Prediction model.
"""
import joblib
from flask import Flask, jsonify, request, render_template
from flasgger import Swagger
from joblib import load

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
    
    # Load preprocessors
    tfidf_preprocessor = load('models/preprocessors/tfidf_preprocessor.joblib')
    bow_preprocessor = load('models/preprocessors/bow_preprocessor.joblib')
    
    # Transform data
    tfidf_processed_post = tfidf_preprocessor.transform([post])
<<<<<<< HEAD
    bow_preprocessed_post = bow_preprocessor.transform([post])

    # Load model
    tfidf_model = load('models/model_tfidf.joblib')
    bow_model = load('models/model_mybag.joblib')

    tfidf_prediction = tfidf_model.predict(tfidf_processed_post)
    bow_prediction = bow_model.predict(bow_preprocessed_post)

    mlb = load('models/mlb.joblib')
    tfidf_tags = mlb.inverse_transform(tfidf_prediction)
    bow_tags = mlb.inverse_transform(bow_prediction)

    # res = {
    #     "result": tags[0],
    #     "classifier": "tfidf",
    #     "post": post
    # }


    # How to programmatically (C#) import from the Xls-XML(xls file Saved in XML Format) data in to SQL Server	
    # ['c#', 'asp.net', '.net', 'sql-server', 'excel']

    # Android Eclipse How to display specific data from phpMySql database to list view	
    # ['php', 'android', 'mysql', 'json', 'eclipse']

=======

    # Load model
    model = joblib.load('models/model_tfidf.joblib')

    prediction = model.predict(tfidf_processed_post)

    mlb = joblib.load('models/mlb.joblib')
    tags = mlb.inverse_transform(prediction)
>>>>>>> 65406bb (refactor processing into pipelines)

    res = {
        "post": post,
        "tfidf_results": {
            "predicted_tags": tfidf_tags[0]
        },
        "bow_results": {
            "predicted_tags": bow_tags[0]
        }
    }

    return jsonify(res)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
