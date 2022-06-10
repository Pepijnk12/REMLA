# Multilabel classification on Stack Overflow tags
Predict tags for posts from StackOverflow with multilabel classification approach.


## Kubernetes instruction 

To initialize the cluster we have to: 

1. Build dockerfile within src/redirecting_service and name this build: remla-redirecting-service:latest (later we will replace with online build version but for testing purposes this is easier).
2. Run `kubectl apply -f .\k8s-local-deployment.yaml`

Raw input for postman post: /deploy-image
{
    "version": "1.4.0"
}


To remove old deployments: `kubectl delete all --all` 







## Dataset
- Dataset of post titles from StackOverflow

## Transforming text to a vector
- Transformed text data to numeric vectors using bag-of-words and TF-IDF.

## MultiLabel classifier
[MultiLabelBinarizer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html) to transform labels in a binary form and the prediction will be a mask of 0s and 1s.

[Logistic Regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) for Multilabel classification
- Coefficient = 10
- L2-regularization technique

## Evaluation
Results evaluated using several classification metrics:
- [Accuracy](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
- [F1-score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
- [Area under ROC-curve](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
- [Area under precision-recall curve](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score)

## Libraries
- [Numpy](http://www.numpy.org/) — a package for scientific computing.
- [Pandas](https://pandas.pydata.org/) — a library providing high-performance, easy-to-use data structures and data analysis tools for the Python
- [scikit-learn](http://scikit-learn.org/stable/index.html) — a tool for data mining and data analysis.
- [NLTK](http://www.nltk.org/) — a platform to work with natural language.

## DVC
Everything in the ```data/``` directory is tracked by DVC.

## Docker
Dockerfiles are found in the docker folder. Note that the build context should be the root project folder, and not the folder the dockerfile is contained in.

To build the inference API:
`docker build -f docker/inference-api/Dockerfile -t inference-api .`

To build the redirecting service:
`docker build -f docker/redirecting-service/Dockerfile -t redirecting-service .`

<hr>
Note: this sample project was originally created by @partoftheorigin
