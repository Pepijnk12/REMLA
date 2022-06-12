"""
Flask API for the redirection service.
"""
import datetime
import requests
from flask import Flask, jsonify, request, render_template
from flasgger import Swagger
from flask_cors import CORS
from kubernetes import client, config
from prometheus_client import generate_latest, Counter, Gauge

config.load_incluster_config()

app = Flask(__name__)
swagger = Swagger(app)

cors = CORS(app, resources={r"/*": {"origins": "http://remla.localdev.me:*"}})

state = {
    "active_model": "None",
}

posts = []
models = set()


MODEL_PREFIX = "stackoverflow-tag-pred-model-"

def pod_name_to_model_version(pod_name: str):
    """e.g stackoverflow-tag-pred-model-1-4-0-74cbcb4b6-68gbg"""
    version_list = pod_name[len(MODEL_PREFIX):].split("-")[:3]
    return '.'.join(version_list)

classesA = Counter('classes_predicted_A', 'The amount of times each class has been predicted by A', ['class'])
classesB = Counter('classes_predicted_B', 'The amount of times each class has been predicted by B', ['class'])
amountRequests = Counter('amount_requests', 'Amount of requests for predictions')

f1A = Gauge('f1_A', 'Cumulative f1 score of A')
f1B = Gauge('f1_B', 'Cumulative f1 score of B')

logs = []

id = 0

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

    input_data = request.get_json(force=True)

    print(input_data)

    version = input_data.get('version')

    version_safe = version.replace(".", "-")

    container = client.V1Container(
        name="stackoverflow-tag-pred-model-" + version_safe,
        image="ghcr.io/pepijnk12/remla:inference-api-" + version,
        ports=[client.V1ContainerPort(container_port=8000)],
        resources=client.V1ResourceRequirements(
            requests={"cpu": "100m", "memory": "200Mi"},
            limits={"cpu": "500m", "memory": "500Mi"},
        ),
    )

    # Create and configure a spec section
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"app": "stackoverflow-tag-pred-model"}),
        spec=client.V1PodSpec(containers=[container]),
    )

    # Create the specification of deployment
    spec = client.V1DeploymentSpec(
        replicas=1, template=template, selector={
            "matchLabels":
                {"app": "stackoverflow-tag-pred-model"}})

    # Instantiate the deployment object
    deployment = client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=client.V1ObjectMeta(name="stackoverflow-tag-pred-model-" + version_safe),
        spec=spec,
    )

    k8s_apps_v1 = client.AppsV1Api()
    k8s_apps_v1.create_namespaced_deployment(
        body=deployment, namespace="default"
    )

    k8s_apps_v1 = client.CoreV1Api()
    body = client.V1Service(
        api_version="v1",
        kind="Service",
        metadata=client.V1ObjectMeta(
            name="stackoverflow-tag-pred-model-" + version_safe + "-service"
        ),
        spec=client.V1ServiceSpec(
            selector={"app": "stackoverflow-tag-pred-model"},
            ports=[client.V1ServicePort(
                port=8000,
                target_port=8000
            )]
        )
    )
    # Creation of the Deployment in specified namespace
    # (Can replace "default" with a namespace you may have created)
    k8s_apps_v1.create_namespaced_service(namespace="default", body=body)

    # replace container-name-replace-me in k8s deployment
    # Read in the file

    # Replace the target string
    # filedata = filedata.replace('container-name-replace-me', 'stackoverflow-tag-pred-model-1')

    # dep = yaml.load(file,Loader=yaml.FullLoader)

    # TODO do something with image url
    # input_data = request.get_json(force=True)
    # image_url = input_data.get('imageUrl')

    # TODO if the model has been deployed before do not deploy
    return jsonify(success=True)


@app.route('/get-all-models', methods=['GET'])
def get_all_models():
    """
    Get all model services running in the cluster
    """
    v1 = client.CoreV1Api()
    print("Listing pods with their IPs:")
    ret = v1.list_pod_for_all_namespaces(watch=False)
    for i in ret.items:
        if i.metadata.name.startswith("stackoverflow-tag-pred-model"):
            models.add(pod_name_to_model_version(i.metadata.name))
    return jsonify(models=list(models))


@app.route('/predict', methods=['POST'])
def predict():
    """
    Redirect prediction call to the inference APIs
    """
    # @todo do sth with the stored shadow model responses
    active_model_res = {}
    shadow_models_res = {}

    input_data = request.get_json(force=True)
    post = input_data.get('post')

    if not post:
        return jsonify(success=False)

    if state['active_model'] == 'None':
        return jsonify(success=False, message="No active model")

    for model in models:
        safe_version = model.replace('.', '-')
        response = requests.post("http://stackoverflow-tag-pred-model-" + safe_version + "-service:8000/predict", json={
            "post": post
        })
        if model == state['active_model']:
            active_model_res = {
                'result': response.json()['result'],
                'active-model': model
            }
        else:
            shadow_models_res.update({
                model: {
                    'result': response.json()['result']
                }
            })

    # # Redirect request to both inference APIs
    # resA = requests.post("http://0.0.0.0:30001/predict", json={
    #         "post": post
    #     })
    #
    # resB = requests.post("http://0.0.0.0:30002/predict", json={
    #         "post": post
    #     })
    # jsonA = resA.json()
    # jsonB = resB.json()
    #
    # # Increase the counts for each tag per label
    # for tag in jsonA['result']:
    #     classesA.labels(tag).inc()
    # for tag in jsonB['result']:
    #     classesB.labels(tag.inc())
    #
    # # Increase the request count
    # amountRequests.inc()
    #
    # global id
    # id += 1
    #
    # for res_json in [jsonA, jsonB]:
    #     res_json['timestamp'] = str(datetime.datetime.now())
    #     logs.append(res_json)
    #
    #
    # res = {
    #     "A": resA.json()['result'],
    #     "B": resB.json()['result'],
    #     "active_model": state["active_model"],
    #     "requestId": id
    # }
    #
    # res['timestamp'] = str(datetime.datetime.now())
    # return res

    active_model_res['timestamp'] = str(datetime.datetime.now())
    return active_model_res


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

    requestId = input_data.get('requestId')

    # Get f1 scores of both inference APIs
    resA = requests.post("http://0.0.0.0:30001/feedback", json={
            "feedback": user_tags,
            "id": requestId
        })

    resB = requests.post("http://0.0.0.0:30002/feedback", json={
            "feedback": user_tags,
            "id": requestId
        })

    jsonA = resA.json()
    jsonB = resB.json()

    f1A.set(jsonA['score'])
    f1B.set(jsonB['score'])
    return jsonify(success=True)


@app.route('/set-active-model', methods=['POST'])
def set_active_model():
    """
    Sets the current active model
    """
    global posts
    posts = []

    input_data = request.get_json(force=True)
    model = input_data.get('model')
    if model in models:
        state['active_model'] = model
        return jsonify(success=True)
    return jsonify(success=False)


@app.route('/metrics', methods=['GET'])
def metrics():
    """
    Metrics
    """
    return generate_latest()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
