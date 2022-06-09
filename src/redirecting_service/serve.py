"""
Flask API for the redirection service.
"""
import datetime
import requests
from flask import Flask, jsonify, request, render_template
from flasgger import Swagger
from flask_cors import CORS
from kubernetes import client, config

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
    response = "# HELP num_pred Number of request predictions\n"
    response = response.join("# TYPE num_pred counter\n")
    response = response.join("num_pred ").join(str(len(logs))).join("\n\n")

    # TODO accuracy of models

    response.mimetype = "text/plain"
    return response

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
