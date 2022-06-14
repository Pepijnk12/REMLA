MODEL_PREFIX = "stackoverflow-tag-pred-model-"


def pod_name_to_model_version(pod_name: str):
    """e.g stackoverflow-tag-pred-model-1-4-0-74cbcb4b6-68gbg"""
    version_list = pod_name[len(MODEL_PREFIX):].split("-")[:3]
    return '.'.join(version_list)
