import json
from spnflow.structure.node import Sum, Mul, assign_ids
from spnflow.structure.leaf import Leaf, Bernoulli, Multinomial, Poisson, Isotonic, Uniform, Gaussian, Gamma


DistributionMapper = {
    'Bernoulli': Bernoulli,
    'Multinomial': Multinomial,
    'Poisson': Poisson,
    'Isotonic': Isotonic,
    'Uniform': Uniform,
    'Gaussian': Gaussian,
    'Gamma': Gamma
}


def save_json(root, filename):
    """
    Save an SPN to a JSON file.

    :param root: The root of the SPN.
    :param filename: The filename of the output JSON file.
    """
    with open(filename, 'w') as file:
        file.write(json.dumps(root, default=json_spn_obj))


def load_json(filename):
    """
    Load a SPN from a JSON file

    :param filename: The filename of the input JSON file.
    :return: The loaded SPN.
    """
    with open(filename, 'r') as file:
        return assign_ids(json_obj_spn(json.load(file)))


def json_spn_obj(node):
    """
    Convert a SPN to a JSON-serializable object.

    :param node: A node of a SPN.
    :return: A JSON-serializable object.
    """
    if isinstance(node, Sum):
        return {
            'kind': Sum.__name__,
            'scope': node.scope,
            'weights': node.weights,
            'children': [json_spn_obj(c) for c in node.children]
        }
    if isinstance(node, Mul):
        return {
            'kind': Mul.__name__,
            'scope': node.scope,
            'children': [json_spn_obj(c) for c in node.children]
        }
    if isinstance(node, Leaf):
        return {
            'kind': node.__class__.__name__,
            'scope': node.scope,
            'params': node.params_dict()
        }

    raise NotImplementedError("JSON serialization not implemented for node of type " + node.__class__.__name__)


def json_obj_spn(obj):
    """
    Convert a JSON-serializable object to a SPN.

    :param obj: The object to convert.
    :return: The converted SPN.
    """
    kind = obj['kind']
    scope = obj['scope']

    if kind == Sum.__name__:
        children = [json_obj_spn(o) for o in obj['children']]
        return Sum(obj['weights'], children, scope)
    if kind == Mul.__name__:
        children = [json_obj_spn(o) for o in obj['children']]
        return Mul(children, scope)
    if kind in DistributionMapper:
        return DistributionMapper[kind](scope, **obj['params'])

    raise NotImplementedError("JSON serialization not implemented for node of type " + kind)
