from importlib import import_module


def make_object(fullname, params=None):
    """
    Args:
        fullname (str): full name of the object (ie. '.'.join([o.__module__, o.__name__]))
        params (dict|None): keyword arguments to initialize the object
    Returns:
        Object, instance of the class.
    """
    params = params or {}

    module_name, class_name = fullname.rsplit(".", 1)

    try:
        module = import_module(module_name)
    except ImportError:
        raise ImportError(f"Could not import module {module_name}")
    try:
        object_instance = getattr(module, class_name)(**params)
    except AttributeError:
        raise ValueError(f"Module {module_name} has no object {class_name}")
    return object_instance
