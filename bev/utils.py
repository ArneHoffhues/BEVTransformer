import importlib

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if config is None:
        return None
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def compute_mem_size(module):
    mem_params = sum([param.nelement()*param.element_size() for param in module.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in module.buffers()])
    mem = mem_params + mem_bufs # in bytes
    mem = int(mem /1000 ** 2) # in mbytes
    return mem

