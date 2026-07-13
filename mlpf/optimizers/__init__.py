import torch
from mlpf.optimizers.lamb import Lamb
from mlpf.logger import _logger

def get_optimizer(model, config):
    # handle both dict and MLPFConfig
    if isinstance(config, dict):
        wd  = config.get("weight_decay", 0.01)
        lr  = config.get("lr", 0.0004)
        opt = config.get("optimizer", "adamw")
    else:
        wd  = config.weight_decay
        lr  = config.lr
        opt = config.optimizer.value if hasattr(config.optimizer, 'value') else config.optimizer

    if opt == "adamw":
        ret = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt == "lamb":
        ret = Lamb(model.parameters(), lr=lr, weight_decay=wd)
    elif opt == "sgd":
        ret = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer type: {opt}")

    _logger.info(f"Created optimizer: {ret}")
    return ret
