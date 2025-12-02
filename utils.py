def set_seed(seed: int = 2025):
    import os, random, numpy as np, torch

    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # strict deterministic behavior (PyTorch ≥ 1.9)
    try:
        torch.use_deterministic_algorithms(True)
    except:
        pass
