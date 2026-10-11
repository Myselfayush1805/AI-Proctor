import torch

def check_gpu(required=False):
    """Returns a torch.device. Falls back to CPU unless required=True."""
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    if required:
        raise RuntimeError("CUDA is not available. This project requires a GPU to run.")
    print("WARNING: no CUDA GPU found, running on CPU (slower).")
    return torch.device("cpu")