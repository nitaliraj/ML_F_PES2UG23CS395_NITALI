# lab.py
import torch

def _entropy_from_labels(labels: torch.Tensor) -> float:
    # labels: 1-D tensor of class ids (integers)
    values, counts = torch.unique(labels, return_counts=True)
    probs = counts.double() / counts.sum().double()
    entropy = -torch.sum(probs * torch.log2(probs)).item()
    return float(entropy)

def get_entropy_of_dataset(tensor: torch.Tensor):
    """
    Entropy(S) = -Σ p(c) log2 p(c); last column is the target.
    """
    if tensor.numel() == 0:
        return 0.0
    labels = tensor[:, -1]
    return _entropy_from_labels(labels)

def get_avg_info_of_attribute(tensor: torch.Tensor, attribute: int):
    """
    Avg_Info(A) = Σ_v (|S_v|/|S|) * Entropy(S_v)
    """
    total = tensor.shape[0]
    col = tensor[:, attribute]
    values, counts = torch.unique(col, return_counts=True)

    avg_info = 0.0
    for v, cnt in zip(values, counts):
        mask = col == v
        subset_labels = tensor[mask][:, -1]
        weight = float(cnt.item()) / float(total)
        avg_info += weight * _entropy_from_labels(subset_labels)
    return float(avg_info)

def get_information_gain(tensor: torch.Tensor, attribute: int):
    """
    IG(S, A) = Entropy(S) - Avg_Info(A); return rounded to 4 decimals.
    """
    ds_entropy = get_entropy_of_dataset(tensor)
    avg_info = get_avg_info_of_attribute(tensor, attribute)
    ig = ds_entropy - avg_info
    ig = 0.0 if abs(ig) < 1e-12 else ig  # clean up -0.0
    return float(round(ig, 4))

def get_selected_attribute(tensor: torch.Tensor):
    """
    Returns: ( {attr_idx: IG, ...}, best_attr_idx )
    """
    n_features = tensor.shape[1] - 1  # exclude target
    info_gains = {}
    for i in range(n_features):
        info_gains[i] = get_information_gain(tensor, i)
    # tie-break: highest IG; if tie, smaller index wins (stable)
    best_attr = max(info_gains, key=lambda k: (info_gains[k], -k))
    return info_gains, best_attr
