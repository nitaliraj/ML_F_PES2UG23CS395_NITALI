import numpy as np
from collections import Counter

# Function 1: Entropy of dataset
def get_entropy_of_dataset(data: np.ndarray) -> float:
    """
    Calculate entropy of the dataset based on target column (last column).
    """
    target = data[:, -1]  # last column is target
    values, counts = np.unique(target, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))
    return float(entropy)

# Function 2: Average information of attribute
def get_avg_info_of_attribute(data: np.ndarray, attribute: int) -> float:
    """
    Calculate weighted average entropy after splitting by a given attribute.
    """
    total_len = data.shape[0]
    values, counts = np.unique(data[:, attribute], return_counts=True)
    avg_info = 0.0

    for v, count in zip(values, counts):
        subset = data[data[:, attribute] == v]
        subset_entropy = get_entropy_of_dataset(subset)
        weight = count / total_len
        avg_info += weight * subset_entropy

    return float(avg_info)

# Function 3: Information gain
def get_information_gain(data: np.ndarray, attribute: int) -> float:
    """
    Information Gain = Entropy(S) - Avg_Info(attribute)
    """
    dataset_entropy = get_entropy_of_dataset(data)
    avg_info = get_avg_info_of_attribute(data, attribute)
    info_gain = dataset_entropy - avg_info
    return round(float(info_gain), 4)

# Function 4: Select best attribute
def get_selected_attribute(data: np.ndarray) -> tuple:
    """
    Return a dictionary of information gains and the attribute with the highest IG.
    """
    num_attributes = data.shape[1] - 1  # exclude target column
    info_gains = {}

    for attr in range(num_attributes):
        ig = get_information_gain(data, attr)
        info_gains[attr] = ig

    best_attr = max(info_gains, key=info_gains.get)
    return info_gains, best_attr
