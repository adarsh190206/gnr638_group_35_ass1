import argparse
import pickle
import json
import os
import cv2
import time

from framework.model_rgb import SimpleCNN
from framework import backend
from dataset_rgb import ImageFolderDataset


# ==============================
# ARGUMENT PARSER
# ==============================

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True,
                    help="Path to test dataset")
parser.add_argument("--weights", type=str, required=True,
                    help="Path to saved weights file")

args = parser.parse_args()

DATA_ROOT = args.dataset
WEIGHTS_PATH = args.weights


# ==============================
# LOAD DATASET
# ==============================

print("Loading evaluation dataset...")

t0 = time.perf_counter()
dataset = ImageFolderDataset(DATA_ROOT)
t1 = time.perf_counter()

print(f"Dataset load time: {t1 - t0:.3f} seconds")

num_classes = len(dataset.class_to_idx)
print("Number of classes:", num_classes)

# ==============================
# DETERMINE INPUT SHAPE
# ==============================


CHANNELS = 3
H = 32
W = 32

# print(f"Input shape: ({CHANNELS}, {H}, {W})")

# ==============================
# BUILD MODEL
# ==============================

model = SimpleCNN(num_classes, in_channels=CHANNELS)

# ==============================
# LOAD WEIGHTS
# ==============================

with open(WEIGHTS_PATH, "rb") as f:
    weights = pickle.load(f)

for i, p in enumerate(model.parameters()):
    p.data = weights[f"param_{i}"]["data"]

print("Weights loaded successfully.")

# ==============================
# EFFICIENCY METRICS
# ==============================

def compute_model_metrics(model, input_size):

    H = input_size
    W = input_size

    total_params = 0
    total_macs = 0

    # Conv1
    F1, C1, K1, _ = model.conv1_w.shape
    outH1 = H - K1 + 1
    outW1 = W - K1 + 1

    params = F1 * C1 * K1 * K1 + F1
    macs = outH1 * outW1 * F1 * (C1 * K1 * K1)

    total_params += params
    total_macs += macs

    outH1 //= 2
    outW1 //= 2

    # Conv2
    F2, C2, K2, _ = model.conv2_w.shape
    outH2 = outH1 - K2 + 1
    outW2 = outW1 - K2 + 1

    params = F2 * C2 * K2 * K2 + F2
    macs = outH2 * outW2 * F2 * (C2 * K2 * K2)

    total_params += params
    total_macs += macs

    outH2 //= 2
    outW2 //= 2

    # FC1
    in_features = outH2 * outW2 * F2
    out_features = model.fc1_w.shape[1]

    params = in_features * out_features + out_features
    macs = in_features * out_features

    total_params += params
    total_macs += macs

    # FC2
    in_features = model.fc2_w.shape[0]
    out_features = model.fc2_w.shape[1]

    params = in_features * out_features + out_features
    macs = in_features * out_features

    total_params += params
    total_macs += macs

    total_flops = 2 * total_macs

    return total_params, total_macs, total_flops


params, macs, flops = compute_model_metrics(model, H)

print("\n===== Model Efficiency Metrics =====")
print(f"Total Parameters: {params}")
print(f"MACs per forward pass: {macs}")
print(f"FLOPs per forward pass: {flops}")
print("=====================================\n")


# ==============================
# EVALUATION
# ==============================

correct = 0
total = 0

for path, label in dataset.samples:

    if CHANNELS == 1:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        img = img.astype(float) / 255.0
        img = img.reshape(1, 28, 28)
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (32, 32))
        img = img.astype(float) / 255.0
        img = img[:, :, ::-1]
        img = img.transpose(2, 0, 1)

    x = backend.Tensor([1, CHANNELS, H, W])
    x.data = img.flatten().tolist()

    logits = model.forward(x)

    row = logits.data[:num_classes]
    pred = row.index(max(row))

    if pred == label:
        correct += 1

    total += 1

accuracy = 100 * correct / total

print(f"Evaluation Accuracy: {accuracy:.2f}%")
