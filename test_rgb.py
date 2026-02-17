from framework.model_rgb import SimpleCNN
from framework import backend
from dataset_rgb import ImageFolderDataset

import random
import time
import pickle
import json
import os

# ==============================
# CONFIG
# ==============================

DATA_ROOT = "Ass1datasets/data_2"   # data_1 = MNIST, data_2 = RGB
BATCH_SIZE = 64
LR = 0.01
EPOCHS = 50
VAL_FRAC = 0.1

# ==============================
# DATASET LOADING
# ==============================

print("Loading dataset...")

t0 = time.perf_counter()
dataset = ImageFolderDataset(DATA_ROOT)
t1 = time.perf_counter()

print(f"Dataset scan time: {t1 - t0:.3f} seconds")

num_classes = len(dataset.class_to_idx)
print("Number of classes:", num_classes)

# ==============================
# MANUAL INPUT CONFIG
# ==============================


                      # RGB dataset
CHANNELS = 3
H = 32
W = 32
dataset_name = "rgb"

# print(f"Input shape: ({CHANNELS}, {H}, {W})")

WEIGHTS_PATH = f"cnn_weights_{dataset_name}.pkl"
CLASS_NAMES_PATH = f"class_names_{dataset_name}.json"

# ==============================
# TRAIN / VAL SPLIT
# ==============================

indices = list(range(len(dataset.samples)))
random.shuffle(indices)

val_size = int(len(indices) * VAL_FRAC)
val_idx = set(indices[:val_size])

train_samples = [dataset.samples[i] for i in indices if i not in val_idx]
val_samples = [dataset.samples[i] for i in indices if i in val_idx]

print("Train size:", len(train_samples))
print("Val size:", len(val_samples))

# ==============================
# PRELOAD IMAGES
# ==============================

print("Preloading images...")

t0 = time.perf_counter()

preloaded = {}
for path, label in dataset.samples:
    preloaded[path] = dataset.load_image(path)

t1 = time.perf_counter()

print(f"Preloaded {len(preloaded)} images in {t1 - t0:.3f} seconds")

# ==============================
# MODEL
# ==============================

model = SimpleCNN(num_classes, in_channels=CHANNELS)

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
# TRAINING
# ==============================

for epoch in range(EPOCHS):

    random.shuffle(train_samples)

    total_loss = 0.0
    correct = 0
    total = 0

    num_batches = len(train_samples) // BATCH_SIZE

    for b in range(num_batches):

        batch = train_samples[b*BATCH_SIZE:(b+1)*BATCH_SIZE]

        images = []
        labels = []

        for path, label in batch:
            images.extend(preloaded[path])
            labels.append(label)

        x = backend.Tensor([BATCH_SIZE, CHANNELS, H, W])
        x.data = images

        logits = model.forward(x)
        loss = model.backward(labels)
        model.step(LR)

        total_loss += loss

        for bb in range(BATCH_SIZE):
            start = bb * num_classes
            end = start + num_classes
            row = logits.data[start:end]
            pred = row.index(max(row))
            if pred == labels[bb]:
                correct += 1
            total += 1

    print("\n==============================")
    print(f"Epoch {epoch}")
    print(f"Train Loss: {total_loss / num_batches:.4f}")
    print(f"Train Accuracy: {100 * correct / total:.2f}%")
    # print(f"Parameters: {params}")
    # print(f"MACs: {macs}")
    # print(f"FLOPs: {flops}")
    print("==============================")

   
# ==============================
# SAVE MODEL
# ==============================

weights = {}
for i, p in enumerate(model.parameters()):
    weights[f"param_{i}"] = {
        "shape": p.shape,
        "data": p.data
    }

with open(WEIGHTS_PATH, "wb") as f:
    pickle.dump(weights, f)

print("Saved weights to", WEIGHTS_PATH)

with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(dataset.class_to_idx, f)

print("Saved class names to", CLASS_NAMES_PATH)
