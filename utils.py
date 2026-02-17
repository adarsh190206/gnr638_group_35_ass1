import pickle

def save_weights(model, filepath):

    weights = {}

    for i, p in enumerate(model.parameters()):
        weights[f"param_{i}"] = {
            "shape": p.shape,
            "data": p.data
        }

    with open(filepath, "wb") as f:
        pickle.dump(weights, f)

    print("Weights saved to", filepath)


def load_weights(model, filepath):

    with open(filepath, "rb") as f:
        weights = pickle.load(f)

    for i, p in enumerate(model.parameters()):
        p.data = weights[f"param_{i}"]["data"]

    print("Weights loaded from", filepath)
