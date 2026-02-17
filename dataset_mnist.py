import os
import cv2
import random

class ImageFolderDataset:

    def __init__(self, root):

        self.root = root
        self.samples = []
        self.class_to_idx = {}

        classes = sorted(os.listdir(root))
        for idx, cls in enumerate(classes):
            self.class_to_idx[cls] = idx

            class_dir = os.path.join(root, cls)

            for file in os.listdir(class_dir):
                path = os.path.join(class_dir, file)
                self.samples.append((path, idx))
            
            random.shuffle(self.samples)


        print("Total samples:", len(self.samples))
        print("Classes:", self.class_to_idx)

    def __len__(self):
        return len(self.samples)

    def get_batch(self, batch_size, batch_idx):

        start = batch_idx * batch_size
        end = start + batch_size

        batch = self.samples[start:end]

        images = []
        labels = []

        for path, label in batch:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28))

            img = img.reshape(1, 28, 28)   # (C,H,W)

            img = img.astype(float) / 255.0
            images.append(img.flatten().tolist())

            labels.append(label)

        return images, labels
    
    def detect_channels(self):
        """
        Detect number of channels from first image in dataset.
        """
        first_path, _ = self.samples[0]
        img = cv2.imread(first_path, cv2.IMREAD_UNCHANGED)

        if len(img.shape) == 2:
            return 1
        else:
            return img.shape[2]


    def load_image(self, path):

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        img = img.astype(float) / 255.0
        # print(len(img.shape))
        img = img.reshape(1, 28, 28)

        return img.flatten().tolist()

