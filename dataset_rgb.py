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

            img = cv2.imread(path, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError(f"Failed to load image: {path}")

            img = cv2.resize(img, (32, 32))
            img = img.astype(float) / 255.0

            # BGR → RGB
            img = img[:, :, ::-1]

            # HWC → CHW
            img = img.transpose(2, 0, 1)

            images.append(img.flatten().tolist())
            labels.append(label)

        return images, labels

    


    def load_image(self, path):

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (32, 32))
        img = img.astype(float) / 255.0
        # print(len(img.shape))
        img = img[:,:,::-1]
        img = img.transpose(2,0,1)

        return img.flatten().tolist()

