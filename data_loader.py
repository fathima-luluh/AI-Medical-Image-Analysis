import os
import cv2
import numpy as np

def load_dataset(data_path):
    X = []
    y = []

    classes = ["NORMAL", "PNEUMONIA"]

    for label, class_name in enumerate(classes):
        class_path = os.path.join(data_path, class_name)

        for img_name in os.listdir(class_path)[:200]:  # small dataset for speed
            img_path = os.path.join(class_path, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128))

            X.append(img)
            y.append(label)

    X = np.array(X).reshape(-1, 128, 128, 1) / 255.0
    y = np.array(y)

    return X, y