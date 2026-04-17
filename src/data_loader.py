import os
import cv2
import numpy as np

def load_dataset(path):
    data = []
    labels = []

    classes = ["NORMAL", "PNEUMONIA"]

    for label, class_name in enumerate(classes):
        class_path = os.path.join(path, class_name)

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (256, 256))
                data.append(img)
                labels.append(label)
            except:
                pass

    data = np.array(data).reshape(-1, 256, 256, 1) / 255.0
    labels = np.array(labels)

    return data, labels