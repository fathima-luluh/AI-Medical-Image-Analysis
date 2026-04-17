import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load model
model = load_model("medical_model.h5")

# Image path
img_path = "data/chest_xray/train/PNEUMONIA/person1_bacteria_1.jpeg"


# Load image
img = image.load_img(img_path, target_size=(256, 256), color_mode="grayscale")
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)

# Result
label = "PNEUMONIA" if prediction[0] > 0.5 else "NORMAL"

# Show image with prediction
plt.imshow(img, cmap='gray')
plt.title(f"Prediction: {label}")
plt.axis('off')

# Save image
plt.savefig("output.png")

# Show
plt.show()