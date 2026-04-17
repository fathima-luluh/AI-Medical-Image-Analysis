from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset path
dataset_path = r"C:\Users\fathi\Documents\AI-Medical-Image-Analysis\data\chest_xray\train"

# Data pipeline (NO MEMORY LOAD)
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(256, 256),
    color_mode="grayscale",
    class_mode="binary",
    batch_size=16,
    subset="training"
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(256, 256),
    color_mode="grayscale",
    class_mode="binary",
    batch_size=16,
    subset="validation"
)

# CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(256,256,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(train_data, validation_data=val_data, epochs=5)

# Save model
model.save("medical_model.h5")

print("✅ Training Completed Successfully")