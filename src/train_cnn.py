import tensorflow as tf
from tensorflow.keras import layers, models
from preprocess import create_data_generators
import matplotlib.pyplot as plt
import os

def build_cnn(input_shape=(224,224,3)):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def train_model(train_dir, val_dir):
    train_gen, val_gen = create_data_generators(train_dir, val_dir)

    model = build_cnn()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(train_gen, validation_data=val_gen, epochs=10)

    os.makedirs('models', exist_ok=True)
    model.save('models/face_mask_cnn.h5')

    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.title('Training Accuracy')
    plt.savefig('outputs/training_accuracy.png')
    plt.close()

if __name__ == "__main__":
    train_model("data/face_mask_dataset/train", "data/face_mask_dataset/val")
