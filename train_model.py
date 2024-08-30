import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # type: ignore
import matplotlib.pyplot as plt
import argparse

def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128))
            img = img / 255.0
            img = np.expand_dims(img, axis=-1)
            images.append(img)
            labels.append(label)
    return images, labels

def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(main_folder, epochs=20, batch_size=32):
    cancerous_folder = os.path.join(main_folder, 'Cancerous')
    non_cancerous_folder = os.path.join(main_folder, 'Non-Cancerous')

    print(f"Loading cancerous images from: {cancerous_folder}")
    print(f"Loading non-cancerous images from: {non_cancerous_folder}")

    cancerous_images, cancerous_labels = load_images_from_folder(cancerous_folder, 1)
    non_cancerous_images, non_cancerous_labels = load_images_from_folder(non_cancerous_folder, 0)

    print(f"Loaded {len(cancerous_images)} cancerous images and {len(non_cancerous_images)} non-cancerous images")

    X = np.array(cancerous_images + non_cancerous_images)
    Y = np.array(cancerous_labels + non_cancerous_labels)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = create_model()
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test))

    loss, accuracy = model.evaluate(X_test, Y_test)
    print(f'Test Loss: {loss}')
    print(f'Test Accuracy: {accuracy}')

    model.save('lung_cancer_detector_model.keras')

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train lung cancer detection model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the main dataset directory')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    args = parser.parse_args()

    train_model(args.data_dir, args.epochs, args.batch_size)