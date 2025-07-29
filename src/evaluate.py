import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model_path, test_dir, img_size=(224,224)):
    model = tf.keras.models.load_model(model_path)
    datagen = ImageDataGenerator(rescale=1./255)
    test_gen = datagen.flow_from_directory(test_dir, target_size=img_size, batch_size=32, class_mode='binary', shuffle=False)

    preds = (model.predict(test_gen) > 0.5).astype("int32")
    print(classification_report(test_gen.classes, preds, target_names=['Without Mask', 'With Mask']))

    cm = confusion_matrix(test_gen.classes, preds)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=['No Mask', 'Mask'], yticklabels=['No Mask', 'Mask'])
    plt.title('Confusion Matrix')
    plt.savefig('outputs/confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    evaluate_model("models/face_mask_cnn.h5", "data/face_mask_dataset/test")
