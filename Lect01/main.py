import numpy as np
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.image import resize

# Load the CIFAR-100 dataset
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# CIFAR-100 has 32x32 images; resize them to 299x299 for InceptionResNetV2
x_test_resized = np.array([resize(image, (299, 299)).numpy() for image in x_test])

# Preprocess the images for the model
x_test_preprocessed = preprocess_input(x_test_resized)

# Load the pre-trained InceptionResNetV2 model
model = InceptionResNetV2(weights='imagenet')

# Predict using the pre-trained model
predictions = model.predict(x_test_preprocessed)

# Decode the predictions
decoded_predictions = decode_predictions(predictions, top=3)

# Print predictions for the first 5 test images
print("Predictions for the first 5 test images:")
for i in range(5):
    print(f"Image {i + 1}:")
    for pred in decoded_predictions[i]:
        print(f"{pred[1]}: {pred[2]*100:.2f}%")
    print()
