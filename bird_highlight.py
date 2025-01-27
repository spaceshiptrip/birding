import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import sys

def load_model():
    """Load the InceptionV3 model pre-trained on ImageNet."""
    model = InceptionV3(weights='imagenet')
    return model

def preprocess_image(img_path):
    """Load and preprocess the image for the model."""
    img = image.load_img(img_path, target_size=(299, 299))  # Resize to model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_bird(model, img_array):
    """Predict whether the image contains a bird and return the top predictions."""
    preds = model.predict(img_array)
    decoded_preds = decode_predictions(preds, top=5)  # Get top-5 predictions

    bird_predictions = []
    for _, label, prob in decoded_preds[0]:
        if 'bird' in label.lower():  # Check if the prediction is related to birds
            bird_predictions.append((label, prob))

    return bird_predictions

def highlight_birds(img_path, model):
    """Highlight the areas of the image recognized as birds and save the intermediate image."""
    original_img = cv2.imread(img_path)
    img_array = preprocess_image(img_path)

    preds = model.predict(img_array)
    decoded_preds = decode_predictions(preds, top=5)

    bird_detected = False
    for _, label, prob in decoded_preds[0]:
        if 'bird' in label.lower():
            bird_detected = True
            break

    if bird_detected:
        # Create a highlight effect (dummy example as bounding boxes are not available)
        height, width, _ = original_img.shape
        overlay = original_img.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 255, 0), thickness=10)
        alpha = 0.4  # Transparency factor
        highlighted_img = cv2.addWeighted(overlay, alpha, original_img, 1 - alpha, 0)
        output_path = "highlighted_birds.jpg"
        cv2.imwrite(output_path, highlighted_img)
        print(f"Highlighted image saved as {output_path}")
    else:
        print("No birds detected, skipping image highlighting.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python bird_recognition.py <path_to_image>")
        sys.exit(1)

    img_path = sys.argv[1]

    try:
        print("Loading model...")
        model = load_model()

        print("Preprocessing image...")
        img_array = preprocess_image(img_path)

        print("Making predictions...")
        bird_predictions = predict_bird(model, img_array)

        bird_count = len(bird_predictions)
        if bird_count > 0:
            print(f"Bird(s) detected! Total birds found: {bird_count}")
            print("Top predictions:")
            for label, prob in bird_predictions:
                print(f"{label}: {prob * 100:.2f}%")

            # Highlight birds in the image
            highlight_birds(img_path, model)
        else:
            print("No birds detected in the image.")

    except Exception as e:
        print(f"An error occurred: {e}")

