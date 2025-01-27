import torch
import torchvision.transforms as transforms
from torchvision import models
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import sys

def load_inceptionv3():
    """Load the InceptionV3 model pre-trained on ImageNet."""
    model = InceptionV3(weights='imagenet')
    return model

def load_inaturalist():
    """Load a pre-trained iNaturalist model using PyTorch."""
    model = models.resnet50(pretrained=True)
    model.eval()  # Set to evaluation mode
    return model

def preprocess_image_tensorflow(img_path):
    """Preprocess image for TensorFlow models."""
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def preprocess_image_pytorch(img_path):
    """Preprocess image for PyTorch models."""
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transforms.ToPILImage()(img)
    img_tensor = preprocess(img).unsqueeze(0)
    return img_tensor

def predict_bird_inceptionv3(model, img_array):
    """Predict bird species using InceptionV3."""
    preds = model.predict(img_array)
    decoded_preds = decode_predictions(preds, top=5)
    bird_predictions = []
    for _, label, prob in decoded_preds[0]:
        if 'bird' in label.lower():
            bird_predictions.append((label, prob))
    return bird_predictions

def predict_bird_inaturalist(model, img_tensor):
    """Predict bird species using iNaturalist model."""
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)

        bird_predictions = []
        for i in range(top5_catid.size(0)):
            bird_predictions.append((top5_catid[i].item(), top5_prob[i].item()))
    return bird_predictions

def try_all_methods(img_path):
    """Try InceptionV3 and iNaturalist models and print results."""
    methods = ["InceptionV3", "iNaturalist"]

    for method in methods:
        print(f"Trying method: {method}")

        if method == "InceptionV3":
            model = load_inceptionv3()
            img_array = preprocess_image_tensorflow(img_path)
            bird_predictions = predict_bird_inceptionv3(model, img_array)
        elif method == "iNaturalist":
            model = load_inaturalist()
            img_tensor = preprocess_image_pytorch(img_path)
            bird_predictions = predict_bird_inaturalist(model, img_tensor)
        
        print(f"Results for {method}:")
        if bird_predictions:
            for label, prob in bird_predictions:
                print(f"{label}: {prob * 100:.2f}%")
        else:
            print(f"No birds detected using {method}.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python bird_recognition.py <path_to_image>")
        sys.exit(1)

    img_path = sys.argv[1]

    try:
        try_all_methods(img_path)
    except Exception as e:
        print(f"An error occurred: {e}")
