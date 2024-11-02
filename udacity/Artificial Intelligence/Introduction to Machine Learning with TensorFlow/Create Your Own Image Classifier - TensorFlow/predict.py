import argparse
import json
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub

# Function to load and process the image
# def preprocess_image(image_path, label):
#     image = Image.open(image_path)
#     image = image.resize((224, 224))
#     image = np.asarray(image) / 255.0
#     image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
#     image = image[np.newaxis, ...]
#     return image

def preprocess_image(image, label):
    # Normalize pixels to the range [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    # Resize images to the desired size
    image = tf.image.resize(image, (224, 224))
    return image, label

# Function to predict the top K classes
def predict(image_path, model, top_k):
    # Load and preprocess the image
    image = Image.open(image_path)
    processed_image, _ = preprocess_image(image, None)
    processed_image = np.expand_dims(processed_image, axis=0)

    # Predict the probabilities
    predictions = model.predict(processed_image)
    top_indices = np.argsort(predictions[0])[-top_k:][::-1]
    top_indices_adjusted = [index + 1 for index in top_indices]  # Adjust indices by adding 1
    top_probabilities = predictions[0][top_indices]

    # Get the class labels
    label_map = json.load(open('label_map.json'))
    classes = [label_map[str(index)] for index in top_indices_adjusted]

    return top_probabilities, classes

# Function to load the label map
def load_label_map(label_map_path):
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    return label_map

# Main function for predicting
def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Flower Image Classifier')

    # Add arguments
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('model_path', help='Path to the saved Keras model')
    parser.add_argument('--top_k', type=int, default=5, help='Return the top K most likely classes')
    parser.add_argument('--category_names', help='Path to the JSON file mapping labels to flower names', default='label_map.json')

    # Parse the arguments
    args = parser.parse_args()

    # Load the model with custom objects
    custom_objects = {'KerasLayer': hub.KerasLayer}  # Replace 'KerasLayer' with the name of your custom layer
    model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects)
    
    top_probs, top_classes = predict(args.image_path, model, args.top_k)

    # Load the label map if provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            label_map = json.load(f)
        top_classes = [label_map.get(str(cls), str(cls)) for cls in top_classes]

    # Print the top classes and probabilities
    for prob, cls in zip(top_probs, top_classes):
        print(f'{cls}: {prob:.4f}')
if __name__ == '__main__':
    main()