import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import os
import numpy as np 
from tqdm import tqdm
import pickle
from numpy.linalg import norm

# Load ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False

# Add GlobalAveragePooling2D layer
global_average_layer = GlobalAveragePooling2D() 
model = tensorflow.keras.Sequential([
    model,
    global_average_layer
])

# Function to extract features from a batch of images
def extract_features_batch(image_paths, model):
    images = []
    for img_path in image_paths:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_arr = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_arr)
        images.append(preprocessed_img)
    images = np.vstack(images)
    results = model.predict(images)
    normalized_results = [result / norm(result) for result in results]
    return normalized_results

# Directory containing images
image_dir = 'images'

# List of image file names
file_names = [os.path.join(image_dir, file) for file in os.listdir(image_dir)]

# Batch size for processing images
batch_size = 128

# Extract features from images in batches
feature_list = []
for i in tqdm(range(0, len(file_names), batch_size)):
    batch_files = file_names[i:i+batch_size]
    batch_features = extract_features_batch(batch_files, model)
    feature_list.extend(batch_features)

# Save features and file names to separate pickle files
pickle.dump(feature_list, open('Embedding_features.pkl', 'wb'))
pickle.dump(file_names, open('Embedding_file_names.pkl', 'wb'))
