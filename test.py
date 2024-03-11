import pickle 
import numpy as np 
import tensorflow
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalAveragePooling2D
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import cv2


feature_list=np.array(pickle.load(open('Embedding_features.pkl','rb')))
file_names=pickle.load(open('Embedding_file_names.pkl','rb'))


model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False

# Add GlobalAveragePooling2D layer
global_average_layer = GlobalAveragePooling2D() 
model = tensorflow.keras.Sequential([
    model,
    global_average_layer
])


img = image.load_img('2164.jpg',target_size=(224,224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

neighbors=NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)

distances, indices=neighbors.kneighbors([normalized_result])

print(indices[0])
for file in indices[0][1:6]:
    temp_img=cv2.imread(file_names[file])
    cv2.imshow('output',temp_img)
    cv2.waitKey(0)