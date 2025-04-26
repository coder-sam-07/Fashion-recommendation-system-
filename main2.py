import streamlit as st
import numpy as np
import pickle
import os
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image
import torch
import re
from transformers import BlipProcessor, BlipForConditionalGeneration
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

# Set Streamlit page configuration
st.set_page_config(page_title='Fashion Recommendation System', layout='wide', page_icon='👗')

# Load pre-trained model for feature extraction
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# Load extracted features
with open("fixed_extracted_features.pkl", "rb") as f:
    feature_array = np.array(pickle.load(f), dtype=np.float32)

# Load filenames
with open("filenames.pkl", "rb") as f:
    filenames = pickle.load(f)

filenames = [os.path.join("images", os.path.basename(f)) for f in filenames]

# Normalize features
feature_array = normalize(feature_array, norm='l2')

# Apply K-Means Clustering
num_clusters = 10  # Adjust the number of clusters as needed
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(feature_array)

# Train KNN classifier on extracted features
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(feature_array, np.arange(len(filenames)))

# Function to extract features from an image
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    features = features.flatten()
    return normalize(features.reshape(1, -1), norm='l2')

# Function to find similar images using KNN
def find_similar_knn(query_img, top_n=5):
    query_features = extract_features(query_img, model)
    distances, indices = knn.kneighbors(query_features, n_neighbors=top_n)
    return [filenames[i] for i in indices[0] if os.path.exists(filenames[i])]

# Function to find similar images using K-Means
def find_similar_kmeans(query_img, top_n=5):
    query_features = extract_features(query_img, model)
    cluster_label = kmeans.predict(query_features)[0]
    cluster_indices = np.where(kmeans_labels == cluster_label)[0]
    cluster_features = feature_array[cluster_indices]
    cluster_filenames = [filenames[i] for i in cluster_indices]
    
    # Find closest images within the cluster
    knn_within_cluster = KNeighborsClassifier(n_neighbors=top_n, metric='euclidean')
    knn_within_cluster.fit(cluster_features, cluster_filenames)
    distances, indices = knn_within_cluster.kneighbors(query_features, n_neighbors=min(top_n, len(cluster_filenames)))
    return [cluster_filenames[i] for i in indices[0] if os.path.exists(cluster_filenames[i])]

# Load BLIP model for image captioning
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, blip_model = load_blip_model()

# Function to generate image description
def generate_image_description(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        generated_ids = blip_model.generate(**inputs, max_length=50)
    description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Remove unwanted words
    unwanted_phrases = ["a person in a", "a man in a", "a woman in a", "a model in a", "someone in a", "a in a"]
    for phrase in unwanted_phrases:
        description = re.sub(rf"\b{phrase}\b", "", description, flags=re.IGNORECASE)
    
    return description.strip().capitalize()

# Sidebar for image upload
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_path = os.path.join("upload", uploaded_file.name)
    os.makedirs("upload", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display uploaded image
    st.sidebar.subheader("Uploaded Image")
    st.sidebar.image(file_path, caption="Uploaded Image", use_container_width=True)
    
    # Get recommendations using KNN and K-Means
    knn_recommendations = find_similar_knn(file_path)
    kmeans_recommendations = find_similar_kmeans(file_path)
    
    # Show KNN recommendations
    st.subheader("🔍 KNN Recommendations")
    cols_knn = st.columns(5)
    for i, img_path in enumerate(knn_recommendations[:5]):
        with cols_knn[i]:
            st.image(img_path, caption=f"KNN Rec {i+1}", use_container_width=True)
            description = generate_image_description(img_path)
            st.write(f"**Description:** {description}")
    
    # Show K-Means recommendations
    st.subheader("🔍 K-Means Recommendations")
    cols_kmeans = st.columns(5)
    for i, img_path in enumerate(kmeans_recommendations[:5]):
        with cols_kmeans[i]:
            st.image(img_path, caption=f"K-Means Rec {i+1}", use_container_width=True)
            description = generate_image_description(img_path)
            st.write(f"**Description:** {description}")
    
    st.success("✅ Fashion Recommendation System is ready!")
else:
    st.info("Please upload an image to get recommendations.")
