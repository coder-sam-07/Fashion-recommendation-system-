import streamlit as st
import numpy as np
import faiss
import pickle
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import cv2
import os

# Load pre-trained model
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# Load extracted features
with open("fixed_extracted_features.pkl", "rb") as f:
    feature_array = np.array(pickle.load(f), dtype=np.float32)

# Load filenames
with open("filenames.pkl", "rb") as f:
    filenames = pickle.load(f)

# Ensure filenames point to local 'images/' folder
filenames = [os.path.join("images", os.path.basename(f)) for f in filenames]

# Initialize FAISS index
index = faiss.IndexFlatL2(feature_array.shape[1])
index.add(feature_array)

# Function to extract features from an image
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

# Function to find similar images
def find_similar_images(query_img, top_n=5):
    query_features = extract_features(query_img, model)
    query_features = np.expand_dims(query_features, axis=0).astype(np.float32)
    distances, indices = index.search(query_features, top_n)
    
    recommended_images = []
    for i in indices[0]:
        img_path = filenames[i]  # Ensure path is correct
        if os.path.exists(img_path):  # Check if file exists
            recommended_images.append(img_path)
        else:
            st.warning(f"⚠️ Image not found: {img_path}")  # Show warning if file is missing
    return recommended_images

# Streamlit UI
st.title("🛍️ Fashion Recommendation System")

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Save uploaded file
    file_path = os.path.join("upload", uploaded_file.name)
    os.makedirs("upload", exist_ok=True)  # Ensure upload folder exists
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display uploaded image
    st.image(file_path, caption="Uploaded Image", width=300)

    # Get recommendations
    recommended_images = find_similar_images(file_path)

    # Show recommended images
    st.subheader("🔍 Recommended Images")
    cols = st.columns(5)  # Display images in a row
    for i, img_path in enumerate(recommended_images):
        with cols[i]:
            st.image(img_path, caption=f"Rec {i+1}", width=150)

st.success("✅ Fashion Recommendation System is ready!")
