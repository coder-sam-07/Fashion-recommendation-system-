import streamlit as st
import numpy as np
import faiss
import pickle
import os
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image
import torch
import re
from transformers import BlipProcessor, BlipForConditionalGeneration

# Set Streamlit page configuration
st.set_page_config(page_title='Fashion Recommendation System', layout='wide', page_icon='👗')

# Custom styling
st.markdown("""
    <style>
    .stButton>button, .stFileUploader>div>button {
        background-color: #ff4b4b;
        color: white;
        font-size: 16px;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# First page - Overview
if "start" not in st.session_state:
    st.session_state.start = False

if not st.session_state.start:
    st.title("🛍️ Fashion Recommendation System")
    st.write("Upload an image to find visually similar fashion items.")
    if st.button("Start Recommendation System"):
        st.session_state.start = True
    st.stop()

# Load pre-trained model for feature extraction
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# Load extracted features
with open("fixed_extracted_features.pkl", "rb") as f:
    feature_array = np.array(pickle.load(f), dtype=np.float32)

# Normalize features
faiss.normalize_L2(feature_array)

# Load filenames
with open("filenames.pkl", "rb") as f:
    filenames = pickle.load(f)

# Ensure filenames point to local 'images/' folder
filenames = [os.path.join("images", os.path.basename(f)) for f in filenames]

# Initialize FAISS index with inner product for better matching
index = faiss.IndexFlatIP(feature_array.shape[1])
index.add(feature_array)

# Load BLIP model for image captioning
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, blip_model = load_blip_model()

# Function to extract features from an image
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    
    # Normalize features
    features = features.flatten()
    faiss.normalize_L2(features.reshape(1, -1))
    return features

# Function to find similar images
def find_similar_images(query_img, top_n=5):
    query_features = extract_features(query_img, model)
    query_features = np.expand_dims(query_features, axis=0).astype(np.float32)
    distances, indices = index.search(query_features, top_n)
    
    recommended_images = [filenames[i] for i in indices[0] if os.path.exists(filenames[i])]
    return recommended_images

# Function to generate **product-focused** image description
def generate_image_description(image_path):
    image = Image.open(image_path).convert("RGB")

    # Encode the image for caption generation
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        generated_ids = blip_model.generate(**inputs, max_length=50)

    description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Remove unwanted words and phrases
    unwanted_words = ["a person in a", "a man in a", "a woman in a", "a model in a", "someone in a", "a in a"]
    for phrase in unwanted_words:
        description = re.sub(rf"\b{phrase}\b", "", description, flags=re.IGNORECASE)

    return description.strip().capitalize()  # Ensure proper formatting

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
    
    # Get recommendations
    recommended_images = find_similar_images(file_path)
    
    # Show recommended images with descriptions
    st.subheader("🔍 Recommended Images")
    cols = st.columns(5)
    for i, img_path in enumerate(recommended_images[:5]):
        with cols[i]:
            st.image(img_path, caption=f"Rec {i+1}", use_container_width=True)
            description = generate_image_description(img_path)
            st.write(f"**Description:** {description}")
    
    st.success("✅ Fashion Recommendation System is ready!")
else:
    st.info("Please upload an image to get recommendations.")
