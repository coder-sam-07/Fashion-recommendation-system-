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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

# Set Streamlit page configuration
st.set_page_config(page_title='Fashion Recommendation System', layout='wide', page_icon='👗')

# === LANDING PAGE ===
if 'start_app' not in st.session_state:
    st.session_state.start_app = False

if not st.session_state.start_app:
    st.markdown("""
        <style>
            .landing-title {
                font-size: 50px;
                font-weight: bold;
                text-align: center;
                margin-top: 20vh;
                animation: fadeIn 2s ease-in-out;
                color: #FF4B4B;
            }
            .landing-subtitle {
                font-size: 24px;
                text-align: center;
                margin-bottom: 40px;
                animation: fadeIn 3s ease-in-out;
                color: #ccc;
            }
            .start-button {
                display: flex;
                justify-content: center;
            }
            @keyframes fadeIn {
                0% { opacity: 0; transform: translateY(-20px); }
                100% { opacity: 1; transform: translateY(0); }
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="landing-title">👗 Fashion Recommendation System</div>', unsafe_allow_html=True)
    st.markdown('<div class="landing-subtitle">Upload an image to get visually similar fashion recommendations!</div>', unsafe_allow_html=True)
    if st.button("🚀 Start Recommendation System", key="start_btn"):
        st.session_state.start_app = True

    st.stop()  # Stop execution here until user clicks the button

# === MAIN APP STARTS HERE ===

# Custom CSS for animations and styling
st.markdown("""
    <style>
    .title {
        font-size: 48px;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        animation: fadeIn 2s ease-in-out;
    }
    .subtitle {
        font-size: 24px;
        color: #444;
        text-align: center;
        margin-bottom: 40px;
        animation: fadeIn 3s ease-in-out;
    }
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(-20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">👗 Fashion Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Get smart recommendations based on your uploaded outfit!</div>', unsafe_allow_html=True)

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

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(feature_array, np.arange(len(filenames)))

# Feature extractor
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    features = features.flatten()
    return normalize(features.reshape(1, -1), norm='l2')

# Find similar
def find_similar_knn(query_img, top_n=5):
    query_features = extract_features(query_img, model)
    distances, indices = knn.kneighbors(query_features, n_neighbors=top_n)
    return [filenames[i] for i in indices[0] if os.path.exists(filenames[i])]

# Load BLIP
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, blip_model = load_blip_model()

def generate_image_description(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        generated_ids = blip_model.generate(**inputs, max_length=50)
    description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    unwanted_phrases = ["a person in a", "a man in a", "a woman in a", "a model in a", "someone in a", "a in a"]
    for phrase in unwanted_phrases:
        description = re.sub(rf"\b{phrase}\b", "", description, flags=re.IGNORECASE)
    description = re.sub(r'\b(\w+)( \1\b)+', r'\1', description)

    return description.strip().capitalize()

# Upload & Recommend Section
st.sidebar.header("📤 Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_path = os.path.join("upload", uploaded_file.name)
    os.makedirs("upload", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.subheader("✅ Uploaded Image")
    st.sidebar.image(file_path, caption="Uploaded Image", use_container_width=True)

    knn_recommendations = find_similar_knn(file_path)

    st.subheader("🔍 Top 5 Recommendations (KNN)")
    cols_knn = st.columns(5)
    for i, img_path in enumerate(knn_recommendations[:5]):
        with cols_knn[i]:
            st.image(img_path, caption=f"Recommendation {i+1}", use_container_width=True)
            description = generate_image_description(img_path)
            st.write(f"**Description:** {description}")

    st.success("✨ Fashion recommendations generated successfully!")
else:
    st.info("👈 Please upload an image to get recommendations.")
