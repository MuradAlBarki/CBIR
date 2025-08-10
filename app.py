# === Streamlit Web App Script ===

import streamlit as st
import numpy as np
import pickle
import os
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torchvision import models, transforms
from skimage.feature import hog
from sklearn.decomposition import PCA

@st.cache_resource(show_spinner=False)
def load_model_and_pca():
    """
    Loads pretrained ResNet-50 model (without classifier) and PCA model for edge features.
    Uses GPU if available.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval().to(device)

    with open("pca_model.pkl", "rb") as f:
        pca_model = pickle.load(f)

    return device, model, pca_model

# Load models and PCA once with caching
device, model, pca = load_model_and_pca()

# Define preprocessing transform matching training
transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load filenames and pre-extracted features
with open("filenames.pkl", "rb") as f:
    filenames = pickle.load(f)

deep_features = np.load("deep_features.npy")
edge_features = np.load("edge_features.npy")

def l2_normalize(features):
    """L2-normalize features for cosine similarity."""
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-10
    return features / norms

# Normalize features loaded from disk
deep_features = l2_normalize(deep_features)
edge_features = l2_normalize(edge_features)

# UI Title and description
st.markdown("### From Image Edges to Deep Features: A Modern Approach to Content-Based Search in Big Data")
st.markdown("Multimedia final project")
st.title("🔍 Content-Based Image Retrieval")

# File uploader for user image input
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

# Sidebar options for user to choose search mode and number of results
st.sidebar.header("🔧 Options")
search_mode = st.sidebar.selectbox("Search using:", ["Edge", "Deep"], index=0)  # Default to Edge
top_k = st.sidebar.slider("Number of results", 1, 30, 5)

def extract_deep_feature(image_pil):
    """Extract normalized deep feature vector from uploaded PIL image."""
    try:
        img_tensor = transform(image_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(img_tensor).squeeze().cpu().numpy()
        feat = feat.astype('float32')
        feat /= (np.linalg.norm(feat) + 1e-10)  # Normalize
        return feat
    except Exception as e:
        st.error(f"Error extracting deep feature: {e}")
        return None

def extract_edge_feature(image_pil):
    """Extract normalized PCA-reduced HOG edge feature vector from uploaded PIL image."""
    try:
        gray_img = image_pil.convert('L').resize((224, 224), Image.BICUBIC)
        gray_np = np.array(gray_img, dtype=np.float32) / 255.0
        fd = hog(gray_np,
                 orientations=9,
                 pixels_per_cell=(16, 16),
                 cells_per_block=(2, 2),
                 block_norm='L2-Hys',
                 visualize=False,
                 feature_vector=True)
        fd = fd.astype('float32')
        fd_pca = pca.transform(fd.reshape(1, -1)).flatten()  # Apply PCA
        fd_pca /= (np.linalg.norm(fd_pca) + 1e-10)  # Normalize
        return fd_pca
    except Exception as e:
        st.error(f"Error extracting edge feature: {e}")
        return None

def show_image_grid(paths, scores, title):
    """Display retrieved images with similarity scores in a grid layout."""
    st.subheader(title)
    cols = st.columns(min(4, len(paths)))  # Up to 4 columns
    for i, (path, score) in enumerate(zip(paths, scores)):
        with cols[i % 4]:
            st.image(Image.open(path), use_container_width=True)
            st.caption(f"🧮 Similarity: {score*100:.2f}%")

if uploaded_file is not None:
    uploaded_img = Image.open(uploaded_file).convert("RGB")
    st.image(uploaded_img, caption="🖼 Uploaded Image", width=300)

    # Extract query features based on user-selected mode
    if search_mode == "Deep":
        query_features = extract_deep_feature(uploaded_img)
        if query_features is None:
            st.stop()
        features = deep_features
    else:  # Edge mode
        query_features = extract_edge_feature(uploaded_img)
        if query_features is None:
            st.stop()
        features = edge_features

    # Compute cosine similarity between query and all dataset features
    similarity = cosine_similarity(query_features.reshape(1, -1), features)[0]
    sorted_indices = np.argsort(similarity)[::-1]  # Descending order
    top_indices = sorted_indices[:top_k]

    # Retrieve paths and similarity scores for top results
    top_paths = [filenames[i] for i in top_indices]
    top_scores = [similarity[i] for i in top_indices]

    # Simple prediction by majority vote of folder (category) names
    from collections import Counter
    top_folders = [os.path.basename(os.path.dirname(p)) for p in top_paths]
    most_common_folder = Counter(top_folders).most_common(1)[0][0]
    st.markdown(f"**Prediction: It's an image of {most_common_folder}.**")

    # Display retrieved images
    show_image_grid(top_paths, top_scores, f"🔍 Top {top_k} Similar Matches")
