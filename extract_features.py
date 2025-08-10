# === Feature Extraction Script ===

import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from tqdm import tqdm
import pickle
from PIL import Image
from skimage.feature import hog
from sklearn.decomposition import PCA

# === Paths ===
DATASET_DIR = "../101_ObjectCategories"  # Path to Caltech 101 dataset folder
DEEP_FEATURES_FILE = "deep_features.npy"  # Where to save deep features
EDGE_FEATURES_FILE = "edge_features.npy"  # Where to save edge features
FILENAMES_FILE = "filenames.pkl"          # Where to save filenames list
PCA_MODEL_FILE = "pca_model.pkl"          # Where to save PCA model for edge features

# === Device selection ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load pretrained ResNet-50 model without classifier ===
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove final classifier layer
model.eval().to(device)  # Set model to evaluation mode and send to device

# === Define image preprocessing pipeline for deep features ===
transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                         std=[0.229, 0.224, 0.225])   # ImageNet std dev
])

def extract_deep_feature(image_pil):
    """
    Extracts a 2048-dimensional deep feature vector from a PIL image
    using pretrained ResNet-50 (without classifier).
    """
    img_tensor = transform(image_pil).unsqueeze(0).to(device)  # Preprocess & batchify
    with torch.no_grad():
        feat = model(img_tensor).squeeze().cpu().numpy()  # Extract features
    return feat.astype('float32')

def extract_edge_feature(image_pil):
    """
    Extracts edge features using Histogram of Oriented Gradients (HOG).
    Converts image to grayscale, resizes to 224x224, computes HOG vector.
    """
    gray_img = image_pil.convert('L').resize((224, 224), Image.BICUBIC)
    gray_np = np.array(gray_img, dtype=np.float32) / 255.0  # Normalize pixels to [0,1]
    fd = hog(gray_np,
             orientations=9,
             pixels_per_cell=(16, 16),
             cells_per_block=(2, 2),
             block_norm='L2-Hys',
             visualize=False,
             feature_vector=True)
    return fd.astype('float32')

def l2_normalize(features):
    """
    L2-normalize feature vectors row-wise for cosine similarity.
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-10  # Avoid div zero
    return features / norms

# Containers for features and filenames
deep_features = []
edge_features = []
filenames = []

print("📦 Extracting features (deep + edge) ...")
# Walk through all image files in dataset directory
for root, _, files in os.walk(DATASET_DIR):
    for file in tqdm(files):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(root, file)
            try:
                img_pil = Image.open(path).convert('RGB')  # Load image as RGB

                # Extract deep and edge features
                deep_feat = extract_deep_feature(img_pil)
                edge_feat = extract_edge_feature(img_pil)

                # Sanity checks to ensure feature integrity
                assert deep_feat.shape == (2048,), f"Deep feat shape mismatch: {deep_feat.shape}"
                assert edge_feat.ndim == 1, f"Edge feat not 1D vector: {edge_feat.shape}"
                assert np.isfinite(deep_feat).all(), "Deep feat contains NaN/inf"
                assert np.isfinite(edge_feat).all(), "Edge feat contains NaN/inf"
                deep_norm = np.linalg.norm(deep_feat)
                edge_norm = np.linalg.norm(edge_feat)
                assert deep_norm > 0, "Deep feat norm is zero"
                assert edge_norm > 0, "Edge feat norm is zero"

                # Append features and filename to lists
                deep_features.append(deep_feat)
                edge_features.append(edge_feat)
                filenames.append(path)
            except Exception as e:
                print(f"⚠️ Error with {path}: {e}")

# Convert feature lists to numpy arrays
deep_features = np.array(deep_features, dtype=np.float32)
edge_features = np.array(edge_features, dtype=np.float32)

# --- Apply PCA to reduce dimensionality of edge features ---
print("⚙️ Applying PCA to edge features ...")
pca = PCA(n_components=256)
edge_features = pca.fit_transform(edge_features).astype(np.float32)

print(f"PCA explained variance ratio sum: {np.sum(pca.explained_variance_ratio_):.4f}")
assert np.sum(pca.explained_variance_ratio_) > 0.7, "Low explained variance from PCA"

# --- Normalize all features ---
deep_features = l2_normalize(deep_features)
edge_features = l2_normalize(edge_features)

# --- Validate final features (no NaNs or infs, norm=1) ---
assert not np.isnan(deep_features).any(), "NaNs in deep_features after normalization"
assert not np.isnan(edge_features).any(), "NaNs in edge_features after normalization"
assert np.all(np.isfinite(deep_features)), "Infinite values in deep_features after normalization"
assert np.all(np.isfinite(edge_features)), "Infinite values in edge_features after normalization"
np.testing.assert_allclose(np.linalg.norm(deep_features, axis=1), 1.0, atol=1e-5)
np.testing.assert_allclose(np.linalg.norm(edge_features, axis=1), 1.0, atol=1e-5)

# --- Save features, filenames, and PCA model to disk ---
np.save(DEEP_FEATURES_FILE, deep_features)
np.save(EDGE_FEATURES_FILE, edge_features)
with open(FILENAMES_FILE, 'wb') as f:
    pickle.dump(filenames, f)
with open(PCA_MODEL_FILE, 'wb') as f:
    pickle.dump(pca, f)

print(f"✅ Done. Saved features and filenames:")
print(f" - {DEEP_FEATURES_FILE} {deep_features.shape}")
print(f" - {EDGE_FEATURES_FILE} {edge_features.shape}")
print(f" - {FILENAMES_FILE} ({len(filenames)} files)")
print(f" - {PCA_MODEL_FILE} (PCA model for query edges)")
