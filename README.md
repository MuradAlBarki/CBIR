echo "# From Image Edges to Deep Features: A Modern Approach to Content-Based Search in Big Data

## Description
This project demonstrates two content-based image retrieval (CBIR) methods on the Caltech 101 dataset:

- Edge-based retrieval using Histogram of Oriented Gradients (HOG) features with PCA for dimensionality reduction.
- Deep-feature-based retrieval using features extracted from a pretrained ResNet-50 deep neural network.

The goal is to show how traditional edge-based methods often fail to capture semantic similarity, while deep learning features provide more meaningful and accurate search results.

## Dataset
The project uses the Caltech 101 Dataset, which contains images of 101 object categories plus background.

Download and extract the dataset:

\`\`\`bash
wget http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
tar -xvzf 101_ObjectCategories.tar.gz
\`\`\`

Make sure the extracted \`101_ObjectCategories\` folder is accessible by the code.

## Usage

### 1. Extract Features (Offline)
Run the feature extraction script to process images and save features:

\`\`\`bash
python extract_features.py
\`\`\`

This creates:

- \`deep_features.npy\`
- \`edge_features.npy\`
- \`filenames.pkl\`
- \`pca_model.pkl\`

### 2. Launch Retrieval Web App
Start the Streamlit app for interactive image search:

\`\`\`bash
streamlit run app.py
\`\`\`

Upload an image and select either Edge or Deep search mode. The app returns top-k visually and semantically similar images from the dataset.

## Appendix
The project includes fully documented code for:

- Feature extraction (\`extract_features.py\`)
- Web interface (\`app.py\`)

with detailed comments explaining each step.
" > README.md
