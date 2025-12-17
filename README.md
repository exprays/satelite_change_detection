# Satellite Change Detection Project

A comprehensive satellite imagery change detection system that identifies differences between before and after images using multiple computer vision and deep learning techniques.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

- **Multiple Detection Methods**:
  - Image Differencing
  - Adaptive Thresholding
  - Change Vector Analysis (CVA)
  - Deep Learning (Siamese Network)

- **Comprehensive Visualization**:
  - Side-by-side before/after comparisons
  - Heatmaps and difference maps
  - Color-coded change overlays
  - Statistical analysis

- **Google Colab Compatible**: Run directly in your browser with GPU support

- **Export Capabilities**: Save results as PNG images and detailed statistics

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Methods Explained](#methods-explained)
- [Custom Data](#custom-data)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Option 1: Google Colab (Recommended)

1. Open the notebook in Google Colab:
   - Upload `satellite_change_detection.ipynb` to Google Colab
   - Or use: `File > Upload notebook`

2. Run the first cell to install all dependencies automatically

3. Mount Google Drive (optional) to save results

### Option 2: Local Installation

```bash
# Clone or download this repository
git clone <your-repo-url>
cd satellite-change-detection

# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook satellite_change_detection.ipynb
```

## 📦 Requirements

Create a `requirements.txt` file with:

```txt
numpy>=1.19.0
matplotlib>=3.3.0
opencv-python>=4.5.0
scikit-image>=0.18.0
pillow>=8.0.0
tensorflow>=2.8.0
torch>=1.10.0
torchvision>=0.11.0
rasterio>=1.2.0
geopandas>=0.10.0
jupyter>=1.0.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

## ⚡ Quick Start

### Using Sample Data (Demo)

The notebook includes built-in sample satellite images for immediate testing:

```python
# Simply run all cells in order
# Sample images are automatically generated in Section 4
```

### Using Your Own Satellite Images

```python
import cv2

# Load your satellite images
before_img = cv2.imread('path/to/before_image.png')
after_img = cv2.imread('path/to/after_image.png')

# For GeoTIFF files
import rasterio
with rasterio.open('satellite_image.tif') as src:
    image = src.read([1, 2, 3])  # Read RGB bands
    image = np.transpose(image, (1, 2, 0))  # Rearrange to (H, W, C)
    image = image.astype(np.uint8)
```

## 📖 Usage

### Basic Workflow

1. **Install Dependencies** (Section 1)
2. **Import Libraries** (Section 2)
3. **Mount Google Drive** (Section 3) - Optional
4. **Load Images** (Section 4)
5. **Run Detection Methods** (Sections 7-11)
6. **View Results** (Section 12)
7. **Export Results** (Section 14)

### Running in Google Colab

```python
# 1. Upload the notebook to Colab
# 2. Run cells sequentially using Shift+Enter
# 3. Results will be saved to Google Drive if mounted
```

### Running Locally

```bash
# Start Jupyter Notebook
jupyter notebook

# Open satellite_change_detection.ipynb
# Run cells sequentially
# Results saved to ./satellite_data/results/
```

## 🔬 Methods Explained

### 1. Image Differencing
- Simple pixel-wise subtraction between images
- Fast and computationally efficient
- Works well for significant changes
- **Best for**: Quick analysis, large structural changes

### 2. Thresholding
- Converts difference map to binary change mask
- Adjustable threshold values
- Morphological operations to reduce noise
- **Best for**: Clear change detection with minimal false positives

### 3. Change Vector Analysis (CVA)
- Analyzes spectral change magnitude and direction
- Considers all color channels simultaneously
- Percentile-based thresholding
- **Best for**: Multi-spectral satellite data, subtle changes

### 4. Deep Learning (Siamese Network)
- CNN-based feature extraction
- Learns complex change patterns
- Requires training data for best results
- **Best for**: Complex scenes, semantic change detection

## 🗂️ Custom Data

### Supported Formats

- **Standard Images**: PNG, JPG, JPEG, BMP
- **Geospatial**: GeoTIFF, TIF (via rasterio)
- **Multi-spectral**: Any format with multiple bands

### Image Requirements

- Before and after images should be:
  - Same geographic area
  - Similar resolution
  - Co-registered (aligned)
  - Same size (or will be resized automatically)

### Loading GeoTIFF Files

```python
import rasterio
import numpy as np

def load_geotiff(filepath, bands=[1, 2, 3]):
    """Load GeoTIFF file with specific bands"""
    with rasterio.open(filepath) as src:
        image = src.read(bands)
        image = np.transpose(image, (1, 2, 0))
        # Normalize if needed
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    return image

before_img = load_geotiff('before.tif', bands=[4, 3, 2])  # NIR, Red, Green
after_img = load_geotiff('after.tif', bands=[4, 3, 2])
```

## 📁 Project Structure

```
satellite-change-detection/
├── satellite_change_detection.ipynb  # Main notebook
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── satellite_data/                   # Working directory (created automatically)
│   └── results/                      # Output directory
│       ├── before_image.png
│       ├── after_image.png
│       ├── difference_map.png
│       ├── threshold_change_mask.png
│       ├── cva_change_mask.png
│       ├── dl_change_mask.png
│       ├── threshold_overlay.png
│       ├── cva_overlay.png
│       ├── dl_overlay.png
│       └── change_statistics.txt
└── sample_images/                    # Your input images (optional)
    ├── before.tif
    └── after.tif
```

## 📊 Output Files

After running the notebook, you'll get:

- **Change Masks**: Binary images showing detected changes
- **Overlay Images**: Changes highlighted on original images
- **Statistics File**: Detailed metrics (area, percentage, regions)
- **Difference Maps**: Heatmaps showing change intensity

## 🎯 Examples

### Urban Development Detection
```python
# Useful for detecting new buildings, roads, infrastructure
# Use CVA or Deep Learning methods for best results
```

### Deforestation Monitoring
```python
# Track forest loss over time
# Threshold method works well for vegetation changes
```

### Disaster Assessment
```python
# Flood extent, fire damage, earthquake impact
# Image differencing provides quick initial assessment
```

## 🔧 Customization

### Adjust Threshold Values

```python
# In Section 8, modify threshold parameter
change_mask = threshold_changes(diff_image, threshold=25)  # Lower = more sensitive
```

### Change CVA Sensitivity

```python
# In Section 9, adjust percentile
magnitude, direction, cva_mask = change_vector_analysis(
    before_proc, after_proc, 
    threshold_percentile=85  # Lower = more changes detected
)
```

### Train Deep Learning Model

```python
# Prepare your labeled training data
X_train_before = np.array([...])  # Shape: (N, H, W, 3)
X_train_after = np.array([...])   # Shape: (N, H, W, 3)
y_train = np.array([...])         # Shape: (N, H, W, 1) - binary masks

# Train the model
history = model.fit(
    [X_train_before, X_train_after],
    y_train,
    epochs=50,
    batch_size=8,
    validation_split=0.2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5),
        tf.keras.callbacks.ModelCheckpoint('best_model.h5')
    ]
)
```

## 🐛 Troubleshooting

### Common Issues

**1. Memory Error with Large Images**
```python
# Resize images before processing
target_size = (512, 512)
before_proc, after_proc, before_norm, after_norm = preprocess_images(
    before_img, after_img, target_size=target_size
)
```

**2. Images Not Aligned**
```python
# Use image registration (requires additional setup)
import cv2
# Detect features and align images
# Or use specialized tools like GDAL for geospatial data
```

**3. TensorFlow/GPU Issues in Colab**
```python
# Check GPU availability
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Use CPU if needed
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

## 💡 Tips for Best Results

1. **Image Quality**: Use high-resolution, clear images
2. **Alignment**: Ensure before/after images are properly co-registered
3. **Same Conditions**: Similar lighting, season, and atmospheric conditions
4. **Preprocessing**: Apply atmospheric correction if available
5. **Multiple Methods**: Compare results from different techniques
6. **Ground Truth**: Validate with known changes when possible

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenCV for image processing capabilities
- TensorFlow team for deep learning framework
- Rasterio for geospatial data handling
- Google Colab for free GPU access

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

## 🔗 Useful Resources

- [Rasterio Documentation](https://rasterio.readthedocs.io/)
- [OpenCV Change Detection Tutorial](https://docs.opencv.org/)
- [TensorFlow Keras Guide](https://www.tensorflow.org/guide/keras)
- [Google Earth Engine](https://earthengine.google.com/) - For downloading satellite data

---

**Happy Change Detection! 🛰️🔍**
