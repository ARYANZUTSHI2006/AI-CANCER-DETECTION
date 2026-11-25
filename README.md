# 🏥 AI-Based Cancer Detection from Chest X-rays

A full-stack deep learning application for detecting cancer from chest X-ray images using PyTorch, React, and Node.js with automatic class balancing for imbalanced datasets.

**Authors:** Sneh Gupta and Arpit Bhardwaj  
**Course:** CSET211 - Statistical Machine Learning  
**Institution:** Bennett University

---

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Training Models](#training-models)
- [Using the Web Application](#using-the-web-application)
- [API Documentation](#api-documentation)
- [Model Performance](#model-performance)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## ✨ Features

### 🤖 Deep Learning
- **Multiple Architectures**: ResNet50, DenseNet121, EfficientNet support
- **Automatic Class Balancing**: WeightedRandomSampler handles imbalanced datasets (17% cancer, 83% normal)
- **Transfer Learning**: Pre-trained ImageNet weights for better performance
- **Data Augmentation**: Rotation, flips, brightness/contrast adjustments
- **Early Stopping**: Prevents overfitting with patience-based stopping

### 🌐 Web Application
- **Modern React UI**: Beautiful, responsive interface
- **Real-time Predictions**: Fast inference with confidence scores
- **Grad-CAM Visualization**: Visual explanations of model decisions
- **Batch Processing**: Analyze multiple X-rays at once
- **RESTful API**: Easy integration with other systems

### 📊 Training & Evaluation
- **TensorBoard Integration**: Real-time training monitoring
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, AUC-ROC, PR-AUC
- **Confusion Matrix**: Detailed performance visualization
- **Google Colab Support**: Free GPU training with detailed guide
- **Checkpoint Management**: Automatic best model saving

---

## 📁 Project Structure

```
AI_Cancer_Detection/
├── README.md                          # This file
├── START_APP.bat                      # Quick start (Windows)
├── start-dev.bat                      # Development mode
│
├── ai/                                # AI/ML Backend
│   ├── main.py                        # Training/evaluation entry point
│   ├── requirements.txt               # Python dependencies
│   ├── environment.yml                # Conda environment
│   ├── COLAB_TRAINING_GUIDE.md        # Google Colab training guide
│   ├── TRAIN_MODEL.bat                # Local training script
│   ├── MONITOR_TRAINING.bat           # Launch TensorBoard
│   ├── train_densenet.bat             # DenseNet121 training
│   │
│   ├── configs/                       # Configuration files
│   │   ├── config.yaml                # Main configuration
│   │   └── model_config.yaml          # Model-specific config
│   │
│   ├── src/                           # Source code
│   │   ├── data_loader.py             # Data loading with WeightedRandomSampler
│   │   ├── models.py                  # Model architectures
│   │   ├── train.py                   # Training logic
│   │   ├── evaluate.py                # Evaluation metrics
│   │   ├── inference.py               # Inference pipeline
│   │   ├── gradcam.py                 # Grad-CAM visualization
│   │   └── utils.py                   # Utility functions
│   │
│   ├── data/                          # Dataset directory
│   │   └── raw/
│   │       ├── train_data/train/      # 5,641 chest X-ray images
│   │       └── ChestXray_Binary_Labels.csv  # Binary labels
│   │
│   ├── notebooks/                     # Jupyter notebooks
│   │   ├── baseline_model.ipynb       # Baseline experiments
│   │   └── data_exploration.ipynb     # Data analysis
│   │
│   ├── tests/                         # Unit tests
│   │   └── test_data_loader.py
│   │
│   └── utils/                         # Utility modules
│       └── checkpoint.py              # Checkpoint loading
│
├── backend/                           # Node.js API Server
│   ├── server.js                      # Express server
│   ├── package.json                   # Node dependencies
│   ├── .env                           # Environment variables
│   └── uploads/                       # Temporary upload storage
│
└── frontend/                          # React Web Application
    ├── src/
    │   ├── App.jsx                    # Main component
    │   ├── index.jsx                  # Entry point
    │   └── index.css                  # Styles
    ├── package.json                   # Frontend dependencies
    └── .env                           # Frontend configuration
```

---

## 🔧 Prerequisites

### Required Software
- **Python**: 3.8 or higher
- **Node.js**: 14.0 or higher
- **npm**: 6.0 or higher
- **Git**: For version control

### Optional (for GPU training)
- **CUDA Toolkit**: 11.0+ (NVIDIA GPUs only)
- **cuDNN**: 8.0+
- **Google Colab Account**: For free GPU training

### Hardware Requirements
- **Minimum**: 8GB RAM, 10GB disk space
- **Recommended**: 16GB RAM, 20GB disk space, NVIDIA GPU with 6GB+ VRAM

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd AI_Cancer_Detection
```

### 2. Set Up Python Environment

```bash
# Navigate to ai folder
cd ai

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

cd ..
```

### 3. Set Up Backend

```bash
cd backend
npm install

# backend/.env defaults to the included ResNet50 checkpoint in ../Models
# To use ResNet101 instead, set CHECKPOINT_PATH=../Models/best_model_resnet101_colab.pth

cd ..
```

### 4. Set Up Frontend

```bash
cd frontend
npm install

# Vite proxies API requests to the backend on port 3000 by default

cd ..
```

### 5. Start the Application

**Option A: Quick Start (Windows)**
```bash
START_APP.bat
```

**Option B: Manual Start**

Terminal 1 - Backend:
```bash
cd backend
npm start
```

Terminal 2 - Frontend:
```bash
cd frontend
npm run dev
```

### 6. Access the Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:3000
- **API Documentation**: http://localhost:3000

---

## 🎓 Training Models

### Dataset Information

**Current Dataset:**
- **Total Images**: 5,641 chest X-rays
- **Cancer Cases**: 985 (17.5%)
- **Normal Cases**: 4,656 (82.5%)
- **Format**: PNG/JPG images
- **Labels**: Binary classification (Cancer / No Cancer)

**Class Imbalance Handling:**
- Uses `WeightedRandomSampler` to balance training batches
- Cancer images sampled ~5× more frequently
- Creates ~50/50 balanced batches automatically
- No manual data duplication needed

### Training Options

#### Option 1: Google Colab (Recommended - FREE GPU)

**Advantages:**
- ✅ FREE Tesla T4 GPU
- ✅ 10-15× faster than CPU
- ✅ No local setup required
- ✅ Can close browser, training continues

**Steps:**

1. **Prepare Project ZIP**
   ```powershell
   cd C:\AI_Lung_Cancer\AI_Cancer_Detection
   Compress-Archive -Path "ai" -DestinationPath "ai.zip" -Force
   ```

2. **Upload to Google Drive**
   - Create folder: `My Drive/AI_Cancer_Detection/`
   - Upload `ai.zip`

3. **Follow Colab Guide**
   - See detailed instructions in: `ai/COLAB_TRAINING_GUIDE.md`
   - 12 ready-to-use cells
   - Automatic setup and training
   - **Training time**: ~2-4 hours for 50 epochs

4. **Download Trained Model**
   - Model saved to Google Drive automatically
   - Download and place in: `ai/experiments/<experiment_name>/checkpoints/`

#### Option 2: Local Training

**CPU Training:**
```bash
cd ai
.venv\Scripts\activate
python main.py train --config configs/config.yaml --experiment-name my_model --device cpu
```
⏱️ **Time**: ~40-50 hours for 50 epochs

**GPU Training (if available):**
```bash
python main.py train --config configs/config.yaml --experiment-name my_model --device cuda
```
⏱️ **Time**: ~4-5 hours for 50 epochs (RTX 3060)

**Using Batch Scripts:**
- **ResNet50**: Double-click `ai/TRAIN_MODEL.bat`
- **DenseNet121**: Double-click `ai/train_densenet.bat`

### Training Configuration

Edit `ai/configs/config.yaml` to customize:

```yaml
# Model selection
model:
  architecture: "resnet50"  # Options: resnet50, densenet121, efficientnet
  pretrained: true
  dropout: 0.5

# Training parameters
training:
  epochs: 50
  batch_size: 32  # Reduce to 16 if out of memory
  learning_rate: 0.0001
  early_stopping_patience: 15

# Data augmentation
data:
  image_size: 224
  augmentation:
    rotation_range: 15
    horizontal_flip: true
    brightness_range: [0.8, 1.2]
```

### Monitor Training

**TensorBoard:**
```bash
cd ai
.venv\Scripts\activate
tensorboard --logdir experiments
```
Open: http://localhost:6006

**Or use batch script:**
```bash
ai\MONITOR_TRAINING.bat
```

### Evaluate Trained Model

```bash
cd ai
.venv\Scripts\activate
python main.py evaluate \
    --config configs/config.yaml \
    --checkpoint experiments/my_model_*/checkpoints/best_model.pth \
    --device auto
```

---

## 🌐 Using the Web Application

### Upload and Analyze X-rays

1. **Open Application**: http://localhost:5173
2. **Upload Image**: Click "Choose File" or drag-and-drop
3. **Analyze**: Click "Analyze X-ray"
4. **View Results**:
   - Prediction: Cancer / No Cancer
   - Confidence Score: 0-100%
   - Grad-CAM Visualization: Highlighted regions of interest

### Supported Image Formats
- PNG, JPG, JPEG
- Recommended size: 224×224 to 1024×1024 pixels
- Grayscale or RGB

### Batch Processing

Upload multiple X-rays (up to 10) for batch analysis:
```javascript
// Using API directly
const formData = new FormData();
formData.append('images', file1);
formData.append('images', file2);

fetch('http://localhost:3000/predict-batch', {
  method: 'POST',
  body: formData
});
```

---

## 📡 API Documentation

### Base URL
```
http://localhost:3000
```

### Endpoints

#### `GET /`
Get API information and available endpoints.

**Response:**
```json
{
  "message": "Cancer Detection API",
  "version": "1.0.0",
  "endpoints": {
    "/health": "Health check",
    "/predict": "Single image prediction",
    "/predict-batch": "Batch prediction",
    "/status": "Server status"
  }
}
```

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2025-11-03T12:00:00.000Z"
}
```

#### `POST /predict`
Analyze a single chest X-ray image.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `image` (file)

**Response:**
```json
{
  "success": true,
  "result": {
    "prediction": "Cancer",
    "probability": 0.8523,
    "confidence": 0.8523,
    "message": "High confidence cancer detection"
  },
  "timestamp": "2025-11-03T12:00:00.000Z"
}
```

#### `POST /predict-batch`
Analyze multiple images (up to 10).

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `images[]` (multiple files)

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "filename": "xray1.png",
      "prediction": "Cancer",
      "probability": 0.85
    },
    {
      "filename": "xray2.png",
      "prediction": "No Cancer",
      "probability": 0.12
    }
  ],
  "timestamp": "2025-11-03T12:00:00.000Z"
}
```

#### `GET /status`
Get server and model status.

**Response:**
```json
{
  "status": "running",
  "modelAvailable": true,
  "inferenceEngine": "onnx",
  "modelError": null,
  "modelPath": "/tmp/cancer-detection.onnx",
  "checkpointPath": "../ai/experiments/.../best_model.pth",
  "timestamp": "2025-11-03T12:00:00.000Z"
}
```

For a Node-only deployment, set `ONNX_MODEL_URL` to an HTTPS ONNX release asset and
`ONNX_MODEL_SHA256` to its 64-character SHA-256 digest. The backend downloads the
model at startup, verifies the digest, and runs CPU inference with ONNX Runtime.
The public inference-only model used by this deployment is
[cancer_resnet50.onnx](https://github.com/ARYANZUTSHI2006/ai-cancer-detection-onnx-model/releases/download/v1.0.0/cancer_resnet50.onnx);
its digest is `7988880c70a893eb08b38adc915b378045ff77fc441ca845c538fbd383dcbde4`.
ONNX mode returns predictions without a Grad-CAM image (`gradcamImage` is `null`).

### Error Responses

```json
{
  "success": false,
  "error": "Error message here",
  "timestamp": "2025-11-03T12:00:00.000Z"
}
```

**Common Error Codes:**
- `400`: Bad Request (invalid input)
- `404`: Not Found (endpoint doesn't exist)
- `500`: Internal Server Error (model/server error)

---

## 📊 Model Performance

### Current Best Model: ResNet50

**Training Details:**
- **Dataset**: 5,641 chest X-rays (17.5% cancer, 82.5% normal)
- **Architecture**: ResNet50 (pre-trained on ImageNet)
- **Training Method**: WeightedRandomSampler for class balancing
- **Training Time**: ~2.5 hours on Google Colab T4 GPU
- **Epochs**: 17 (early stopping from 50)

**Performance Metrics:**
- **ROC-AUC**: 0.6922 (Moderate)
- **Accuracy**: 76.15%
- **Precision**: 38.46%
- **Recall**: 60.81% (detects 61% of cancer cases)
- **Specificity**: 79.40%
- **F1-Score**: 0.4712

**Interpretation:**
- ✅ **Good for screening**: High recall catches most cancer cases
- ⚠️ **Some false positives**: Lower precision means some healthy patients flagged
- 📈 **Room for improvement**: AUC of 0.69 is moderate, target is 0.85+

### Model Comparison

| Model | Parameters | AUC | Training Time (Colab) | Memory |
|-------|------------|-----|----------------------|--------|
| ResNet50 | 25M | 0.69 | 2.5 hours | High |
| DenseNet121 | 8M | TBD | ~2 hours | Lower |
| EfficientNet-B0 | 5M | TBD | ~2 hours | Lowest |

### Improving Performance

**To achieve better results (AUC > 0.85):**

1. **More Training Data**
   - Current: 5,641 images
   - Target: 10,000+ images
   - More cancer cases (currently only 985)

2. **Longer Training**
   - Try 100-200 epochs
   - Use learning rate scheduling
   - Experiment with different optimizers

3. **Advanced Techniques**
   - Ensemble multiple models
   - Test-time augmentation
   - Focal loss for hard examples
   - Mix-up or CutMix augmentation

4. **Different Architectures**
   - Try DenseNet121 (often better for medical images)
   - EfficientNet-B0 (more efficient)
   - Vision Transformers (ViT)

---

## 🔍 Troubleshooting

### Common Issues

#### Backend: Model Checkpoint Not Found

**Error:**
```
Error: Model checkpoint not found
```

**Solution:**
1. Train a model first (see [Training Models](#training-models))
2. Or download pre-trained model
3. Update `backend/.env`:
   ```env
   CHECKPOINT_PATH=../ai/experiments/<experiment_name>/checkpoints/best_model.pth
   ```

#### Backend: Port Already in Use

**Error:**
```
Error: Port 3000 already in use
```

**Solution:**
```bash
# Option 1: Kill process on port 3000
# Windows:
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# Option 2: Change port in backend/.env
PORT=3001
```

#### Frontend: Cannot Connect to Backend

**Error:**
```
Network Error / Cannot connect to API
```

**Solution:**
1. Ensure backend is running: `cd backend && npm start`
2. Check backend URL in `frontend/.env`:
   ```env
   VITE_API_URL=http://localhost:3000
   ```
3. Verify backend is accessible: http://localhost:3000

#### Python: CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
1. Reduce batch size in `ai/configs/config.yaml`:
   ```yaml
   data:
     batch_size: 16  # or even 8
   ```
2. Or use CPU:
   ```bash
   python main.py train --config configs/config.yaml --device cpu
   ```

#### Python: Module Not Found

**Error:**
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**
```bash
cd ai
.venv\Scripts\activate
pip install -r requirements.txt
```

#### Windows: num_workers Error

**Error:**
```
RuntimeError: DataLoader worker (pid XXXX) is killed by signal
```

**Solution:**
Already fixed in `config.yaml`:
```yaml
data:
  num_workers: 0  # Set to 0 for Windows
```

#### Colab: numpy.dtype size changed

**Error:**
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility
```

**Solution:**
Run Cell 1 (dependencies) FIRST before any other cells:
```python
!pip uninstall -y numpy pandas scikit-learn -q
!pip install -q numpy==1.24.3 pandas==2.0.3 scikit-learn==1.3.2
```

---

## 🛠️ Development

### Running Tests

```bash
# Python tests
cd ai
.venv\Scripts\activate
pytest tests/

# Or specific test
python -m pytest tests/test_data_loader.py -v
```

### Code Style

**Python:**
- Follow PEP 8
- Use type hints where possible
- Docstrings for all functions

**JavaScript:**
- ESLint configuration included
- Prettier for formatting

**React:**
- Functional components with hooks
- PropTypes for type checking

### Adding New Models

1. **Define model in `ai/src/models.py`:**
   ```python
   class MyCustomModel(nn.Module):
       def __init__(self, num_classes=1):
           super().__init__()
           # Define layers
   ```

2. **Register in `get_model()` function:**
   ```python
   elif architecture == "my_custom_model":
       model = MyCustomModel(num_classes=num_classes)
   ```

3. **Update config:**
   ```yaml
   model:
     architecture: "my_custom_model"
   ```

### Project Conventions

- **Experiments**: Saved in `ai/experiments/<name>_<timestamp>/`
- **Checkpoints**: `experiments/<name>/checkpoints/best_model.pth`
- **Logs**: `experiments/<name>/logs/`
- **Results**: `experiments/<name>/results/`

---

## 📝 Configuration Files

### `ai/configs/config.yaml`

Main configuration file for training and data:

```yaml
# Data configuration
data:
  dataset_path: "data/raw/train_data/train"
  labels_file: "data/raw/ChestXray_Binary_Labels.csv"
  image_size: 224
  batch_size: 32
  num_workers: 0  # 0 for Windows, 4+ for Linux
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15

# Model configuration
model:
  architecture: "resnet50"
  pretrained: true
  num_classes: 1
  dropout: 0.5

# Training configuration
training:
  epochs: 50
  learning_rate: 0.0001
  weight_decay: 0.0001
  optimizer: "adam"
  scheduler: "cosine"
  early_stopping_patience: 15

# Loss function
loss:
  type: "bce_with_logits"
  pos_weight: 4.73  # Calculated from class imbalance
```

### `backend/.env`

Backend configuration:

```env
PORT=3000
CHECKPOINT_PATH=../ai/experiments/<experiment_name>/checkpoints/best_model.pth
CONFIG_PATH=../ai/configs/config.yaml
```

### `frontend/.env`

Frontend configuration:

```env
VITE_API_URL=http://localhost:3000
```

---

## 🤝 Contributing

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Run tests**
   ```bash
   pytest tests/
   npm test
   ```
5. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
6. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request**

### Areas for Contribution

- 🎯 **Model Improvements**: New architectures, better hyperparameters
- 📊 **Data**: More labeled chest X-rays
- 🌐 **UI/UX**: Better frontend design
- 📝 **Documentation**: Tutorials, guides, examples
- 🐛 **Bug Fixes**: Report and fix issues
- ✨ **Features**: New functionality

---

## 📄 License

MIT License - See LICENSE file for details.

---

## 👥 Authors & Acknowledgments

**Authors:**
- **Sneh Gupta** - AI/ML Development, Model Training
- **Arpit Bhardwaj** - Full-stack Development, System Integration

**Course:** CSET211 - Statistical Machine Learning  
**Institution:** Bennett University  
**Supervisor:** [Dr. Nitin Arvind Shelke]

**Acknowledgments:**
- NIH Chest X-ray Dataset
- PyTorch and torchvision teams
- React and Vite communities
- Google Colab for free GPU access
- All open-source contributors

---

## 📞 Support & Contact

**For issues and questions:**
1. Check the [Troubleshooting](#troubleshooting) section
2. Review [Colab Training Guide](ai/COLAB_TRAINING_GUIDE.md)
3. Open an issue on GitHub
4. Contact the development team

**Email:** [your-email@example.com]  
**GitHub:** [repository-url]

---

## ⚠️ Medical Disclaimer

**IMPORTANT:** This system is intended for **research and educational purposes only**. It is **NOT** a substitute for professional medical diagnosis. 

- ❌ Do NOT use for clinical decision-making
- ❌ Do NOT use without consulting qualified healthcare professionals
- ❌ Do NOT rely solely on AI predictions for medical diagnosis

Always consult qualified healthcare professionals for medical advice, diagnosis, and treatment.

---

## 🗺️ Roadmap

### Current Version: 1.0.0

**Completed:**
- ✅ Basic cancer detection model
- ✅ Web application interface
- ✅ RESTful API
- ✅ Google Colab training support
- ✅ Automatic class balancing
- ✅ TensorBoard integration

**Planned Features:**

**Version 1.1.0:**
- [ ] Multi-class classification (different cancer types)
- [ ] Improved model ensemble
- [ ] User authentication
- [ ] Prediction history

**Version 1.2.0:**
- [ ] DICOM file support
- [ ] Advanced Grad-CAM visualizations
- [ ] Model comparison dashboard
- [ ] Export reports (PDF)

**Version 2.0.0:**
- [ ] 3D CT scan support
- [ ] Real-time video analysis
- [ ] Mobile app (React Native)
- [ ] Cloud deployment (AWS/Azure)

---

## 📚 Additional Resources

### Documentation
- [Colab Training Guide](ai/COLAB_TRAINING_GUIDE.md) - Detailed GPU training instructions
- [AI Module README](ai/README.md) - AI-specific documentation
- [API Documentation](#api-documentation) - Complete API reference

### Tutorials
- Training your first model
- Improving model performance
- Deploying to production
- Creating custom models

### Research Papers
- ResNet: "Deep Residual Learning for Image Recognition"
- DenseNet: "Densely Connected Convolutional Networks"
- Grad-CAM: "Visual Explanations from Deep Networks"
- Class Imbalance: "A Survey on Deep Learning with Class Imbalance"

### Datasets
- NIH Chest X-ray Dataset
- ChestX-ray14
- MIMIC-CXR
- CheXpert

---

**Made with ❤️ by Sneh Gupta and Arpit Bhardwaj**

**Last Updated:** November 3, 2025


# efficientnet
# CHECKPOINT_PATH=..\ai\experiments\colab_efficientnet_b1_balanced_20251103_112600\checkpoints\best_model.pth

# resnet50
# CHECKPOINT_PATH=..\ai\experiments\colab_resnet50_balanced_20251101_135255\checkpoints\best_model_colab.pth