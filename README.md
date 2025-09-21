# Signature Verification System ✍️

A comprehensive signature verification system that uses machine learning to authenticate handwritten signatures. This system combines computer vision techniques with neural networks to detect forged signatures with high accuracy.

## 🚀 Features

- **Real-time Signature Verification**: Upload and verify signatures instantly through a web interface
- **Multiple ML Models**: Choose between custom neural network implementation and TensorFlow-based models
- **Image Preprocessing**: Advanced image processing pipeline for optimal signature analysis
- **Web Interface**: User-friendly Flask web application for easy interaction
- **High Accuracy**: Trained on professional signature datasets for reliable results

## 🛠️ Technologies Used

- **Python 3.10+** - Core programming language
- **Flask 2.3.3** - Web framework for the user interface
- **TensorFlow 2.13.0** - Machine learning framework
- **OpenCV 4.8.1** - Computer vision and image processing
- **NumPy 1.24.3** - Numerical computing
- **Pillow 10.0.1** - Image handling and manipulation

## 📊 Dataset

The system is trained using the [ICDAR 2009 Signature Verification Competition (SigComp2009)](http://www.iapr-tc11.org/mediawiki/index.php?title=ICDAR_2009_Signature_Verification_Competition_(SigComp2009)) dataset, which provides:
- Genuine signature samples
- Skilled forgery samples
- Professional benchmark for signature verification systems

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/V-S-Sanjana/signature-verification.git
   cd signature-verification
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the web application**
   ```bash
   python app.py
   ```
   Or use the batch file:
   ```bash
   run_app.bat
   ```

4. **Open your browser**
   Navigate to `http://localhost:5000` to access the web interface

## 💻 Usage

### Web Interface
1. Open the application in your browser
2. Upload a signature image for verification
3. View the verification results and confidence score
4. Check sample signatures for reference

### Command Line
- **Custom Neural Network**: `python sigrecog.py`
- **TensorFlow Model**: `python sigrecogtf.py`

## 📁 Project Structure

```
signature-verification/
├── app.py                 # Flask web application
├── network.py            # Custom neural network implementation
├── sigrecogtf.py         # TensorFlow model implementation
├── preprocessor.py       # Image preprocessing utilities
├── requirements.txt      # Python dependencies
├── run_app.bat          # Windows batch file to run the app
├── templates/           # HTML templates
│   ├── index.html       # Main interface
│   └── samples.html     # Sample signatures page
├── data/               # Training and test datasets
│   ├── training/       # Training data
│   └── test/          # Test data
└── uploads/           # Uploaded signature storage
```

## 🔬 How It Works

1. **Image Preprocessing**: Uploaded signatures are processed using OpenCV to:
   - Convert to grayscale
   - Apply noise reduction
   - Normalize image dimensions
   - Extract signature features

2. **Feature Extraction**: The system extracts key features from signatures:
   - Geometric features (aspect ratio, density)
   - Texture features
   - Statistical features

3. **Classification**: Machine learning models analyze features to determine:
   - Authenticity probability
   - Confidence score
   - Similarity to known genuine signatures

## 📈 Model Performance

The system achieves high accuracy rates on the ICDAR 2009 dataset:
- **Genuine Acceptance Rate**: High precision for authentic signatures
- **Skilled Forgery Detection**: Effective detection of sophisticated forgeries
- **Processing Speed**: Real-time verification capabilities

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**V.S. Sanjana**
- GitHub: [@V-S-Sanjana](https://github.com/V-S-Sanjana)
- Repository: [signature-verification](https://github.com/V-S-Sanjana/signature-verification)

## 🙏 Acknowledgments

- ICDAR 2009 Signature Verification Competition for providing the dataset
- OpenCV and TensorFlow communities for excellent documentation
- Flask framework for enabling rapid web development

---

⭐ **Star this repository if you found it helpful!** ⭐


