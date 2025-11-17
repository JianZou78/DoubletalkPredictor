# DoubletalkNet Documentation

## Overview

DoubletalkNet is a neural network system designed for analyzing audio recordings of "five" utterances in both doubletalk and singletalk segments. The system extracts sophisticated audio features and uses a neural network to predict the total number of "five" utterances heard across both segments, achieving high accuracy (R² > 0.96).

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Core Components](#core-components)
3. [Feature Extraction](#feature-extraction)
4. [Model Training](#model-training)
5. [Model Evaluation](#model-evaluation)
6. [Using the Trained Model](#using-the-trained-model)
7. [Troubleshooting](#troubleshooting)

## System Architecture

DoubletalkNet consists of four main components:

1. **Audio Feature Extractor**: Analyzes WAV files to extract time and frequency domain features
2. **Dataset Preparation**: Processes audio files and ground truth labels
3. **Neural Network Model**: Multi-layer neural network that predicts utterance counts
4. **Training & Evaluation System**: Handles model training, validation, and performance metrics

## Core Components

### AudioFeatureExtractor

This class extracts comprehensive audio features for doubletalk assessment:

- **Time Domain Features**: RMS energy, zero-crossing rate, peak amplitude
- **Frequency Domain Features**: Spectral centroid, bandwidth, rolloff
- **MFCC Features**: 13 Mel-frequency cepstral coefficients with standard deviations
- **Segment Analysis**: Separate analysis of doubletalk and singletalk regions
- **"Five" Pattern Detection**: Advanced energy and spectral consistency features

### DoubletalkNet Neural Network

A flexible neural network architecture:

- **Input Layer**: Variable size based on extracted features
- **Hidden Layers**: Configurable sizes (default: [256, 128, 64])
- **Regularization**: Dropout (0.3) and BatchNorm/LayerNorm
- **Output**: Single value (total "five" count prediction)

### DoubletalkTrainer

The main trainer class that handles:

- Data preparation and feature scaling
- Model initialization and training
- Performance evaluation and visualization
- Model saving and loading

## Feature Extraction

The system extracts over 100 sophisticated audio features:

- **Segment-specific features**: Separate analysis for doubletalk and singletalk segments
- **Robustness features**: Error handling for various audio quality issues
- **Delay detection**: Automatic alignment of recorded audio with source reference
- **Pattern analysis**: Specialized features for detecting "five" utterance patterns

Example feature extraction:

```python
# Load and process audio
audio, sr = librosa.load(wav_file, sr=16000)
features = feature_extractor.extract_doubletalk_features(audio)
```

## Model Training

Training a new model requires:

1. **Preparing audio data**: WAV files with consistent format (will be converted to 16kHz)
2. **Ground truth labels**: Number of "five" utterances heard in each recording
3. **Optional source reference**: For delay detection and compensation

Example training process:

```python
# Initialize trainer
trainer = DoubletalkTrainer()

# Prepare data
X, y = trainer.prepare_data(wav_files, labels, source_file="reference.wav")

# Train model
metrics = trainer.train_model(X, y, epochs=100, batch_size=32)

# Save model
trainer.save_model("doubletalk_model.pth")
```

## Model Evaluation

The system evaluates model performance using:

- **Mean Squared Error (MSE)**: Overall prediction error
- **Mean Absolute Error (MAE)**: Average absolute difference between predictions and ground truth
- **R² Score**: Coefficient of determination (model explains > 96% of variance)
- **Visualization**: Training curves and prediction scatter plots

## Using the Trained Model

### Prerequisites

- Python 3.6+
- Required packages: torch, numpy, pandas, librosa, scikit-learn

### Basic Usage

1. **Load the trained model**:

```python
from doubletalk_ai_trainer import DoubletalkTrainer

# Initialize trainer and load model
trainer = DoubletalkTrainer()
trainer.load_model("doubletalk_model.pth")
```

2. **Make predictions on new audio**:

```python
# Predict for a single file
result = trainer.predict("new_recording.wav")
print(f"Predicted 'five' count: {result:.1f}")

# Optional: Provide source file for automatic delay detection
result = trainer.predict("new_recording.wav", source_file="reference.wav")
```

3. **Batch prediction**:

```python
# Process multiple files
results = []
for wav_file in wav_files:
    prediction = trainer.predict(wav_file)
    results.append((wav_file, prediction))
    
# Export results
import pandas as pd
df = pd.DataFrame(results, columns=["File", "Prediction"])
df.to_csv("predictions.csv", index=False)
```

### Advanced Usage

#### Custom Delay Compensation

If you know the recording delay, you can specify it directly:

```python
# Provide known delay in seconds (positive if recorded audio is delayed)
result = trainer.predict("new_recording.wav", delay_compensation=0.75)
```

#### Audio Validation

Validate audio files before processing:

```python
from doubletalk_ai_trainer import validate_audio_file

# Check audio properties
info = validate_audio_file("recording.wav")
if info["valid"]:
    print(f"Valid audio: {info['duration']:.1f}s, {info['sample_rate']}Hz")
else:
    print(f"Invalid audio: {info['error']}")
```

## Troubleshooting

### Common Issues

#### "Model not trained yet" Error

This error occurs when attempting to use prediction methods before loading a trained model:

```python
# Solution: Load the model first
trainer.load_model("doubletalk_model.pth")
```

#### PyTorch 2.6+ Loading Issues

With PyTorch 2.6+, you may encounter security-related model loading errors:

```
RuntimeError: Weights only load failed: sklearn.preprocessing._data.StandardScaler isn't allowed in a weights-only load
```

Solution: The latest version of DoubletalkTrainer handles this automatically with the updated `load_model()` method:

```python
# Will work with PyTorch 2.6+
trainer.load_model("doubletalk_model.pth")
```

#### Audio Format Issues

The system automatically handles most audio format issues, including:
- Resampling to 16kHz
- Converting stereo to mono
- Normalizing audio levels
- Padding short recordings

For very problematic audio files, use the validation function first:

```python
info = validate_audio_file("problematic.wav")
print(f"Audio info: {info}")
```

## Best Practices

1. **Consistent audio format**: While the system handles format conversion, consistent input files produce better results.

2. **Delay compensation**: For best accuracy, provide the source file or specify known delay.

3. **Regular updates**: Check for updates to the model and training system as improvements are made.

4. **Batch processing**: For large datasets, process files in batches to improve efficiency.

5. **Model retraining**: If your audio characteristics change significantly, consider retraining the model with your specific data.
