# Music Genre Classification using Neural Networks

This project implements a Convolutional Neural Network (CNN) to classify music genres from audio files. The model uses Mel-Frequency Cepstral Coefficients (MFCCs) as features to train a deep learning model that can accurately predict the genre of a given audio clip.

## Dataset

The project uses the GTZAN Genre Collection Dataset, which contains:
- 1000 audio files (30 seconds each)
- 10 genres: Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock

## Features

- Audio preprocessing using Librosa
- MFCC feature extraction
- CNN-based deep learning model
- Model evaluation and visualization
- Early stopping to prevent overfitting

## Requirements

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Project Structure

- `music_genre_classification.py`: Main script containing the implementation
- `requirements.txt`: List of required Python packages
- `Data/`: Directory containing the GTZAN dataset

## Usage

1. Make sure you have the GTZAN dataset in the `Data/genres_original` directory
2. Run the main script:
```bash
python music_genre_classification.py
```

The script will:
- Load and preprocess the audio files
- Extract MFCC features
- Train the CNN model
- Evaluate the model's performance
- Save the trained model as `music_genre_classifier.h5`

## Model Architecture

The CNN model consists of:
- 3 Convolutional layers with MaxPooling and Dropout
- Flatten layer
- Dense layers with Dropout
- Output layer with softmax activation

## Performance

The model achieves an accuracy of approximately 70-90% on the test set, depending on the training parameters and data preprocessing.

## Visualization

The script generates plots showing:
- Training and validation accuracy over epochs
- Training and validation loss over epochs 