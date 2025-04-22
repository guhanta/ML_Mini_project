import os
import math
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# Constants
SAMPLE_RATE = 22050
DURATION = 30  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
NUM_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
NUM_SEGMENTS = 10

def load_audio_file(file_path):
    """Load audio file with error handling."""
    try:
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        return signal, sr
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None, None

def extract_features(signal, sr):
    """Extract MFCC features from audio signal."""
    try:
        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=NUM_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
        return mfcc.T
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        return None

def load_data(data_path):
    """Load and preprocess the audio files with error handling."""
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }
    
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(data_path)):
        if dirpath is not data_path:
            genre = os.path.basename(dirpath)
            data["mapping"].append(genre)
            print(f"Processing {genre}")
            
            for f in filenames:
                if f.endswith(".wav"):
                    file_path = os.path.join(dirpath, f)
                    signal, sr = load_audio_file(file_path)
                    
                    if signal is None or sr is None:
                        continue
                    
                    samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
                    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / HOP_LENGTH)
                    
                    for d in range(NUM_SEGMENTS):
                        start = samples_per_segment * d
                        finish = start + samples_per_segment
                        
                        mfcc = extract_features(signal[start:finish], sr)
                        if mfcc is not None and len(mfcc) == num_mfcc_vectors_per_segment:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(i-1)
    
    return data

def prepare_datasets(data):
    """Prepare training and testing datasets with data augmentation."""
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Reshape for CNN input
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    
    return X_train, X_test, y_train, y_test

def build_model(input_shape):
    """Build an improved CNN model with batch normalization."""
    model = Sequential([
        # First conv block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2), strides=(2, 2), padding='same'),
        Dropout(0.25),
        
        # Second conv block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2), strides=(2, 2), padding='same'),
        Dropout(0.25),
        
        # Third conv block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2), strides=(2, 2), padding='same'),
        Dropout(0.25),
        
        # Flatten and dense layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])
    
    return model

def plot_history(history):
    """Plot training history with improved visualization."""
    fig, axs = plt.subplots(2, figsize=(10, 8))
    
    # Create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy Evaluation")
    
    # Create error subplot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error Evaluation")
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def predict_genre(file_path, model, data_mapping):
    """Predict genre of a new audio file."""
    signal, sr = load_audio_file(file_path)
    if signal is None:
        print("Failed to load the audio file.")
        return

    samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / HOP_LENGTH)

    # Process first segment
    start = 0
    finish = samples_per_segment

    mfcc = extract_features(signal[start:finish], sr)

    if mfcc is not None and len(mfcc) == num_mfcc_vectors_per_segment:
        mfcc = mfcc[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions
        prediction = model.predict(mfcc)
        predicted_index = np.argmax(prediction)
        predicted_genre = data_mapping[predicted_index]
        print(f"Predicted Genre: {predicted_genre}")
    else:
        print("Couldn't extract proper MFCC features from the audio file.")

def visualize_sample_mfcc(data):
    """Visualize MFCC of a random sample from the dataset and other useful plots."""
    
    # --- 1. MFCC of a random sample ---
    idx = np.random.randint(0, len(data["mfcc"]))
    mfcc_sample = np.array(data["mfcc"][idx])
    label_idx = data["labels"][idx]
    genre = data["mapping"][label_idx]

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc_sample.T, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    plt.title(f"MFCC of Genre: {genre}")
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficients")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    # --- 2. Genre Distribution Plot ---
    labels = np.array(data["labels"])
    genres = [data["mapping"][label] for label in labels]

    plt.figure(figsize=(12, 6))
    sns.countplot(x=genres, order=np.unique(genres))
    plt.title("Genre Distribution")
    plt.xlabel("Genre")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # --- 3. MFCC Heatmap for the same random sample ---
    plt.figure(figsize=(12, 6))
    sns.heatmap(mfcc_sample.T, cmap='coolwarm')
    plt.title(f"MFCC Heatmap of Genre: {genre}")
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficient Index")
    plt.tight_layout()
    plt.show()

    # --- 4. Mean and Variance of MFCCs across dataset ---
    all_mfccs = np.array(data["mfcc"])
    mean_mfcc = np.mean(all_mfccs, axis=0).T
    var_mfcc = np.var(all_mfccs, axis=0).T

    fig, axs = plt.subplots(2, figsize=(10, 8))

    sns.heatmap(mean_mfcc, ax=axs[0], cmap="viridis")
    axs[0].set_title("Mean MFCCs Across Dataset")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("MFCC Coefficient")

    sns.heatmap(var_mfcc, ax=axs[1], cmap="magma")
    axs[1].set_title("Variance of MFCCs Across Dataset")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("MFCC Coefficient")

    plt.tight_layout()
    plt.show()

def main():
    # Load data
    data_path = "Data/genres_original"
    data = load_data(data_path)
    
    # Prepare datasets
    X_train, X_test, y_train, y_test = prepare_datasets(data)
    
    # Check if a trained model exists
    model_path = "music_genre_classifier.h5"
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = tf.keras.models.load_model(model_path)
    else:
        print("No existing model found. Training new model...")
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
        model = build_model(input_shape)
        
        # Compile model
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
        
        # Train model
        history = model.fit(X_train, y_train,
                           validation_data=(X_test, y_test),
                           epochs=50,
                           batch_size=32,
                           callbacks=[early_stopping, model_checkpoint, reduce_lr])
        
        # Plot history
        plot_history(history)
        
        # Save model
        model.save(model_path)
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()

    # Load the best model for prediction
    best_model = tf.keras.models.load_model("best_model.h5")

    # Visualize sample MFCCs from dataset
    data_path = "Data/genres_original"
    data = load_data(data_path)
    visualize_sample_mfcc(data)

    # Predict a new song
    new_song_path = "/Users/abhishek_r/Desktop/Programs/ML_LAB/ML_Mini_project/Data/genres_original/disco/disco.00000.wav"
    predict_genre(new_song_path, best_model, data["mapping"])