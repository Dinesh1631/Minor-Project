import librosa
import numpy as np
import keras
from keras.models import load_model

from CNN import extract_feature

# Load the pre-trained CNN model
model = load_model("cnn.h5")

# List of emotions
list = ['neutral','calm','happy','sad','angry','fearful','disgust','surprised']

# Load and process the audio file for prediction
XX, sr = librosa.load("C:\\Users\\dines\\OneDrive\\Desktop\\testing_files\\03-01-07-02-01-02-08.wav")
feature = extract_feature(XX, sr, mfcc=True, chroma=True, mel=True)
XX_processed = np.expand_dims(feature, axis=0)
XX_processed = np.expand_dims(XX_processed, axis=2)

# Make predictions using the CNN model
temp = model.predict(XX_processed)

predicted_index = np.argmax(temp)

# Convert the predicted emotion from numerical representation to label
print("Predicted emotion:", list[predicted_index])
