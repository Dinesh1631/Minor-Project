import os
from flask import Flask,render_template, request, session
import librosa
import numpy as np
import tensorflow as tf

from CNN import extract_feature, load_single_data
app = Flask(__name__)
app.secret_key = '12345'

# Load the trained model
model = tf.keras.models.load_model('cnn.h5')

# Define a function to process the audio file and make predictions
def process_audio(audio_file):
    XX, sr = librosa.load(audio_file)
    feature = extract_feature(XX, sr, mfcc=True, chroma=True, mel=True)
    XX_processed = np.expand_dims(feature, axis=0)
    XX_processed = np.expand_dims(XX_processed, axis=2)

    # Make predictions using the CNN model
    temp = model.predict(XX_processed)

    predicted_index = np.argmax(temp)
    return predicted_index



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the audio file was uploaded
        if 'audio' not in request.files:
            return 'No audio file uploaded.'
        
        audio = request.files['audio']
        
        # Save the audio file to a temporary location
        audio_path = os.path.join("C:\\Users\\dines\\OneDrive\\Desktop\\testing_files\\", audio.filename)
        audio.save(audio_path)
        
        # Process the audio file and make predictions
        predicted_label = process_audio(audio_path)
        labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        predicted_emotion = labels[predicted_label]
        
        # Store the uploaded audio file name in session
        session['uploaded_audio'] = audio.filename
        
        # Render the result.html page with the predicted label
        return render_template('result.html', predicted_label=predicted_emotion)
    
    return render_template('index.html')


if __name__=='__main__':
    # app.run(debug=True)
    app.run(debug=True, port=5000)


