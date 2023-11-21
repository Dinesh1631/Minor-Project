# from flask import Flask, render_template, request, session
# import os
# import tensorflow as tf
# import numpy as np
# from CNN import load_single_data

# app = Flask(__name__)
# app.secret_key = '12345'

# # Load the trained model
# model = tf.keras.models.load_model('cnn.h5')

# # Define a function to process the audio file and make predictions
# def process_audio(audio_file):
#     XX, yy = load_single_data(audio_file)
#     XXTemp = np.expand_dims(XX, axis=2)
#     temp = model.predict(XXTemp)
#     predicted_index = np.argmax(temp)
#     return predicted_index


# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         # Check if the audio file was uploaded
#         if 'audio' not in request.files:
#             return 'No audio file uploaded.'
        
#         audio = request.files['audio']
        
#         # Save the audio file to a temporary location
#         audio_path = os.path.join('static', 'uploaded_audio', audio.filename)
#         audio.save(audio_path)
        
#         # Process the audio file and make predictions
#         predicted_label = process_audio(audio_path)
#         labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
#         predicted_emotion = labels[predicted_label]
        
#         # Store the uploaded audio file name in session
#         session['uploaded_audio'] = audio.filename
        
#         # Render the result.html page with the predicted label
#         return render_template('result.html', predicted_label=predicted_emotion)
    
#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)