import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix, f1_score
import pickle
import keras
from keras import layers, Sequential
from keras.layers import Conv1D, Activation, Dropout, Dense, Flatten, MaxPooling1D
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import np_utils
from keras import regularizers
from matplotlib import pyplot as plt
import seaborn as sn
import pandas as pd
import sklearn.metrics as metrics


data_directory = "C:\\Users\\dines\\OneDrive\\Desktop\\minor_project\\audio_speech_actors_01-24"

# Emotions in the RAVDESS dataset, different numbers represent different emotion
emotions = {
    '01':'neutral',
    '02':'calm',
    '03':'happy',
    '04':'sad',
    '05':'angry',
    '06':'fearful',
    '07':'disgust',
    '08':'surprised'
}

def extract_feature(data, sr, mfcc, chroma, mel):
    
    if chroma:                          
        stft = np.abs(librosa.stft(data))  
    result = np.array([])
    if mfcc:                          
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:                          
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
        result = np.hstack((result, chroma))
    if mel:                             
        mel = np.mean(librosa.feature.melspectrogram(y=data,sr=sr).T,axis=0)
        result = np.hstack((result, mel))
        
    return result 

def noise(data, noise_factor):
    
   
    noise = np.random.randn(len(data)) 
    augmented_data = data + noise_factor * noise
    
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

def shift(data, sampling_rate, shift_max, shift_direction):
    
    
    shift = np.random.randint(sampling_rate * shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shift)
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
        
    return augmented_data

def load_data(save=False):
    
   
    x, y = [], []
    for file in glob.glob(data_directory + "/Actor_*/*.wav"):
        # load an audio file as a floating point time series.    
        data, sr = librosa.load(file)
        
        # extract features from audio files into numpy array
        feature = extract_feature(data, sr, mfcc=True, chroma=True, mel=True)
        x.append(feature)

        file_name = os.path.basename(file)
        
        # get emotion label from the file name
        emotion = emotions[file_name.split("-")[2]]  
        y.append(emotion)

        # add noise to the data
        n_data = noise(data, 0.001)
        n_feature = extract_feature(n_data, sr, mfcc=True, chroma=True, mel=True)
        x.append(n_feature)
        y.append(emotion)

        # shift the data
        s_data = shift(data,sr,0.25,'right')
        s_feature = extract_feature(s_data, sr, mfcc=True, chroma=True, mel=True)
        x.append(s_feature)
        y.append(emotion)
    
    if save==True:
        np.save('X', np.array(x))
        np.save('y', y)
        
    return np.array(x), y

# Load the data and extract features for each sound file
def load_single_data(file):
    x, y = [], []
    file_name = os.path.basename(file)
    emotion = emotions[file_name.split("-")[2]]
    data, sr = librosa.load(file)
    feature = extract_feature(data, sr, mfcc=True, chroma=True, mel=True)
    x.append(feature)
    y.append(emotion)
    return np.array(x), y

if __name__ == '__main__':
    X, y = load_data(save=True)

    # X = np.load("X.npy")
    # y = np.load("y.npy")

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=9)

    labelencoder = LabelEncoder()
    labelencoder.fit(y_train)
    le_name_mapping = dict(zip(labelencoder.classes_, labelencoder.transform(labelencoder.classes_)))
    print(le_name_mapping)

    y_train = labelencoder.transform(y_train)
    y_test = labelencoder.transform(y_test)

    print(f'Features extracted: {x_train.shape[1]}')

    model = Sequential()
    model.add(Conv1D(256, 5,padding='same', input_shape=(180,1))) # 1st layer
    model.add(Activation('relu'))
    model.add(Conv1D(128, 5,padding='same', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))) # 2nd layer
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, 5,padding='same', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))) # 3rd layer
    model.add(Activation('relu'))
    model.add(Conv1D(128, 5,padding='same', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))) # 4th layer
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(units=8,
                    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                    bias_regularizer=regularizers.l2(1e-4),
                    activity_regularizer=regularizers.l2(1e-5)
                    )
    ) # 7th layer
    model.add(Activation('softmax'))


    import tensorflow 
    opt = tensorflow.keras.optimizers.legacy.Adam(decay=1e-6)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt,metrics=['accuracy'])




    XProccessed = np.expand_dims(x_train, axis=2)
    XTestProcessed = np.expand_dims(x_test, axis=2)


    history = model.fit(XProccessed, y_train, epochs=50, validation_data=(XTestProcessed, y_test), batch_size=64)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    y_pred = model.predict(XTestProcessed)

    confusion_emotions = ['angry', 'calm', 'disgust', 'fearful','happy','neutral','sad','surprised']
    cm=metrics.confusion_matrix(y_test,np.argmax(y_pred,axis=-1))


    df_cm=pd.DataFrame(cm,index=[i for i in confusion_emotions],columns=[i for i in confusion_emotions])
    plt.figure(figsize=(10,7))
    sn.heatmap(df_cm,annot=True)

    f1_score(y_test,np.argmax(y_pred,axis=-1),average='weighted')

    model.summary()

    model.save("cnn.h5")

    

    XX, yy = load_single_data("C:/Users/dines/OneDrive/Desktop/minor_project/audio_speech_actors_01-24/Actor_01/03-01-02-02-02-02-01.wav")
    # Audio(view,rate=3200)

    # Predict for the test set
    XXTemp=np.expand_dims(XX, axis=2)
    XX, yy = load_single_data("C:/Users/dines/OneDrive/Desktop/minor_project/audio_speech_actors_01-24/Actor_01/03-01-01-01-02-02-01.wav")
    temp =  model.predict(XXTemp)

    print(temp)