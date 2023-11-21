from keras.models import load_model
from CNN import load_single_data
import numpy as np
from IPython.display import Audio

model = load_model('cnn.h5')

XX, yy = load_single_data("C:\\Users\\dines\\OneDrive\\Desktop\\minor_project\\audio_speech_actors_01-24\\Actor_09\\03-01-03-01-01-02-09.wav")
print(yy)


# Predict for the test set
XXTemp=np.expand_dims(XX, axis=2)
# XX, yy = load_single_data("C:/Users/dines/OneD/rive/Desktop/minor_project/audio_speech_actors_01-24/Actor_01/03-01-01-01-02-02-01.wav")
temp =  model.predict(XXTemp)

print(temp)