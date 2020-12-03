# loading json and creating model
from keras.models import model_from_json
import keras
import librosa
import numpy as np
from tensorflow.keras import optimizers
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("saved_models/Emotion_Voice_Detection_Model.h5")
print("Loaded model from disk")
data, sampling_rate = librosa.load('output10.wav')
import os
import pandas as pd
import librosa
import glob 

#plt.figure(figsize=(15, 5))
#librosa.display.waveplot(data, sr=sampling_rate)
#livedf= pd.DataFrame(columns=['feature'])
X, sample_rate = librosa.load('Input4.m4a', res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
sample_rate = np.array(sample_rate)
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
featurelive = mfccs
livedf2 = featurelive
#import pandas as pd
livedf2= pd.DataFrame(data=livedf2)
livedf2 = livedf2.stack().to_frame().T
twodim= np.expand_dims(livedf2, axis=2)
livepreds = loaded_model.predict(twodim, 
                         batch_size=32, 
                         verbose=1)
livepreds1=livepreds.argmax(axis=1)
liveabc = livepreds1.astype(int).flatten()
if liveabc == 0:
	print("Female_angry")
elif liveabc == 1:
	print("Female Calm")
elif liveabc == 2:
	print("Female Fearful")
elif liveabc == 3:
	print("Female Happy")
elif liveabc == 4:
	print("Female Sad")
elif liveabc == 5:
	print("Male Angry")
elif liveabc == 6:
	print("Male calm")
elif liveabc == 7:
	print("Male Fearful")
elif liveabc == 8:
	print("Male Happy")
elif liveabc == 9:
	print("Male sad")
