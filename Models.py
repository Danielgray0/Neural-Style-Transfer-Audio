
from keras.layers import LSTM, Dense, Activation, Dropout, Flatten
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import numpy as np
from mido import MidiFile,MidiTrack,Message
from midi2audio import FluidSynth
import os
import copy as cp

n_prev = 400
def PrepareData(file):
    mid=MidiFile(file)
    notes = []
    velocities = []
    for msg in mid:
     if not msg.is_meta:
            if msg.channel == 0:
                if msg.type == 'note_on':
                    data = msg.bytes()
                    # append note and velocity from [type, note, velocity]
                    note = data[1]
                    velocity = data[2]
                    notes.append(note)
                    velocities.append(velocity)
    combine = [[i,j] for i,j in zip(notes, velocities)]
    note_min = np.min(notes)
    note_max = np.max(notes)
    velocities_min = np.min(velocities)
    velocities_max = np.max(velocities)

    for i in combine:
        i[0] = 2*(i[0]-((note_min+note_max)/2))/(note_max-note_min)
        i[1] = 2*(i[1]-((velocities_min+velocities_max)/2))/(velocities_max-velocities_min)
    X = []
    Y = []
    for i in range(len(combine) - n_prev):
        x = combine[i:i + n_prev]
        y = combine[i + n_prev]
        X.append(x)
        Y.append(y)
    # save a seed to do prediction later
    seed = combine[0:n_prev]
    return X,Y,note_max,note_min,velocities_max,velocities_min,seed


model = Sequential()
model.add(LSTM(256, input_shape=(n_prev, 2), return_sequences=True))
model.add(Dropout(0.6))
model.add(LSTM(128, input_shape=(n_prev, 2), return_sequences=True))
model.add(Dropout(0.6))
model.add(LSTM(64, input_shape=(n_prev, 2), return_sequences=False))
model.add(Dropout(0.6))
model.add(Dense(2))
model.add(Activation('linear'))
optimizer = Adam(lr=0.001)
model.compile(loss='mse', optimizer=optimizer)
filepath="./Checkpoints/checkpoint_model_{epoch:02d}.hdf5"
model_save_callback = ModelCheckpoint(filepath, monitor='val_acc',
                                      verbose=1, save_best_only=False,
                                      mode='auto', period=5)
# X,Y,note_max,note_min,velocities_max,velocities_min,seed=PrepareData('./Nintendo_-_Pokemon_Fire_Red_Route_1_Piano_Cover_Hard_Version.mid')
# model.fit(np.array(X),np.array(Y),32,10,verbose=1,callbacks=[model_save_callback])

for f in os.listdir('./Midi_Data'):
    if f.endswith('.mid'):
        file ='./Midi_Data/' + f
        X, Y, note_max, note_min, velocities_max, velocities_min, seed = PrepareData(
        file)
        test = 0
        model.fit(np.array(X), np.array(Y), 128, 1, verbose=1, callbacks=[model_save_callback])
        # model.train_function=None
        # model.test_function =None
        # model.metrics_tensors += model.output
        # model.metrics_name += ['predictions']

        x = seed
        x = np.expand_dims(x, axis=0)
        prediction = []
        for i in range(300):
            preds = model.predict(x)
            x = np.squeeze(x)
            x = np.concatenate((x, preds))
            x = x[1:]
            x = np.expand_dims(x, axis=0)
            preds = np.squeeze(preds)
            prediction.append(preds)
        for pred in prediction:
            # Undo the preprocessing
            pred[0] = int((pred[0] / 2) * (note_max - note_min) + (note_min + note_max) / 2)
            pred[1] = int((pred[1] / 2) * (velocities_max - velocities_min) + (velocities_min + velocities_max) / 2)
            if pred[0] < 24:
                pred[0] = 24
            elif pred[0] > 102:
                pred[0] = 102
            if pred[1] < 0:
                pred[1] = 0
            elif pred[1] > 127:
                pred[1] = 127
        mid = cp.deepcopy(MidiFile())
        track = MidiTrack()

        t = 0
        for note in prediction:
            # 147 means note_on
            note = np.asarray([147, note[0], note[1]])
            bytes = note.astype(int)
            msg = Message.from_bytes(bytes[0:3])
            t += 1
            msg.time = t
            track.append(msg)

        mid.tracks.append(track)
        mid.save('Generated_file'+str(test)+'.mid')
        test=test+1

x = seed
x = np.expand_dims(x, axis=0)

for i in range(300):
    preds = model.predict(x)
    x = np.squeeze(x)
    x = np.concatenate((x, preds))
    x = x[1:]
    x = np.expand_dims(x, axis=0)
    preds = np.squeeze(preds)
    prediction.append(preds)

for pred in prediction:
# Undo the preprocessing
    pred[0] = int((pred[0]/2)*(note_max-note_min) + (note_min+note_max)/2)
    pred[1] = int((pred[1]/2)*(velocities_max-velocities_min) + (velocities_min+velocities_max)/2)
    if pred[0] < 24:
        pred[0] = 24
    elif pred[0] > 102:
        pred[0] = 102
    if pred[1] < 0:
        pred[1] = 0
    elif pred[1] > 127:
        pred[1] = 127
mid = MidiFile()
track = MidiTrack()

t = 0
for note in prediction:
    # 147 means note_on
    note = np.asarray([147, note[0], note[1]])
    bytes = note.astype(int)
    msg = Message.from_bytes(bytes[0:3])
    t += 1
    msg.time = t
    track.append(msg)

mid.tracks.append(track)
mid.save('Generated_song_epoch =10.mid')





