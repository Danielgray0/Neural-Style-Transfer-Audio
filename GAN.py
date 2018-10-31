
import keras
from mido import MidiFile
import numpy as np
import os
import random
from keras import backend as k
file ='elise.mid'
file2='appass_1.mid'
chunk =5000

def PrepareData(file):
    mid=MidiFile(file)
    type(mid)
    times =[]
    velocities=[]
    notes=[]
    #print(mid.msg)
    mid=MidiFile(file)
    for msg in mid:
        if not msg.is_meta:
            if msg.channel == 0:
                if msg.type == 'note_on':
                    data = msg.bytes()
                    times.append(msg.time)
                    note = data[1]
                    velocity = data[2]
                    notes.append(note)
                    velocities.append(velocity)
                    #print(velocities,notes,times)
    combine=[]
    for i in range(len(velocities)):
        combine.append([notes[i],velocities[i],times[i]])
    note_min = np.min(notes)
    note_max = np.max(notes)
    velocities_min = np.min(velocities)
    velocities_max = np.max(velocities)
    for i in combine:
        i[0] = 2*(i[0]-((note_min+note_max)/2))/(note_max-note_min)
        i[1] = 2*(i[1]-((velocities_min+velocities_max)/2))/(velocities_max-velocities_min)
    Song = np.array(combine)
    return Song

def CompileMidi(file):
    Songs={}
    for f in os.listdir('./Midi_Data'):
        if f.endswith('.mid'):
            l = './Midi_Data/'+f
            Songs[l]=(PrepareData(l))

    return Songs


Songs=CompileMidi('./Midi_Data')
Songlist=[]
for i in Songs.keys():
    Songlist.append(i)
Goal = Songs[Songlist[random.randint(0,len(Songlist)-1)]]
print(Goal)
In = np.random.normal(0,.3,Goal.shape)
print(In.shape)


model = keras.Sequential()
keras.layers.SimpleRNN(In.shape[0],activation='tanh',use_bias=True,kernel_initializer='glorot_uniform',recurrent_initializer='orthogonal',bias_initializer='zeros')

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(In,Goal,100,1000,verbose=1)

noise = np.random.normal(0,.3,Goal.shape)
pred=model.predict(noise,200)


