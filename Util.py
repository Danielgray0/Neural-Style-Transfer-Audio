
import numpy as np
from mido import MidiFile,MidiTrack,Message

mid = MidiFile('example.mid')
notes = []
velocities = []
def PrepareData():
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
    n_prev = 30
    for i in range(len(combine) - n_prev):
        x = combine[i:i + n_prev]
        y = combine[i + n_prev]
        X.append(x)
        Y.append(y)
    # save a seed to do prediction later
    seed = combine[0:n_prev]
    return X,Y


# path = './Midi_Data'
# print('offset,  note,  octave,  velocity,  channel')
# pitchnames=['Rest','A','A#','B','C','C#','D','D#','E','F','F#','G','G#']
# note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
#
#
#
#
# def MiditoArray(midi):  # Takes one midi file and returns numpy array
#     return mom.midi.process_notes(midi)
#
# def parse_midi(path):
#     midi = None
#     try:
#         midi = pretty_midi.PrettyMIDI(path)
#         midi.remove_invalid_notes()
#     except Exception as e:
#         raise Exception(("%s\nerror readying midi file %s" % (e, path)))
#     return midi
#
# prettymidi=parse_midi('./Midi_Data/beethoven_opus10_1.mid')
# #print(prettymidi.get_beats()[9])
# #print(prettymidi.get_chroma()[9])
# for i in prettymidi.get_piano_roll(fs=1000):
#     print(i)
#
#
#
#
# # def MiditoArray2(midi):
# #     data = converter.parse(midi).parts[0]
# #     print(type(data))
# #     notes_to_parse = data.flat.notes
# #     notes=[]
# #     count=0
# #     for i in data.notesAndRests:
# #
# #         notes.append(i)
# #         count=count+1
# #     Song = np.array((count,5), dtype=object)
# #     iter=0
# #     for n in notes:
# #         if isinstance(n, note.Note):
# #             for i in range(0,4):
# #                 Song[iter][i]=n.pitch.name, n.pitch.octave, n.quarterLength, n.offset,n.volume.getRealized()
# #         elif isinstance(n, note.Rest):
# #             Song[iter]=[0, None, n.quarterLength, n.offset,0]
# #         iter=iter+1
# #     return Song
# #
# # MiditoArray2('example.mid')
#
#
#
#
# def DirtoArray(path):  # Places a Directory of midi files in to an array of numpy arrays
#     Songs = []
#     SongNames = []
#     for f in os.listdir(path):
#         if f.endswith('.mid'):
#             file = './Midi_Data/' + f
#             SongNames.append(file)
#             print(file)
#             print(MiditoArray(file))
#             Songs.append(MiditoArray(file))
#     return Songs, SongNames
#
#
# #Train_Set, Train_Set_Names = DirtoArray(path)
#
#
# def PrintSet(Train_Set):
#     "Prints first three songs in Training set "
#     for i in range(0, 3):
#         for v in Train_Set[i]:
#             print(v)
#
#
# #PrintSet(Train_Set)
# #print(Train_Set_Names)
