""" Take in example.mid and rewrite for use with scipy
    sparse matrix. Example of how to parse MIDI with Music21.
    Execution: python parse.py [file to print to] """

from music21 import *
example = converter.parse('Midi_Data unused/chp_op18_new.mid')
data = example.parts[0]
#data.show()
#print(data.show())
print ("%s,%s,%s,%s, %s" % ("Note/Rest", "Octave", "Len","Offset", "Velocity"))
notes = [i for i in data.notesAndRests]
Song =[]
for n in notes:
    if isinstance(n, note.Note):
        # alternative n.pitch.nameWithOctave
        Song.append([n.pitch.name, n.pitch.octave, n.quarterLength, n.offset,n.volume.getRealized()])
    elif isinstance(n, note.Rest):
        Song.append(["Rest", None, n.quarterLength, n.offset])
print(Song)