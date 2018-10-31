import librosa as lb


def WavtoSpec(file):
    wav,sr= lb.core.load(file,sr=48000,mono=True)
    wav = lb.to_mono(wav)
    Spec=lb.feature.melspectrogram(wav,sr=48000)
    return Spec

def WavtoTimeSeries(file):
    wav, sr = lb.core.load(file, sr=48000, mono=True)
    return wav