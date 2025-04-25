import librosa
import numpy as np
from pydub import AudioSegment
from pydub.utils import which

AudioSegment.converter = which("ffmpeg")
print("FFmpeg path being used:", AudioSegment.converter)

def extract_features(file_path):
    # Convert mp3 to wav if needed
    if file_path.endswith('.mp3'):
        sound = AudioSegment.from_mp3(file_path)
        file_path = file_path.replace('.mp3', '.wav')
        sound.export(file_path, format='wav')

    max_pad_len = 174

    # Load audio file WITHOUT specifying duration or fixed sr
    y, sr = librosa.load(file_path, res_type='kaiser_fast')  # match training logic

    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # Pad or trim MFCC to shape (40, 174)
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]

    # Reshape to (1, 40, 174, 1)
    mfcc = mfcc.reshape(1, 40, 174, 1)
    return mfcc
