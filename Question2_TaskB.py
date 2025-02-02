import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Function to compute STFT and generate spectrogram
def compute_spectrogram(file_path, window_type='hann', n_fft=1024, hop_length=512):
    # Load the audio file
    sr, audio = wavfile.read(file_path)
    
    # Check if audio is stereo and convert to mono if necessary
    if audio.ndim == 2:  # Check if audio has 2 dimensions (stereo)
        audio = np.mean(audio, axis=1)  # Convert to mono by averaging channels
    
    audio = audio / np.max(np.abs(audio))  # Normalize

    # Select the window type
    if window_type == 'hann':
        window = np.hanning(n_fft)
    elif window_type == 'hamming':
        window = np.hamming(n_fft)
    elif window_type == 'rectangular':
        window = np.ones(n_fft)
    else:
        raise ValueError("Unsupported window type")
    
    # Short-Time Fourier Transform
    num_segments = 1 + (len(audio) - n_fft) // hop_length
    spectrogram = np.zeros((n_fft // 2 + 1, num_segments), dtype=np.complex64)
    
    for i in range(num_segments):
        start = i * hop_length
        end = start + n_fft
        segment = audio[start:end] * window
        fft_result = np.fft.rfft(segment)
        spectrogram[:, i] = fft_result

    # Convert to magnitude and log scale
    magnitude = np.abs(spectrogram) ** 2
    return 10 * np.log10(magnitude + 1e-10)  # Avoid log(0)

# Function to plot spectrogram
def plot_spectrogram(spectrogram, sr, hop_length, title, genre):
    plt.figure(figsize=(10, 4))
    time_axis = np.linspace(0, len(spectrogram[0]) * hop_length / sr, len(spectrogram[0]))
    freq_axis = np.linspace(0, sr / 2, len(spectrogram))
    plt.pcolormesh(time_axis, freq_axis, spectrogram, shading='gouraud', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'{genre}: {title}')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()

# Main Execution
# 10 seconds of each song is taken
songs = {
    'Classical': './content/chanakya-by-rishabh-sharma.wav',
    'Jazz': './content/podcast-smooth-jazz-instrumental-music.wav',
    'Rock': './content/rock-again-guitar-and-drum-instrumental.wav',
    'Disco-Pop': './content/bts-dynamite-instrumental.wav'
}

for genre, file_path in songs.items():
    print(f"Processing {genre}...")
    spectrogram = compute_spectrogram(file_path, window_type='hann')
    plot_spectrogram(spectrogram, sr=44100, hop_length=512, title="Hann Window", genre=genre)