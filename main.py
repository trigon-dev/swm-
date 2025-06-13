import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pydub import AudioSegment

for file in os.listdir('.'):
    if file.endswith('.mp3'):
        base = os.path.splitext(file)[0]

        # 1️⃣ Loading and converting to monophonics first
        sound = AudioSegment.from_file(file).set_channels(1).set_frame_rate(44100)
        samples = np.array(sound.get_array_of_samples(), dtype=np.int16)

        # 2️⃣ Analyze pitch (using librosa)
        y, sr = librosa.load(file, sr=44100, mono=True)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_vals = pitches.max(axis=0)
        dominant_pitch = np.median([p for p in pitch_vals if p > 0])

        print(f"{file} - dominant pitch ≈ {dominant_pitch:.2f} Hz")

        # 3️⃣ Plot waveform and spectrum
        time = np.linspace(0, len(samples) / sr, len(samples))
        freq = np.fft.fftfreq(len(samples), d=1/sr)
        spectrum = np.abs(np.fft.fft(samples))

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(time, samples)
        plt.title(f"Waveform of {file}")
        plt.xlabel("Time (sec)")

        plt.subplot(1, 2, 2)
        idx = freq > 0
        plt.plot(freq[idx], spectrum[idx])
        plt.title("Spectrum")
        plt.xlabel("frequency (Hz)")

        plt.tight_layout()
        img_file = base + '_analysis.png'
        plt.savefig(img_file)
        print(f"Analysis graph saved to {img_file}")
        plt.close()

        # 4️⃣ Transform into pure square wave
        square_samples = np.where(samples >= 0, np.iinfo(np.int16).max, np.iinfo(np.int16).min).astype(np.int16)

        square = AudioSegment(
            square_samples.tobytes(), 
            frame_rate=sr, 
            sample_width=2, 
            channels=1
        )

        output_file = base + '_square.wav'
        square.export(output_file, format='wav')
        print(f"Converted {file} -> {output_file}")
