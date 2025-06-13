import os
from pydub import AudioSegment
import numpy as np

for file in os.listdir('.'):
    if file.endswith('.mp3'):
        input_file = file
        output_file = os.path.splitext(file)[0] + '_square.wav'

        sound = AudioSegment.from_file(input_file)
        mono = sound.set_channels(1).set_frame_rate(44100)

        samples = np.array(mono.get_array_of_samples(), dtype=np.int16)

        square_samples = np.where(samples >= 0, np.iinfo(np.int16).max, np.iinfo(np.int16).min).astype(np.int16)

        square = AudioSegment(
            square_samples.tobytes(), 
            frame_rate=mono.frame_rate, 
            sample_width=2, 
            channels=1
        )

        square.export(output_file, format='wav')
        print(f"Converted {input_file} -> {output_file}")
