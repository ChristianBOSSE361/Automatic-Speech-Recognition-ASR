""" Find the mean and maximum duration of a set of audio files. """

import librosa
import os

#=== Initialization ===
input = "../cfpr_audio_2022a_2023"
maxi = 0
val = 0
 
# === Loading of the files ===
audio_files = os.listdir(input)

# === Main Loop ===
for audio_path in audio_files:
    # for audio_path in os.listdir(f"{input}/{file}"):
        aud = f"{input}/{audio_path}"
        
        if "json" in aud.split("."): continue
        
        audio, sr = librosa.load(aud, sr=16000)
        
        duration = librosa.get_duration(y=audio, sr=sr)
        print(duration/60 ,"min")

        val+=duration
        if duration > maxi: maxi = duration

# === Display ===
print("The maximum is:", maxi/60, "min")
print("The mean is   :", val/(len(audio_files)*60), "min")