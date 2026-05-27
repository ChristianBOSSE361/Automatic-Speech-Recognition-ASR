#!../../miniconda3/envs/wx/bin/python3
import os
import glob
from pydub import AudioSegment


# === TO FIX THE ENVIRONMENT ===
os.environ["PATH"] = "/home/getalp/bossec/cfpr/ffmpeg-7.0.2-amd64-static:" + os.environ["PATH"]
AudioSegment.converter = "/home/getalp/bossec/cfpr/ffmpeg-7.0.2-amd64-static/ffmpeg"
AudioSegment.ffprobe = "/home/getalp/bossec/cfpr/ffmpeg-7.0.2-amd64-static/ffprobe"

# === EXTENSIONS ALLOWED ===
audio_extensions = ['mp3', 'wav', 'ogg', 'flac', 'm4a', 'mp4']

# === VERSION 1 ===
def clean_audios_file(files_path, output_folder):
    """
    Clean and standardize a set of audio files.

    This function converts audio files with different extensions into a common
    .wav format. When multiple audio files correspond to the same recording,
    it selects the one with the shortest duration. (Used to clean audio files from 2022a and 2023)

    Parameters:
        files_path    (str): Path to the folder containing subfolders,
            each corresponding to an audio recording.
        output_folder (str): Path to the folder where cleaned audio files will be saved.

    Returns:
        None 
    """

    # === Main Loop ===
    for folder in os.listdir(files_path):

        # construction of the new audio path
        output_path = os.path.join(output_folder, folder) + ".wav"

        # check if the file already exist
        if os.path.exists(output_path):
            print(f"*** File {output_path} already exist ***")
            continue
        
        # initialization
        audios = []
        
        # selection of the different audios in the folder
        for file in os.listdir(f"{files_path}/{folder}"):
            
            # check if this an audio file
            if file.split(".")[-1] not in audio_extensions: continue
            audios.append(file)

        # choosing the one with the lower duration
        if len(audios)<=0:
            print("Error: No audio in ", folder)
            return
        elif len(audios)>1:
            min, id = AudioSegment.from_file(os.path.join(files_path, folder, audios[0])).duration_seconds , 0
            for i in range(1,len(audios)):
                val = AudioSegment.from_file(os.path.join(files_path, folder, audios[0])).duration_seconds
                if min > val:
                    min = val
                    id  = i
        else:
            id = 0

        file = audios[id]

        # path of the audio
        audio_path  = os.path.join(files_path, folder, file)

        # Loading of the file and we convert it into a wav file
        audio = AudioSegment.from_file(audio_path)
        audio.export(output_path, format="wav")

        print(f"Saved : {output_path}.")

# === VERSION 2 ===
def clean_audio_files_v2(files_path, output_folder):
    """
    Clean and standardize a set of audio files.

    This function converts audio files with different extensions into a common
    .wav format. When multiple audio files correspond to the same recording,
    it selects the one with the shortest duration (used to clean audio files
    from the "all_interviews_2024_2025" dataset).

    This version differs from the previous one because the file structure
    being processed is also different.

    Parameters:
        files_path (str): Path to the folder containing subfolders,
            each corresponding to a single audio recording.
        output_folder (str): Path to the folder where the cleaned
            audio files will be saved.

    Returns:
        None
    """
    # === Creation of the output folder ===
    os.makedirs(output_folder, exist_ok=True)

    # === Main Loop ===
    for folder in os.listdir(files_path):
        
        # We take all the audio files in the folder
        files = os.listdir(f"{files_path}/{folder}")
        files = [ files[i] for i in range(len(files)) if files[i].split(".")[-1] in audio_extensions ]

        # We convert in a standard format all audios in the folder
        for audio in files:

            # path of the audio
            audio_path        = os.path.join(files_path, folder, audio)

            # path of the ouput composed by : folder + audio file name + .wav
            audio_output_path = os.path.join(output_folder, folder + "+" + "".join(audio.split(".")[:-1]) + ".wav")

            # Skip if audio file already create
            if os.path.exists(audio_output_path):
                print(f"*** File {audio} already made. ***")
                continue

            # Loading of the file and we convert it into a wav file
            try:
                aud = AudioSegment.from_file(audio_path)
                aud.export(audio_output_path, format="wav")
            except Exception as e :
                print(f"**** Error : {e}\n Not able to save {audio_output_path}. ****")
                continue
            
            print(f"Saved : {audio_output_path}.")
    
        

if __name__ == "__main__":

    print("=== Starting ... ===")
    
    files_path    = "../cfpr_all_interviews_2024_2025_semianon/CFPR_interviews_only_2/"
    output_folder = "../cfpr_audio_all_interviews_2024_2025"

    print("Input  :", files_path)
    print("Output :", output_folder)

    # clean_audios_file(files_path, output_folder)
    clean_audio_files_v2(files_path, output_folder)
    print("=== Completed. ===")
