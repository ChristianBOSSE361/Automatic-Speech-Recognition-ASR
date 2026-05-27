#!../../miniconda3/envs/wx/bin/python3
import sys
import os
from pydub import AudioSegment
import json

# Local fix to use ffmpeg on server
os.environ["PATH"] = "/home/getalp/bossec/cfpr/ffmpeg-7.0.2-amd64-static:" + os.environ["PATH"]
AudioSegment.converter = "/home/getalp/bossec/cfpr/ffmpeg-7.0.2-amd64-static/ffmpeg"
AudioSegment.ffprobe = "/home/getalp/bossec/cfpr/ffmpeg-7.0.2-amd64-static/ffprobe"
print("ffmpeg path:", AudioSegment.converter)
print("ffprobe path:", AudioSegment.ffprobe)


def split_audio(file_path, file_transcription, output_path):
    """
    Split an audio file into multiple segments based on timestamps from its transcription.

    The function reads a JSON transcription file containing timestamped segments
    and splits the original audio accordingly. Each resulting audio segment is saved
    in a folder named after the original audio file.

    Parameters:
        file_path (str)          : Path to the input audio file.
        file_transcription (str) : Path to the JSON transcription file. The file must contain timestamped segments of the audio.
        output_path (str)        : Directory where the audio segments will be saved.

    Returns:
        None
    """
   
    # === Loading of the audio and the transcription ===
    audio = AudioSegment.from_file(file_path)
    with open(file_transcription) as f:
        transcription = json.load(f)
    
    segments = transcription["segments"]

    # === Creation of the folder to regroup the segments ===
    os.makedirs(output_path, exist_ok=True)
    basename = os.path.splitext(os.path.basename(file_path))[0]
    
    try:
        os.mkdir(f"{output_path}/{basename}")
    except FileExistsError:
        print(f"*** File {output_path}/{basename} already exist ***")
        return
    
    # === Creation of the segments ===
    for item in segments:
        start_time, end_time = int(item["start"]*1000), int(item["end"]*1000)#to make it in millisecond
        segment = audio[start_time:end_time]

        # Generation of the output file name
        output = f"{output_path}/{basename}/{basename}_{start_time}_{end_time}.mp3"

        # Storage of the segment
        segment.export(output, format="mp3")

        print(f"Exported: {output} completed.")

    print(f"=== Segmentation Completed ===")


def split_outlier(file_path, gold_transcription):
    """
    Function made only to correct an outlier : ALG2019_0058
    Split the audio regarding the end of the gold trancription because the audio was longer than wanted.
    The parameters have the same goals has the previous function. 
    """
    
    # === Loadings ===
    audio = AudioSegment.from_file(file_path)

    with open(gold_transcription, "r") as f:
        transcription = json.load(f)
    
    #=== Take the end time ===
    segments = transcription["segments"]
    endTime = int(segments[-1]["end"]*1000) # to make it in millisecond (ms)

    # === Segmentation of the audio === 
    new_audio = audio[:endTime]

    # === Storage ===
    new_audio.export(file_path, format="mp3")

    print(f"*** Modification made ***")


if __name__=="__main__":
    print("=== Starting... ===")
    
    # === Part 1 : Segmentation using all speakers ===
    input        = "../cfpr_audio_all_interviews_2024_2025"
    transcription= "../output_all_interviews_2024_2025"
    output_path  = "../segmentation/audio_all_interviews_2024_2025"

    print("Input        :", input)
    print("Transcription:", transcription)

    for file in os.listdir(input):
        print(f"Segmentation of {file}")
        
        # searching of the right transcription
        for trs in os.listdir(transcription):
            
            # we remove the extension
            true_name = "".join(trs.split(".")[0])
            
            # we take only the transcription file with the name "transcription" in the file name
            if "transcription" not in true_name.split("_"): continue

            # we select the right transcription
            trs_list = true_name.split("_")
            
            if ("_".join(trs_list[:-2]) == file[:-4] ):
                split_audio(f"{input}/{file}" ,f"{transcription}/{trs}", output_path)
                break

    # === Part 2 : Segmentation using only one speaker(the "main" one) ===
    # input = "../cfpr_audio"
    # transcription= "../CFPR_classique_one_speaker_json"
    # output_path  = "../segmentation/audio_one_speaker"

    # print("Input        :", input)
    # print("Transcription:", transcription)

    # for file in os.listdir(input):
    #     print(f"Segmentation of {file}")
    #     # searching of the right transcription
    #     for trs in os.listdir(transcription):
    #         trs_list = trs.split("_")
    #         if "_".join(trs_list[:2]) == file[:-4]:
    #             split_audio(f"{input}/{file}" ,f"{transcription}/{trs}", output_path)
    #             break

    print("=== Completed. ===")