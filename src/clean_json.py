import sys
import json
import os
from merge_rover import pre_processing, load_words, load_segments


def clean_json(json_file_path, output_file_path):
    """
    Clean a JSON file by removing some special character and
    made it in a "standard" format.

    This function clean a JSON file with only word dictionaries
    (word_segments in WhisperX trancription).

    Parameters:
        json_file_path   (str) : Path of the JSON file.
        output_file_path (str) : Path of the cleaned JSON file.
    
    Returns:
        None.
    """

    #loading
    data = load_words(json_file_path)

    #cleaning using preprocessing
    data_cleaned = pre_processing(data)

    #saving
    with open(output_file_path, "w") as f:
        json.dump(data_cleaned, f, indent=4, ensure_ascii=False)


def clean_json_v2(json_file_path, output_file_path):
    """
    Clean a JSON file by removing some special character and
    made it in a "standard" format.

    This function clean a JSON file with segments and word dictionaries
    (word_segments in WhisperX transcription).

    Parameters:
        json_file_path   (str) : Path of the JSON file.
        output_file_path (str) : Path of the cleaned JSON file.
    
    Returns:
        None.
    """
    
    # === Loading ===
    segments = load_segments(json_file_path)
    words    = load_words(json_file_path)

    # print(segments, words)
    # === Cleaning the words part ===
    words = pre_processing(words)

    # === Cleaning the segments part ===
    for segment in segments:
        segment["words"] = pre_processing(segment["words"])    

    # === Saving ===
    transcription_cleaned = {"segments":segments , "word_segments": words}
    
    with open( output_file_path, "w") as f:
        json.dump(transcription_cleaned, f, indent=4, ensure_ascii=False)
    

if __name__=="__main__":
    print("=== Starting... ===")
    # IMPORTANT: add modification depending of the folder you pass as arguments

    input  = sys.argv[1] #folder where to take all files. Ex: ../output
    output = sys.argv[2] #folder where to put all cleaned files. Ex: ../output_cleaned
    
    print("Input:", input)
    print("Output:",output)

    for file in os.listdir(input):
        
        if "transcription" not in file.split("_"): continue
        
        print(f"Transformation of {input}/{file} into {output}/{file[:-5]}.json")
        clean_json_v2(f"{input}/{file}", f"{output}/{file[:-5]}.json")
    
    print("=== Completed ===")

