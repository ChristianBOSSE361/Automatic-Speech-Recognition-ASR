import os
import json
from json2trs import generate_trs
from merge_rover import load_segments , load_words

def assign(files_path: str, target_folder_path: str):
    """
    Assign transcription JSON and TRS files to their corresponding subfolder.

    Designed for the *all_interviews_2024_2025* dataset, this function places
    each transcription file into the correct subfolder within the target directory.

    Parameters:
        files_path         (str) : Path to the directory containing all transcription files.
        target_folder_path (str) : Path to the directory containing the subfolders
                                   into which the transcription files will be moved.
    Returns:
        None
    """

    for transcription in os.listdir(files_path):

        for folder in os.listdir(target_folder_path):

            folder_name        = transcription.split("+")[0]
            transcription_name = transcription.split("+")[1]
            
            # Check if the transcription come from this folder
            if folder_name != folder: continue

            # Loading of JSON file
            dico = { "segments":[] , "word_segments":[] }
            dico["segments"] = load_segments( os.path.join(files_path , transcription) )
            dico["word_segments"] = load_words( os.path.join(files_path , transcription) )

            # Save the JSON file and the TRS file
            with open(os.path.join(target_folder_path, folder, transcription_name) , "w") as f:
                json.dump( dico, f, indent=4, ensure_ascii=False)
            
            generate_trs(os.path.join(target_folder_path, folder, transcription_name))

            # dispaly
            print(f"Save {os.path.splitext(transcription_name)[0]} ....")
    

if __name__== "__main__":
    print("=== Starting ...===")

    # Arguments
    files_path = "../merged_transcriptions_all_interviews_2024_2025_full"
    target_folder_path = "../cfpr_all_interviews_2024_2025_semianon/CFPR_interviews_only_2"

    # Runing
    assign(files_path, target_folder_path)

    print("=== Completed.===")
                


