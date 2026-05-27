import json
import sys
from merge_rover import load_words

def change_to_text(json_file_path, output_path):
    """
    Transform a JSON file into a text, a dialog between the speaker for a better visualization.
    
    Parameters:
        json_file_path (str) : Path of the transcription file. 
        output_path    (str) : Path of the output file to save the result.
    
    Returns:
        None.
    """
    
    # Loading of the json file
    data =load_words(json_file_path)
    
    dialog= ""
    speaker = ""

    #Loop to have all words
    for word in data:
        if speaker!=word["speaker"]:
            dialog+=f"\n{word["speaker"]}:{word["word"]}"
            speaker=word["speaker"]  
        else:
            dialog+=f" {word["word"]}" 

    # Writting of in the text file
    with open(output_path, "w") as f:
        f.write(dialog)


if __name__=="__main__":
    print("=== Starting ...===")
    
    json_file_path   = sys.argv[1]
    output_file_path = sys.argv[2]
    
    change_to_text(json_file_path, output_file_path)
    print("=== Completed. ===")
