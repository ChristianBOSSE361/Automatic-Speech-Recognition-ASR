import json
import os


# === Loads functions ====
# (same as in merge_rover.py )
def load_words(whisperx_path:str):
    with open(whisperx_path, "r") as f:
        transcription = json.load(f)
    if type(transcription)==type([]): return transcription 
    return transcription["word_segments"]

def load_segments(model_path:str):
    with open(model_path, "r") as f:
        transcription = json.load(f)
    return transcription["segments"]

# === Main function ===
def involve_whisperx_trs(whisperx_trs_path , merged_trs_path, output_folder_path):
    """
    Involve the WhisperX transcription by injected the words added in the merged transcription.

    The WhisperX transcription needed here must be a cleaned version of the transcription.

    Parameters:
        whisperx_trs_path  (str) : Path of the WhisperX transcription.
        merged_trs_path    (str) : Path of the merged transcription.
        output_folder_path (str) : Path of the folder where to store the new transcription.
    
    Returns:
        None.
    """

    # === Loading ===
    whisperx_segments = load_segments(whisperx_trs_path)
    merged_words      = load_words(merged_trs_path)

    # === Modification of the segments ===
    i , id_start , id_end  = 0 , 0 , 0
    while i < len(whisperx_segments):
        segment = whisperx_segments[i]

        # We look for the subset of words in the merged transcription that corresponds
        # to the current segment, in order to replace the segment's word list.
        # The idea is to iterate through the merged list of words until we find
        # the first word whose start time matches the segment's start time.
        # Then, we continue iterating until we reach a word whose end time
        # exceeds the segment's end time.
        # This way, we identify all the words that belong to the segment.
        
        id_end = id_start+1

        while id_end < len(merged_words):
            if merged_words[id_end]["tag"]!="filler" and merged_words[id_end]["tag"]!="insertion": # if the word is not a filler (an hesitation word added)
                if merged_words[id_end]["end"]<=segment["end"]: id_end+=1
                else : break
            else : id_end+=1
              
        
        segment["words"] = merged_words[id_start:id_end] #change of the words part
        segment["text"]  = " ".join([word["word"] for word in segment["words"]]) #change of the text part
        
        # incrementation
        i+=1
        id_start = id_end

    # === Saving ===
    transcrption = {"segments": whisperx_segments, "word_segments":merged_words}

    output = os.path.basename(merged_trs_path)

    os.makedirs(output_folder_path, exist_ok=True)

    with open(f"{output_folder_path}/{output}", "w") as f:
        json.dump(transcrption, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    print("=== STARTING ... ===")

    whisperx_trs_path = "../output_cleaned"
    merged_trs_path   = "../merged_transcriptions"
    output_folder_path= "../merged_transcriptions_full"

    print("Whisperx files folder:", whisperx_trs_path)
    print("Merged files folder  :", merged_trs_path)

    for ref_file in os.listdir(whisperx_trs_path):
        
        for hyp_file in os.listdir(merged_trs_path):

            if hyp_file[:-5]!=ref_file[:12]: continue
            print(f">>> Full transcription of {hyp_file[:-5]} ...")
            
            involve_whisperx_trs(f"{whisperx_trs_path}/{ref_file}",f"{merged_trs_path}/{hyp_file}", output_folder_path)
            break
    
    print("=== COMPLETED ===")
