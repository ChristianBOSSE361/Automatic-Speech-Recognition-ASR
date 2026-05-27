#!../../miniconda3/envs/wx/bin/python3
"""
The goal is to implement the ROVER (Recognizer Output Voting Error Reduction) to combine or merge the different transcriptions given by the two models.

Approach:
    - The Fist step : is to create a alignment module where each word of a trancsription will be faced to a word in the other transcription at the same place.
                    The main transcription is called the reference and the one, the one that as to be aligned is the hypotesis.

    - The Second step : is the voting module that aims to combine the different output regarding a score ( called weight in the code ).
"""

import re
import json
from collections import defaultdict
import os
import jiwer
from involve_whisperx import involve_whisperx_trs

# === GLOBAL VARIABLES ===
PUNCT_RE = re.compile(r"[.,;:?()\"]") #character we want to remove 
#took from Hugging face description of the model FrWhisper
FILLERS = {"ah", "bah", "beh", "ben", "chh", "eh", "euh",
        "ha", "hé", "hein", "hop", "hum", "m-hm", "mmh",
        "mm", "oh", "ouf", "pff", "youh"}


# ===============================================
# ============= HELPER FUNCTIONS ================
# ===============================================

def normalization(word):
    """
    Clean the word by removing some particular characters to have an standard form.

    Parameters:
        word(dict): Dictionary that contains the word in the format { "word":... ,"start":... , "end":..., "score":..., "speaker":... }.
    
    Returns:
        None
    """
    if (word is None) or ("word" not in word.keys()) or (word["word"] is None): return []

    w = word["word"] # we take the currrent word
    w = w.strip() #remove edge space
    w = w.lower() # ervything to lower case
    w = w.replace("-", " ") #help to transform "dis-moi" and "dis moi" into the same ["dis", "moi"]
    w = w.replace("'", " ") #help to transform "c'est" and "c' est" into the same ["c", "est"]
    w = PUNCT_RE.sub("", w) #remove special character from the edge of the word
    w = w.split()

    res = []
    for i in range(len(w)):
        res.append({"word"    : w[i],
                    "start"   : word["start"] if "start" in word.keys() else None,
                    "end"     : word["end"] if "end" in word.keys() else None,
                    "score"   : word["score"] if "score" in word.keys() else None,
                    "speaker" : word["speaker"] if "speaker" in word.keys() else None })

    return res

def pre_processing(word_list: list):
    """
    Preprocess a list of words (or word dictionaries) by applying normalization to each element.

    Parameters:
        word_list (list): List of words to clean.

    Returns:
        list: A list of normalized words.
    """
    new_word_list = []
    for word in word_list:
        new_word_list.extend(normalization(word))
    return new_word_list


def seq_words(words_liste):
    """
    Select only words from a list of word dictionaries that contain "word", "start","end", "speaker" and "score" as keys. 

    Parameters:
        words_list (list): List of word dictionaries.

    Returns:
        list: Filtered list of words.
    """

    return [ word["word"] for word in words_liste ]

def aggregate_times(time_liste):
    """
    Aggregate the times in the list of times. We take the mean of the sarting values and the mean of the ending values.

    Parameters:
        time_list (list) : List of times. The times are also tuple of 2 values (start and end).

    Returns:
        tuple: The aggregatation of the starting values and the ending values.
    """
    # for the moment I choose to take the mean between the different time, maybe I have just
    # to take the time from whisperx
    start = [time[0] for time in time_liste ]
    end = [time[1] for time in time_liste ]

    return (sum(start)/len(start) , sum(end)/len(end))

def avg_score(scores_liste):
    return sum(scores_liste)/len(scores_liste)


# ===============================================
# ============= STEP1: ALIGNMENT ================
# ===============================================

def align_hyp_to_ref(hyp_seg_list, ref_seg_list, all_speaker=True, threshold=0.8):
    """
    Align words from the hypothesis transcription with those from the reference transcription.

    This function optionally filters words depending on whether all speakers should be included
    or only the interviewees.

    A threshold is used to determine whether a hypothesis word is good enough to replace
    a reference word during alignment.

    Parameters:
        hyp_seg_list (list): List of word dictionaries from the hypothesis transcription.
        ref_seg_list (list): List of word dictionaries from the reference transcription.
        all_speaker  (bool): Whether to include all speakers.
            - True: include all speakers (default)
            - False: include only the interviewees
        threshold   (float): Minimum score required for a word to be considered in substitution.

    Returns:
        list: Alignment as a list of tuples (ref_index, hyp_index),
              where each tuple maps a reference word to a hypothesis word.
    """
    
    #=== Initialization ===
    alignment = []

    # Taking the word sequences from each output
    hyp_seq = seq_words(hyp_seg_list)
    ref_seq = seq_words(ref_seg_list)

    # === Computation of the alignment using Jiwer ===

    # computation of the word alignment
    output = jiwer.process_words(" ".join(ref_seq), " ".join(hyp_seq))

    # constrution of the list of indexes
    results = output.alignments[0]

    # === Main Loop ===
    # we add in the alignment list some tuples of indexes. We associate in the tuple, a word from the reference transcription
    # to his corresponding word (or list of words) from the hypothesis transcriptions. 
    for value in results:
        tag, i1, i2, j1, j2 = value.type, value.ref_start_idx ,\
            value.ref_end_idx, value.hyp_start_idx, value.hyp_end_idx
        
        if tag=="equal":
            for k in range(i2 -i1):
                alignment.append( ([i1+k], [j1+k]) )

        elif tag=="delete":
            for k in range(i2-i1):
                alignment.append( ([i1+k],[]) )
        
        elif tag=="substitute":
            len_ref = i2 - i1
            len_hyp = j2 - j1
            if len_ref==len_hyp:
                for k in range(i2-i1):
                    alignment.append( ([i1+k], [j1+k]) )
            else:
                for k in range(i1,i2):

                    current_w = ref_seg_list[k]
                    candidates = []
                    start = current_w["start"] - 0.1
                    end   = current_w["end"] + 0.1
                    
                    for l in range(j1,j2):
                        other_w = hyp_seg_list[l]
                        # if one of the tops of the intervals is in the current interval, it might be is candidates
                        if start <= other_w["start"] <= end or start <= other_w["end"] <= end:
                            if other_w["score"] >= threshold: candidates.append(l)
                    
                    alignment.append( ([k],candidates) )

        elif tag=="insert":
            alignment.append( ([], list(range(j1,j2))) )
        
        else:
            print("!!!! Error in alignment !!!!")
            break
    
    return alignment


# === Loads functions ====
def load_words(whisperx_path:str):
    with open(whisperx_path, "r") as f:
        transcription = json.load(f)
    if type(transcription)==type([]): return transcription 
    return transcription["word_segments"]


def load_segments(model_path:str):
    with open(model_path, "r") as f:
        transcription = json.load(f)
    return transcription["segments"]


# ===============================================
# === STEP2: CONFUSION NETWORK AND SCORING ======
# ===============================================

def build_cn(ref_words, hyp_words, systems_weights=[1 , 0.7] ):
    """
    Build a Confusion Network.
    
    Precisely it consist into creating a list of dictionaries where for each word in the reference transcription we have some candidates
    at the same position. Then we will use another function to choose which one is the best candidate.

    Each dictionary has the following format :
    
        {  
            "candidates": ["wordA": {"weights":float, "scores":list , "times":list, "speaker":str ,"origin":str},
                            "wordB":{"weights":float, "scores":list , "times":list, "speaker":str ,"origin":str}],
            "is_insertion": bool
        }
    
    Parameters:
        ref_words (list)       : List of word from the reference transcription.
        hyp_words (list)       : List of word from the hypothesis transcription.
        systems_weights (list) : List of the weights of the 2 transcriptions.
                            The default value is [1, 0.7] so 1 for the ref and 0.7 for the hyp transcription.
    
    Returns:
        list: List of dictionaries.
    """
    ref_weight = systems_weights[0]
    hyp_weight = systems_weights[1]
    slots = []

    # Initailisation of the slots by the reference words
    for word in ref_words:

        slot = defaultdict(lambda : {"weights":0.0, "scores":[] , "times":[], "speaker":None ,"origin":"ref"})
        # print("speaker" in word.keys())
        if word:
            slot[word["word"]]["weights"] = ref_weight * word["score"] 
            slot[word["word"]]["scores"].append(word["score"])
            slot[word["word"]]["times"].append((word["start"] , word["end"]))
            slot[word["word"]]["speaker"] = word["speaker"]

        slots.append({"candidates":slot , "is_insertion":False})
    
    # Alignment with respcet to the other model
    alignment = align_hyp_to_ref(hyp_words, ref_words)
    new_slots = []

    for (r_ind, h_ind) in alignment:
        slot = defaultdict(lambda : {"weights":0.0, "scores":[] , "times":[], "speaker":None , "origin":None})
        is_insertion = (len(r_ind)==0) #according the structure of SequenceMacther

        #copy candidates comming from ref words
        #Check it is an insertion or not: if it is not an insertion, we know that r_ind in not empty
        # so we have values/words to add in the slot. And if this is an insertion , we oncly copy the candidate coming
        # from the hyp words
        
        # if is_insertion: print("Insertion de:" ,[hyp_words[i]["word"] for i in h_ind])
        # if len(r_ind)!=0 and len(r_ind)<= len(h_ind): print("Replacement of:",[ref_words[id]["word"] for id in r_ind]," by ", [hyp_words[id]["word"] for id in h_ind])

        if not is_insertion: 
            for idr in r_ind:
                for word, info in slots[idr]["candidates"].items():
                    slot[word]["weights"]+=info["weights"] 
                    slot[word]["scores"].extend(info["scores"])
                    slot[word]["times"].extend(info["times"])
                    slot[word]["speaker"] = info["speaker"]
                    slot[word]["origin"]  = "ref"

        #copy hyp candidates
        ref_word_in_slot = [val for val,_ in slot.items()]
        for idh in h_ind:
            word_info = hyp_words[idh]
            if word_info["word"] and word_info["word"] not in ref_word_in_slot:
                slot[word_info["word"]]["weights"] += word_info["score"] * hyp_weight 
                slot[word_info["word"]]["scores"].append( word_info["score"] )
                slot[word_info["word"]]["times"].append( (word_info["start"],word_info["end"]) )
                slot[word_info["word"]]["origin"] = "ref" if slot[word_info["word"]]["origin"] else "hyp"
                # slot[word_info["word"]]["speaker"] = word_info["speaker"]
        
        new_slots.append({"candidates":slot , "is_insertion":is_insertion})

    return new_slots


def choose_word_in_slot(slot, theta_filler=0.25):
    """
    Select the best word from a list of candidates at a given alignment position.

    A threshold is used to determine whether a filler word (hesitation)
    should be included or not.

    Parameters:
        slot (list): List of dictionaries with the following format:
            {
                "candidates": list,
                "is_insertion": bool
            }
        theta_filler (float): Minimum score required for a filler word
            to be considered. The default value is 0.25.

    Returns:
        dict: Dictionary representing the selected word.
    """

    if not slot:
        return None

    total_weight = sum(info["weights"] for info in slot["candidates"].values())
    if total_weight == 0:
        return None
    
    # If insertion
    if slot["is_insertion"]:
        #we look only at the fillers (hesitation words)
        best_word, best_info = max(slot["candidates"].items(), key=lambda kv: kv[1]["weights"] if kv[0] in FILLERS else 0.0)
        relative_weight = best_info["weights"] / total_weight

        # searching of a valid speaker
        speaker = best_info["speaker"]
        if speaker == None:
            for info in slot["candidates"].values():
                if info["speaker"]!=None:
                    speaker = info["speaker"]
                    break

        if relative_weight >= theta_filler:
            start, end = aggregate_times(best_info["times"])
            return { "word": best_word, "start": start, "end": end, "score": avg_score(best_info["scores"]),
                    "speaker": speaker,
                    "tag": "filler" if best_word in FILLERS else "insertion"}
        return None
    
    # If there is no insertion , we want to keep the word coming from the ref transcription
    best_word, best_info = max(slot["candidates"].items(), key=lambda kv: kv[1]["weights"] if kv[1]["origin"]=="ref" else 0.0)
    start, end = aggregate_times(best_info["times"])
    
    # searching of a valid speaker
    speaker = best_info["speaker"]
    if speaker == None:
        for info in slot["candidates"].values():
            if info["speaker"]!=None:
                speaker = info["speaker"]
                break

    return {"word": best_word,"start": start,"end": end,"score": avg_score(best_info["scores"]),
            "speaker": speaker,
            "tag": None}


# ===============================================
# ============== MAIN FUNCTION ==================
# ===============================================

def merge(whisperx_trs_path:str, other_trs_path:str, output_trs_path:str , systems_weights:list = [1, 0.7]):
    """
    TO DO
    """

    #Loads
    print("Loads...")
    ref = load_words(whisperx_trs_path)
    hyp = load_words(other_trs_path)

    #Prepoccessing to remove some punctation and have the same structure
    print("Prepocessing...")
    ref_clean = pre_processing(ref)
    hyp_clean = pre_processing(hyp)
    
    # Compute the slots using the confusion netword and alignment
    print("Computing slots..")
    slots = build_cn(ref_clean, hyp_clean, systems_weights)

    #Choosing the best slot
    print("Choosing the best slots...")
    result = []
    for slot in slots:
        to_add = choose_word_in_slot(slot)
        if to_add!=None: result.append( to_add )


    print("Saving...\n")  
    # Storing this new transcription in a file
    with open(output_trs_path, "w" ) as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    

if __name__ == "__main__":
    print("=== STARTING ...===")
    
    #Arguments
    whisperx_trs_path       = "../output_all_interviews_2024_2025"
    whisperx_trs_path_clean = "../output_all_interviews_2024_2025" # NOTE: you have to create this folder using clean_json.py
    second_model_trs_path   = "../segmentation/transcriptions_all_interviews_2024_2025"
    output_folder_path      = "../merged_transcriptions_all_interviews_2024_2025"
    
    for ref_file in os.listdir(whisperx_trs_path):
        
        if "transcription" not in ref_file.split("_"): continue

        for hyp_file in os.listdir(second_model_trs_path):
            
            # we remove the extension
            true_name = "".join(ref_file.split(".")[0])
            
            # we take only the transcription file with the name "transcription" in the file name
            if "transcription" not in true_name.split("_"): continue

            # we select the right transcription
            ref_file_list = true_name.split("_")
            
            if ("_".join(ref_file_list[:-2]) != hyp_file[:-5] ): continue

            # Check if the merge is not already done
            if os.path.exists(f"{output_folder_path}/{hyp_file}"):
                print(f"**** File {output_folder_path}/{hyp_file} already exist***")
                break
        
            os.makedirs(output_folder_path, exist_ok=True)

            # We do the merge
            merge(whisperx_trs_path = f"{whisperx_trs_path}/{ref_file}",
                  other_trs_path    = f"{second_model_trs_path}/{hyp_file}",
                  output_trs_path   = f"{output_folder_path}/{hyp_file}")

            # We involve the whisperx transcription
            involve_whisperx_trs(whisperx_trs_path  = f"{whisperx_trs_path_clean}/{ref_file}",
                                 merged_trs_path    = f"{output_folder_path}/{hyp_file}",
                                 output_folder_path = output_folder_path+"_full")
            break
        
    
    print("=== COMPLETED ===")