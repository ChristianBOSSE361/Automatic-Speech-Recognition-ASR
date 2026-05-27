import sys
import xml.etree.ElementTree as ET
import json
from merge_rover import normalization
import os
from lxml import etree
import re

# === Global variable ===
ONE_CHAR_ALLOWED = ["à","a","y","ô"]

# === Add to the dico the words ===
def add_to_dico(text:str, time:float, speaker:str, dico:list):
    """
    Add to the list named "dico" the word dictionaries creating from the text.
    
    Parameters:
        text    (str)  : The text in which we should take the words.
        time    (float): Starting time to associate to every word.
        speaker (str)  : Speaker to associate to every word.
        dico    (list) : List where to add the word dictionaries.
    
    Returns:
        None.
    """
    text_list = text.split()
    for i in range(len(text_list)):
        word = { "word": text_list[i] , "start":time , "end":0.0 , "speaker":speaker}
        dico.extend(normalization(word))
    

# === Add to dico the segments and word_segments part ====
def add_to_dico_full(text:str, speaker:str,seg_start:float, seg_end:float, dico:dict):
    """
    Add to the dictionary named "dico" the segment.
    
    Parameters:
        text      (str)   : The text of the segment.
        seg_start (float) : Starting time of the segment.
        seg_end   (float) : Ending time of the segment. 
        speaker   (str)   : Speaker of the segment.
        dico      (list)  : Dictionary where to add the segment.
    
    Returns:
        None.
        
    """

    # Initialisation of the segment
    segment = { "start":seg_start, "end":seg_end, "text":text, "words":[], "speaker":speaker}

    text_list = text.split()
    for i in range(len(text_list)):
        word = { "word": text_list[i] , "start": seg_start , "end":0.0 , "speaker":speaker}
        segment["words"].extend(normalization(word))

    # adding to the dictionnary
    dico["segments"].append(segment)
    dico["word_segments"].extend(segment["words"])


# === Change the .trs file into .json file ===
def change_into_json(trs_file_path, output_path, multiple_speaker = True):
    """
    Change a TRS file in a JSON file.
    
    This function could whether use all speakers in the TRS file
    or only consider the interviwees.
    
    Parameters:
        trs_file_path    (str) : Path of the TRS file.
        ouput_path       (str) : Path where to add the JSON file.
        multiple_speaker (bool) : Wheter take all speakers or not.
                        - True  : Take all speakers. (default)
                        - False : Take only the interviewees.
    
    Returns:
        None.
    """

    # === Initialization ===  
    # We encode en byte and after in utf-8 to be able to remove some illegal specail character 
    with open(trs_file_path, "rb") as f:
        raw = f.read()

    file = raw.decode("utf-8", errors="replace")

    # Suppression of forbidden character in the file (because some file have if)
    file = re.sub(r"[^\x09\x0A\x0D\x20-\uD7FF\uE000-\uFFFD]","",file)

    parser = etree.XMLParser(recover=True)
    root = etree.fromstring(file.encode("utf-8"), parser)
    dico = {"segments":[] , "word_segments":[] }
    # dico = []

    for turn in root.iter("Turn"):
        elements  = list(turn)
        speaker   = None
        seg_start = float(turn.get("startTime"))
        

        # we add the speakers
        if len(turn.get("speaker").split())==1:
            speaker = turn.get("speaker")
        
        # we check for number of speaker
        # if we don't want part with multiple speaker, this part will be ignore
        if not multiple_speaker and len(turn.get("speaker").split())>=2:
            continue

        # We only want one speaker but not the "enquêteur" in the transcription
        if not multiple_speaker and turn.get("speaker")=="enquêteur":
            continue
        
        for i ,elt in enumerate(elements):
            
            if elt.tag == "Who": speaker = turn.get("speaker").split()[int(elt.get("nb"))]
            
            elif elt.tag == "Sync":
                
                # === 2 different cases ===
                text = []
                seg_start = float(elt.get("time"))
                seg_end   = float(turn.get("endTime"))

                #if there is a text next:
                if elt.tail.strip():
                    text.append(elt.tail.strip())
                    # add_to_dico(elt.tail.strip().split(), float(elt.get("time")),speaker,dico )
                
                else:
                    #we look at the next line until "Sync"
                    j = i+1
                    if elements[j].tag=="Who":
                        text.append(elements[j].tail.strip())
                        speaker = turn.get("speaker").split()[int(elements[j].get("nb"))]
                    else:
                        raise SyntaxError("Problem after 'Sync' empty , there is no 'Who'")
                    
                    # add_to_dico(elements[j].tail.strip().split(), float(elt.get("time")), speaker,dico)
                    # add_to_dico_seg(elements[j].tail.strip(), float(elt.get("time")), speaker, seg_start, seg_end ,dico )
                    
                    # print(text)
                
                #searching for the next "Sync" to have our endTime for the segment
                # or if there is no Sync after (we are at the end) we take the end time of the turn
                for j in range(i+1, len(elements)):
                    if elements[j].tag=="Sync":
                        seg_end = float(elements[j].get("time"))
                        break
                    
                
                #finally adding to the dico
                for j in range(len(text)):
                    if len(text[j])==0 or (len(text[j])==1 and text[j] not in ONE_CHAR_ALLOWED): continue #if there is no word or it is a special character
                    add_to_dico_full(text[j], speaker, seg_start, seg_end, dico)
                    # add_to_dico(text[j],seg_start,speaker,dico)

    # === Saving ===
    with open(output_path , "w") as f:
        json.dump(dico, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    print("=== Starting ...===")

    # === Part for all speakers in the transcription ===
    # input  = "../../CFPR_classique_trs"
    # output = "../../CFPR_classique_cleaned_json"

    # for file in os.listdir(input):
    #     print(f"Transformation of {file} into {file[:-4]}.json ...") 
    #     change_into_json( f"{input}/{file}" , f"{output}/{file[:-4]}.json" )
    
    # print("\n\n")

    # === Part for only the "main" speaker ===
    input  = "../../CFPR_classique_trs"
    output = "../../CFPR_classique_one_speaker_json"

    for file in os.listdir(input):
        print(f"Transformation of {file} into {file[:-4]}.json ...") 
        change_into_json( f"{input}/{file}" , f"{output}/{file[:-4]}.json" , False)
    
    print("=== Completed. ===")