#!../../miniconda3/envs/wx/bin/python3
import jiwer
from pyannote.core import Segment, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate
import os
from merge_rover import seq_words, load_words, load_segments,pre_processing, FILLERS


# ===============================================
# ================= WER Metrics =================
# ===============================================


# === For transcription with all speakers ====
def WER_metric(whisperx_trs_path, second_model_trs_path, all_speakers=True):
    """
    Compute the WER (Word Error Rate) between 2 cleaned transcriptions.

    The function filters the segments in the transcriptions depending
    whether we take all speakers or only the interviwees. 

    Parameters:
        whisperx_trs_path     (str) : Path of the gold or the reference transcription.
        second_model_trs_path (str) : Path of the hypothesis transcription.
        all_speakers          (bool): Whether to take all speakers or not.
                            - True : Take all the speakers (default),
                            - False: Take only the interviwee.
    
    Returns:
        float: The WER.
    """

    # === Loading ===
    ref_transcription = load_words(whisperx_trs_path)
    hyp_transcription = load_words(second_model_trs_path)
    
    ref_segments = load_segments(whisperx_trs_path)
    hyp_segments = load_segments(second_model_trs_path)

    # === We look at the speakers ===
    if all_speakers:
        # === Transform into a sequence of words ===
        ref_words = seq_words(ref_transcription, all_speakers)
        hyp_words = seq_words(hyp_transcription, all_speakers)
    else:
        # === Building of ref annotation ===
        ref = Annotation()
        for i in range(len(ref_segments)):
            start, end = ref_segments[i]["start"] , ref_segments[i]["end"]
            ref[Segment(start, end)] = ref_segments[i]["speaker"]

        # === Building of hyp annotaion ===
        hyp = Annotation()
        for i in range(len(hyp_segments)):
            start, end = hyp_segments[i]["start"] , hyp_segments[i]["end"]
            # if "speaker" not in hyp_segments[i].keys(): print(f"Segment with no speaker:", hyp_segments[i]["text"])
            hyp[Segment(start, end)] = hyp_segments[i]["speaker"] if "speaker" in hyp_segments[i].keys() else None

        metric = DiarizationErrorRate()
        mapping = metric.optimal_mapping(ref, hyp)
        #inverse the mapping because currently it is: hyp speaker:ref speaker
        mapping = {v:k for k,v in mapping.items()}
        
        # print(type(mapping),  "enquêteur" in mapping.keys()) 

        if "enquêteur" in mapping.keys():
            ref_words = [word["word"] for word in ref_transcription if word["speaker"]!=mapping["enquêteur"]]
            hyp_words = [word["word"] for word in hyp_transcription if word["speaker"]!=mapping["enquêteur"]]
        else:
            ref_words = [word["word"] for word in ref_transcription ]
            hyp_words = [word["word"] for word in hyp_transcription ]

    # === Computation ===
    output = jiwer.process_words(" ".join(ref_words), " ".join(hyp_words)) #make the argument into character is better

    wer = 100*output.wer

    # === Some display to debug ===
    # print("The WER is:",wer,"%")
    # print(jiwer.visualize_alignment(output))
    return wer , ref_words, hyp_words


# === For transcription with only one speaker ===
def WER_metric_one_speaker(whisperx_trs_path, second_model_trs_path):
    """
    Compute the WER between 2 cleaned transcriptions.   

    Parameters:
            whisperx_trs_path     (str) : Path of the gold or reference transcription.
            second_model_trs_path (str) : Path of folder containing the hypothesis transcriptions.
    
    Returns:
        float: The WER.
    """

    # === Initialization ===
    hyp_words = []
    files = sorted(os.listdir(second_model_trs_path), key= lambda file: int(file.split("_")[2]))

    # === Construction of the hypothsis sequence of word ===
    for i in range (len(files)):
        file = files[i]
        if "transcription" not in file.split("_"): continue
        
        # === Loading ===
        hyp_transcription = load_words(f"{second_model_trs_path}/{file}")
        hyp_transcription = pre_processing(hyp_transcription)
        
        # === Transform into a sequence of words ===
        hyp_words.extend(seq_words(hyp_transcription))

    # same thing for the ref transcription
    ref_transcription = load_words(whisperx_trs_path)
    
    # ref_transcription = pre_processing(ref_transcription)
    ref_words         = seq_words(ref_transcription)

    # === Computation ===
    output = jiwer.process_words(" ".join(ref_words), " ".join(hyp_words)) #make the argument into character is better

    wer = 100*output.wer
    
    # === Some display to debug ===
    # print("The WER is:",wer,"%")
    # print(jiwer.visualize_alignment(output))
    return wer , ref_words, hyp_words


# ===============================================
# ============== Main Functions =================
# ===============================================

# === For all speakers in the transcription and in the audio ===
def compute_WER(gold_trs_path, other_trs_folder_path, output_path=None, all_speakers=True, limit=10000):
    """
    Compute the WER values between a gold transcription and his corresponding hypothesis transcriptions (cleaned transcriptions).
    
    This function compute exactly three things:
        - WER between the gold transcription and the whisperx transcription,
        - WER between the gold transcription and the Second model choose (here FrWhisper).
        - WER between the gold transcription and the merged transcription.

    It filters the segments in the transcriptions depending
    whether we take all speakers or only the interviwees. 

    The results are stored in a text file in with each line correspond to the WER for an Audio.
    Specifically, on each line, we have : "name of the audio";"Wer regarding whisperx";"wer regarding scd model";"wer regarding merdeg trs".
    
    Parameters:
        gold_trs_path        (str) : Path of the folder containing the paths of the gold or the ref_segments transcriptions.
        other_trs_folder_path(str) : Path of the folder containing the paths of the other transcriptions (for the second model).
        output_path          (str) : The path of the file in which the results will be send.
        limit                (int) : Maximum number of audios to compute their wer.
    
    Returns:
        dict: Ditionary with the WER values for each transcription. The dictionary has the following format:
                    { "whisperx":list, "other_model":list , "merged":list }.

    """

    # === Initialization ===
    wer_values = {"whisperx":[], "other_model":[], "merged":[] }
    wer_liste  = ""
    count      = 0
    seen       = 0

    # === Computation ===
    for ref_file in os.listdir(gold_trs_path):
        line=""
        ref,hyp,mer = False,False,False #to know if for one audio we have the 3 transcriptions
        
        # === For the Whisperx transcription ===
        for hyp_file in os.listdir(other_trs_folder_path[0]):

            if ref_file[:-5]!=hyp_file[:12]: continue

            
            value,_,_ = WER_metric(f"{gold_trs_path}/{ref_file}", f"{other_trs_folder_path[0]}/{hyp_file}", all_speakers)
            wer_values["whisperx"].append(value)
            line+= f"{ref_file[:-5]};{value}" #we save the name and the corresponding Wer
            ref = True
            break
        
        # === For the Second model transcription ===
        for hyp_file in os.listdir(other_trs_folder_path[1]):

            if ref_file[:-5]!=hyp_file[:-5]: continue
            
            value,_,_ = WER_metric(f"{gold_trs_path}/{ref_file}", f"{other_trs_folder_path[1]}/{hyp_file}", all_speakers)
            wer_values["other_model"].append(value)
            line+= f";{value}" #we save the name and the corresponding Wer
            hyp = True
            break

        # === For the merged transcription ===
        for hyp_file in os.listdir(other_trs_folder_path[2]):

            if ref_file[:-5]!=hyp_file[:-5]: continue
            
            value,_,_ = WER_metric(f"{gold_trs_path}/{ref_file}", f"{other_trs_folder_path[2]}/{hyp_file}", all_speakers)
            wer_values["merged"].append(value)
            line+= f";{value}\n" #we save the name and the corresponding Wer
            mer = True
            break
        
        seen+=1
        # add the line only if the 3 measures appeare
        if ref and hyp and mer:
            wer_liste += line
            count += 1
        
        # just to be able to stop early
        if count >= limit: print("*** BREAK: Limit of iteration reached ***");break
    
    # === Saving in a file ===
    if output_path!=None:
        with open(output_path, "w") as f:
            f.write(wer_liste)
    

    # === Some display ===
    print(f"The WER after {count}(seen:{seen}) computations is:")
    print(f"\t - Whisperx :{sum(wer_values["whisperx"])/len(wer_values["whisperx"])}")
    print(f"\t - Scd model:{sum(wer_values["other_model"])/len(wer_values["other_model"])}")
    print(f"\t - Merged   :{sum(wer_values["merged"])/len(wer_values["merged"])}")
    
    print("List of WER values returned.")

    return wer_values

# === For only one speaker in the transcription and in the audio ===
def compute_WER_one_speaker(gold_trs_path, other_trs_folder_path, output_path=None, limit=10000):
    """
    Compute the WER values between a gold transcription and his corresponding hypothesis transcriptions (cleaned transcriptions).

    This function compute exactly two things:
        - WER between the gold transcription and the whisperx transcription,
        - WER between the gold transcription and the Second model choose (here FrWhisper).

    The results are stored in a text file in with each line correspond to the WER for an Audio.
    Specifically, on each line, we have : "name of the audio";"Wer regarding whisperx";"wer regarding scd model";0.0 .
    0.0 is just added to have the same file format as 'compute_WER' to be able to use the same functions to display.


    Parameters:
        gold_trs_path         (str) : Path of the folder containing the paths of the gold or the ref_segments transcriptions.
        other_trs_folder_path (str) : Path of the folder containing the paths of the folder of the hypothesis transcriptions.
        output_path           (str) : The path of the file in which the results will be send.
        limit                 (int) : Maximum number of audios to compute their WER.

    Returns:
        dict: Ditionary with the WER values for each transcription. The dictionary has the following format:
                    { "whisperx":list, "other_model":list }.
    """
    # === Initialization ===
    wer_values = {"whisperx":[], "other_model":[] }
    wer_liste  = ""
    count      = 0
    seen       = 0

    # === Computation ===
    for ref_file in os.listdir(gold_trs_path):
        line=""
        ref,hyp = False,False #to know if for one audio we have the 3 transcriptions
        
        # We don't want to take into account the CIV2020_0028 because of the size after the segmentation
        if ref_file[:-5]=="CIV2020_0028": continue
        
        # === For the Whisperx transcription ===
        for hyp_file in os.listdir(other_trs_folder_path[0]):
            
            if ref_file[:-5]!=hyp_file: continue

            value,_,_ = WER_metric_one_speaker(f"{gold_trs_path}/{ref_file}", f"{other_trs_folder_path[0]}/{hyp_file}")
            wer_values["whisperx"].append(value)
            line+= f"{ref_file[:-5]};{value}" #we save the name and the corresponding Wer
            ref = True
            return
            break
        
        # === For the Second model transcription ===
        for hyp_file in os.listdir(other_trs_folder_path[1]):

            if ref_file[:-5]!=hyp_file[:-5]: continue
            
            value,_,_ = WER_metric(f"{gold_trs_path}/{ref_file}", f"{other_trs_folder_path[1]}/{hyp_file}")
            wer_values["other_model"].append(value)
            line+= f";{value};{0.0}\n" #we save the name and the corresponding Wer
            hyp = True
            break
        
        seen+=1
        # add the line only if the 3 measures appeare
        if ref and hyp:
            wer_liste += line
            count += 1
        
        # just to be able to stop early
        if count >= limit: print("*** BREAK: Limit of iteration reached ***");break
    
    # === Saving in a file ===
    if output_path!=None:
        with open(output_path, "w") as f:
            f.write(wer_liste)
    

    # === Some display ===
    print(f"The WER after {count}(seen:{seen}) computations is:")
    print(f"\t - Whisperx :{sum(wer_values["whisperx"])/len(wer_values["whisperx"])}")
    print(f"\t - Scd model:{sum(wer_values["other_model"])/len(wer_values["other_model"])}")
    
    print("List of WER values returned.")

    return wer_values


if __name__== "__main__":
    print("=== Starting ...===")
    
    # === Part1 : For all speakers ===
    print("*** For all speakers: ***")
    refs_path   = "../CFPR_classique_cleaned_json"
    hyps_path   =  ["../output_cleaned","../segmentation/transcriptions_cleaned","../merged_transcriptions_full"]
    output_path = "../data/scores_all_speakers.txt"
    all_speakers =  True

    print("Whisperx transcription path     = ",refs_path)
    print("Second model transcription path = ",hyps_path)

    compute_WER(refs_path, hyps_path, output_path=output_path, all_speakers=all_speakers)
    
    # # === Part2 : For all speakers in the audio but we select the main speaker ===
    # print("\n***For speaker selected:***")
    # output_path = "../data/scores_speaker_selected.txt"
    # all_speakers =  False

    # print("Whisperx transcription path     = ",refs_path)
    # print("Second model transcription path = ",hyps_path)

    # compute_WER(refs_path, hyps_path, output_path=output_path, all_speakers=all_speakers)

    # # === Part3 : For only one speaker in the audio ===
    # print("\n***For only one speaker:***")
    # refs_path   = "../CFPR_classique_one_speaker_json"
    # hyps_path   =  ["../output_one_speaker","../segmentation/transcriptions_one_speaker_cleaned"]
    # output_path = "../data/scores_one_speaker.txt"
    
    # print("Whisperx transcription path     = ",refs_path)
    # print("Second model transcription path = ",hyps_path)

    # compute_WER_one_speaker(refs_path, hyps_path, output_path=output_path)
    
    print("=== Completed. ===")