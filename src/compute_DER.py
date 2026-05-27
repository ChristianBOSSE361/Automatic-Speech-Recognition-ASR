from pyannote.core import Segment, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate

from merge_rover import  load_segments
import json
import os

# ===============================================
# ============= DER Metric ======================
# ===============================================

def DER_metric(reference_trs_path , hypothesis_trs_path):
    """
    Compute the DER (Diarization Error Rate) between a reference transcription and a hypothesis trancription.

    Parameters:
        reference_trs_path  (str) : Path of the reference transcription.
        hypothesis trs_path (str) : Path of the hypothesis transcription.

    Returns:
        None.
    """

    # === Loading the transcriptions ===
    reference = load_segments(reference_trs_path)
    hypothesis= load_segments(hypothesis_trs_path)

    # === Building of ref annotation ===
    ref = Annotation()
    for i in range(len(reference)):
       start, end = reference[i]["start"] , reference[i]["end"]
       ref[Segment(start, end)] = reference[i]["speaker"]

    # === Building of hyp annotaion ===
    hyp = Annotation()
    for i in range(len(hypothesis)):
        start, end = hypothesis[i]["start"] , hypothesis[i]["end"]
        if "speaker" not in hypothesis[i].keys(): print(f"Segment with no speaker:", hypothesis[i]["text"])
        hyp[Segment(start, end)] = hypothesis[i]["speaker"] if "speaker" in hypothesis[i].keys() else None

    # ==== COmputation of the DER ===
    metric = DiarizationErrorRate()
    der =  metric(ref, hyp, detailed=True)

    # === Some Display ===
    print("The DER is:",100*der["diarization error rate"], "%")

    return der


# ===============================================
# ============= Main Function ===================
# ===============================================

def diarization_rate(reference_trs_path, hypothesis_trs_path, output_path, limit=1000):
    """
    Compute the DER (Diarization Error Rate) between a reference trancription and a hypothesis transcription

    This function compute exactly two things:
            - DER between the reference transcription and the WhisperX transcription,
            - DER between the reference transcription and the FrWhisper trancription.
    
    Parameters:
        reference_trs_path  (str) : Path of the reference transcription.
        hypothesis_trs_path (str) : Path of the hypothesis transcription.
        output_path         (str) : Path of the file to save the results.
        limit               (int) : Maximum number of audio to compute their DER.
    
    Returns:
        None
    """
    # === Initialization ===
    wer_values = {"whisperx":[], "other_model":[] }
    wer_liste  = ""
    count      = 0

    # === Computation ===
    for ref_file in os.listdir(reference_trs_path):
        line=""
        ref,hyp = False,False #to know if for one audio we have the 3 transcriptions
        
        # === For the Whisperx transcription ===
        for hyp_file in os.listdir(hypothesis_trs_path[0]):
            
            if ref_file[:-5]!=hyp_file[:12]: continue
            # print(ref_file)
            value = DER_metric(f"{reference_trs_path}/{ref_file}", f"{hypothesis_trs_path[0]}/{hyp_file}")
            # if value<5: print(ref_file, value);return
            wer_values["whisperx"].append(100*value["diarization error rate"])
            line+= f"{value["total"]};\
                    {value["confusion"]};\
                    {value["missed detection"]};\
                    {value["false alarm"]}--" #we save the name and the corresponding Wer
            ref = True
            break
        
        # === For the Second model transcription ===
        for hyp_file in os.listdir(hypothesis_trs_path[1]):

            if ref_file[:-5]!=hyp_file[:-5]: continue
            
            value = DER_metric(f"{reference_trs_path}/{ref_file}", f"{hypothesis_trs_path[1]}/{hyp_file}")
            wer_values["other_model"].append(100*value["diarization error rate"])
            #we save the name and the corresponding Wer
            line+= f"{value["total"]};\
                    {value["confusion"]};\
                    {value["missed detection"]};\
                    {value["false alarm"]}\n" #we save the name and the corresponding Wer

            hyp = True
            break
        
        # add the line only if the 2 measures appeare
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
    print(f"The Diarization Rate after {count} computations is:")
    print(f"\t - Whisperx :{sum(wer_values["whisperx"])/len(wer_values["whisperx"])} % with max={max(wer_values["whisperx"])} % and min={min(wer_values["whisperx"])} % ")
    print(f"\t - Scd model:{sum(wer_values["other_model"])/len(wer_values["other_model"])} % with max={max(wer_values["other_model"])} % and min={min(wer_values["other_model"])} % ")
    
    print("List of WER values returned.")

    return wer_values


if __name__ == "__main__":
    print("=== Starting... ===")
    
    # === Arguments ===
    ref_trs = "../CFPR_classique_cleaned_seg_json"
    hyp_trs = ["../output_cleaned","../segmentation/transcriptions"]
    output  = "../data/d_rate.txt"

    print("Reference Transcription path  :", ref_trs)
    print("Hypothesis Transcription path :", hyp_trs)

    # === Running ===
    diarization_rate(ref_trs, hyp_trs, output)

    print("=== Completed. ===")