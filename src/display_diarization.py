import matplotlib.pyplot as plt
import numpy as np


def display_bars(file_path):
    """
    Display bar charts representing the repartition of the DER (Diarization Error Rate)
    regarding "Speaker Confusion" , "Missed Detection" and "False Alarm".

    This function present on bar chart for the WhisperX transcription and one for the FrWhisper transcription.

    Parameters:
        file_path (str) : Path of the file with the DER values.

    Returns:
        None.
    
    """
    # === Loading ===
    file  = open(file_path, "r")
    lines = file.readlines()
    
    # === Labels ===
    labels = ["WhisperX" , "FrWhisper"]
    x = np.arange(len(labels))

    # === The different values to display ===
    speaker_confusion = {"ref":[] , "hyp":[] }
    missed_detection  = {"ref":[] , "hyp":[] }
    false_alarm       = {"ref":[] , "hyp":[] }
    total             = {"ref":[] , "hyp":[] }

    for line in lines:
        ref, hyp = line.split("--")
        
        ref = ref.split(";")
        hyp = hyp.split(";")
        
        total["ref"].append(float(ref[0]))
        total["hyp"].append(float(hyp[0]))

        speaker_confusion["ref"].append(float(ref[1]) / float(ref[0]))
        speaker_confusion["hyp"].append(float(hyp[1]) / float(hyp[0]))

        missed_detection["ref"].append(float(ref[2]) / float(ref[0]))
        missed_detection["hyp"].append(float(hyp[2]) / float(hyp[0]))

        false_alarm["ref"].append(float(ref[3]) / float(ref[0]))
        false_alarm["hyp"].append(float(hyp[3]) / float(hyp[0]))
    
    #=== The different stage of the bar ===
    part1 = np.array([100*sum(speaker_confusion["ref"])/len(speaker_confusion["ref"]) ,100*sum(speaker_confusion["hyp"])/len(speaker_confusion["hyp"])])
    part2 = np.array([100*sum(missed_detection["ref"])/len(missed_detection["ref"]) ,100*sum(missed_detection["hyp"])/len(missed_detection["hyp"])])
    part3 = np.array([100*sum(false_alarm["ref"])/len(false_alarm["ref"]) ,100*sum(false_alarm["hyp"])/len(false_alarm["hyp"])])


    # === Construction of the figure ===
    plt.bar(x, part1, label="Speaker Confusion")
    plt.bar(x, part2, bottom=part1, label="Missed Detection")
    plt.bar(x, part3, bottom=part1 + part2, label="False Alarm")

    plt.xticks(x, labels)
    plt.legend()
    plt.ylabel("DER(%)")
    plt.title("Representation of the DER regarding the models")
    plt.savefig(f"./figures/{file_path[:-4]}.png")
    print(f"Saved: ./figures/{file_path[:-4]}.png")
    plt.show()

    # === Some Display ===
    # count          = len(part1)
    # rate_whisperx  = 
    # rate_Frwhisper =
    # print(f"The Diarization Rate after {len(part1)} computations is:")
    # print(f"\t - Whisperx  : {rate_whisperx} % ")
    # print(f"\t - Scd model : {rate_Frwhisper} % ")

if __name__ == "__main__":
    print("=== Starting... ===")
    display_bars("../data/d_rate.txt")
    print("=== Completed. ===")