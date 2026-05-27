import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import date

def display_plot(scores_file, all_speakers = True):
    """
    Display a plot showing WER values for each audio file.

    Each audio is associated with three WER metrics:
        - WER from WhisperX transcription
        - WER from FrWhisper transcription
        - WER from the merged transcription

    The function also indicates whether all speakers were included in the transcription.

    Note:
        Including all speakers in the transcription does not necessarily mean that all
        speakers are used when computing the WER. It is possible to restrict the computation
        to interviewees only, even if multiple speakers are present.

    Parameters:
        scores_file (str): Path to the file containing WER values.
        all_speakers (bool): Whether to include all speakers in the WER computation.
            - True: include all speakers (default)
            - False: include only interviewees

    Returns:
        None
    """ 

    # === Loading ===
    with open(scores_file, "r") as f:
        scores = f.readlines()
    
    # === Initialization ===
    file_name , data1 , data2 = [], [], []
    if all_speakers: data3 = []

    # === Affectation of the values ===
    for line in scores:
        value = line.strip().split(";")
        
        file_name.append(value[0])     # name of the file
        data1.append(float(value[1]))  # whisperx 
        data2.append(float(value[2]))  # Scd Model
        
        if all_speakers: data3.append(float(value[3]))     # Merged

    # === Colors ===
    col1 =  "blue"
    col2 =  "red"
    col3 =  "green"

    # === Construction of the plot ===
    x = range(len(data1))

    plt.figure(figsize=(30, 14))

    plt.plot(x,data1,label= "Whisperx" ,color = col1, marker="o", linestyle="dashed", linewidth=2, markersize=8)
    plt.plot(x,data2,label= "FrWhisper",color = col2, marker="o", linestyle="dashed", linewidth=2, markersize=8)
    
    if all_speakers: plt.plot(x,data3,label= "Merged"   ,color = col3, marker="o", linestyle="dashed", linewidth=2, markersize=8)
    
    plt.legend(fontsize=12)
    plt.xticks(x,file_name, rotation=90)

    plt.xlabel("Audios")
    plt.ylabel("WER (%)")
    plt.title(f"Representation of the WER for the different audio (all speakers={all_speakers})", fontsize=20)
    
    plt.tight_layout()
    plt.savefig(f"./figures/{scores_file[:-4]}.png")

    print(f"*** Saved ./figures/{scores_file[:-4]}.png***")


def boxplot_accent(scores_file, all_speakers = True):
    """
    Display some box plots showing the distributions of the WER values for the different accents we have.

    This function display three box plots:
            - one for the WhisperX transcription
            - one for the FrWhisper transcription
            - one for the merged transcription
    
    The function also indicates whether all speakers were included in the transcription.

    Note:
        Including all speakers in the transcription does not necessarily mean that all
        speakers are used when computing the WER. It is possible to restrict the computation
        to interviewees only, even if multiple speakers are present.

    Parameters:
        scores_file (str): Path to the file containing WER values.
        all_speakers (bool): Whether to include all speakers in the WER computation.
            - True: include all speakers (default)
            - False: include only interviewees

    Returns:
        None

    """

    # === Loading of the scores file ===
    file_names, data1, data2 = [], [], []
    if all_speakers: data3 = []
    
    number_good_merged = 0

    with open(scores_file, "r") as file:
        lines = file.readlines()

    # === Affectation of the values ===
    for line in lines:
        values = line.strip().split(";")
        # if len(values) < 4:continue

        file_names.append(values[0])    # name of the file
        data1.append(float(values[1]))  # whisperx
        data2.append(float(values[2]))  # second model
        
        
        if all_speakers:
            data3.append(float(values[3]))  # merged
            if data3[-1] <= data1[-1]: number_good_merged += 1

    # === Reading accents file ===
    df_accents = pd.read_csv("../data/CFPR_with_accents_labels.csv", sep="\t")

    # === Build a DataFrame with scores + accent label ===
    if all_speakers :
        df_scores = pd.DataFrame({
            "file_name"    : file_names,
            "wer_whisperx" : data1,
            "wer_frwhisper": data2,
            "wer_merged"   : data3})
    else:
        df_scores = pd.DataFrame({
            "file_name"    : file_names,
            "wer_whisperx" : data1,
            "wer_frwhisper": data2})

    df_merged = df_scores.merge(
        df_accents[["nom_dossier", "label"]],
        left_on="file_name",
        right_on="nom_dossier",
        how="left")

    accents_list = sorted(df_merged["label"].dropna().unique())

    # === Build data groups for each model ===
    if all_speakers:
        models = {
            "WhisperX" : "wer_whisperx",
            "FrWhisper": "wer_frwhisper",
            "Merged"   : "wer_merged"}
    else:
        models = {
            "WhisperX" : "wer_whisperx",
            "FrWhisper": "wer_frwhisper"}

    # === Main Loop ===
    for model_name, col in models.items():

        # One group per accent + one "Global" group at the beginning 
        groups      = ["Global"] + list(accents_list)
        plot_data   = [df_merged[col].dropna().tolist()] # Global

        for accent in accents_list:
            subset = df_merged[df_merged["label"] == accent][col].dropna().tolist()
            plot_data.append(subset)

        # Plot 
        fig, ax = plt.subplots(figsize=(max(10, len(groups) * 1.4), 6))

        bp = ax.boxplot(
            plot_data,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            notch=False,
        )

        # Color: Global in grey, accents in blue shades
        colors = ["#B0B0B0"] + [
            plt.cm.tab20(i / len(accents_list)) for i in range(len(accents_list))
        ]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        # Annotations: median value above each box
        for i, (box_data, med_line) in enumerate(zip(plot_data, bp["medians"])):
            if box_data:
                median_val = np.median(box_data)
                x = med_line.get_xdata().mean()
                ax.text(x, median_val + 0.5, f"{median_val:.1f}",
                        ha="center", va="bottom", fontsize=8, color="black")

        # Annotations : Legends
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("WER (%)")
        ax.set_title(f"WER per accent — {model_name}")
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)

        plt.tight_layout()
        os.makedirs("./figures", exist_ok=True)
        out_path = f"./figures/{scores_file[:-4]}_{model_name.replace(' ', '_')}_accent_boxplot.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"*** Saved: {out_path} ***")

    print(f"Number of good merge : {number_good_merged}/{len(file_names)}\n")


def boxplot_gender(scores_file, all_speakers = True):
    """
    Display some box plots showing the distributions of the WER values regarding the gender ("H":Men ; "F": Women)
    but also some age groups ( 20-30, 30-40, 40-50, 50-60, 60-70, 70-80, 80-90, 90-100 years old ).

    This function display three box plots for gender and three for age groups (so 6):
            - two for the WhisperX transcription
            - two for the FrWhisper transcription
            - two for the merged transcription
    
    The function also indicates whether all speakers were included in the transcription.

    Note:
        Including all speakers in the transcription does not necessarily mean that all
        speakers are used when computing the WER. It is possible to restrict the computation
        to interviewees only, even if multiple speakers are present.

    Parameters:
        scores_file (str): Path to the file containing WER values.
        all_speakers (bool): Whether to include all speakers in the WER computation.
            - True: include all speakers (default)
            - False: include only interviewees

    Returns:
        None

    """
    
    # === Loading of the scores file ===
    file_names, data1, data2 = [], [], []
    if all_speakers : data3 = []
    number_good_merged = 0

    with open(scores_file, "r") as file:
        lines = file.readlines()

    # === Affectation of the values ===
    for line in lines:
        values = line.strip().split(";")
        if len(values) < 4:continue

        file_names.append(values[0])    # name of the file
        data1.append(float(values[1]))  # whisperx
        data2.append(float(values[2]))  # second model
        
        if all_speakers :
            data3.append(float(values[3]))  # merged

    # === Reading accents file ===
    df_gender_age = pd.read_csv("../data/cfpr_gender_age.csv", sep=",")
    
    # === We add a column for the age (in years) ===
    df_gender_age["date_naissance"] = pd.to_datetime(df_gender_age["date_naissance"], format="%m/%d/%Y")
    df_gender_age["age"] = (pd.Timestamp("today") - df_gender_age["date_naissance"]).dt.days // 365

    # === Build a DataFrame with scores + accent label ===
    if all_speakers:
        df_scores = pd.DataFrame({
        "file_name"    : file_names,
        "wer_whisperx" : data1,
        "wer_frwhisper": data2,
        "wer_merged"   : data3})
    else:
        df_scores = pd.DataFrame({
        "file_name"    : file_names,
        "wer_whisperx" : data1,
        "wer_frwhisper": data2})

    df_merged = df_scores.merge(
        df_gender_age[["nom_dossier", "genre", "age"]],
        left_on="file_name",
        right_on="nom_dossier",
        how="left"
    )

    # === Build data groups for each model ===
    if all_speakers:
        models = {
            "WhisperX" : "wer_whisperx",
            "FrWhisper": "wer_frwhisper",
            "Merged"   : "wer_merged"}
    else:
        models = {
            "WhisperX" : "wer_whisperx",
            "FrWhisper": "wer_frwhisper"}

    # === GENDER PART: ===
    gender_list = ["H", "F"]

    # === Main Loop ===
    for model_name, col in models.items():

        # One group per accent + one "Global" group at the beginning 
        groups      = ["Global"] + list(gender_list)
        plot_data   = [df_merged[col].dropna().tolist()] # Global

        for g in gender_list:
            subset = df_merged[df_merged["genre"] == g][col].dropna().tolist()
            plot_data.append(subset)

        # Plot 
        fig, ax = plt.subplots(figsize=(max(10, len(groups) * 1.4), 6))

        bp = ax.boxplot(
            plot_data,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            notch=False,
        )

        # Color: Global in grey, accents in blue shades
        colors = ["#B0B0B0"] + [
            plt.cm.tab20(i / len(gender_list)) for i in range(len(gender_list))
        ]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        # Annotations: median value above each box
        for i, (box_data, med_line) in enumerate(zip(plot_data, bp["medians"])):
            if box_data:
                median_val = np.median(box_data)
                x = med_line.get_xdata().mean()
                ax.text(x, median_val + 0.5, f"{median_val:.1f}",
                        ha="center", va="bottom", fontsize=8, color="black")

        # Annotations : Legends
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("WER (%)")
        ax.set_title(f"WER per gender — {model_name}")
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)

        plt.tight_layout()
        os.makedirs("./figures", exist_ok=True)
        out_path = f"./figures/{scores_file[:-4]}_{model_name.replace(' ', '_')}_gender_boxplot.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"*** Saved: {out_path} ***")
    
    # === AGE PART: ===
    age_list = [ [i,i+10] for i in range(20,100,10) ]
    
    # === Main Loop ===
    for model_name, col in models.items():

        # One group per accent + one "Global" group at the beginning 
        groups      = ["Global"] + [f"{age_list[i][0]}-{age_list[i][1]}" for i in range(len(age_list))]
        plot_data   = [df_merged[col].dropna().tolist()] # Global

        for interval in age_list:
            subset = df_merged[(interval[0]<=df_merged["age"]) & (df_merged["age"]< interval[1])][col].dropna().tolist()
            plot_data.append(subset)

        # Plot 
        fig, ax = plt.subplots(figsize=(max(10, len(groups) * 1.4), 6))

        bp = ax.boxplot(
            plot_data,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            notch=False,
        )

        # Color: Global in grey, accents in blue shades
        colors = ["#B0B0B0"] + [
            plt.cm.tab20(i / len(age_list)) for i in range(len(age_list))
        ]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        # Annotations: median value above each box
        for i, (box_data, med_line) in enumerate(zip(plot_data, bp["medians"])):
            if box_data:
                median_val = np.median(box_data)
                x = med_line.get_xdata().mean()
                ax.text(x, median_val + 0.5, f"{median_val:.1f}",
                        ha="center", va="bottom", fontsize=8, color="black")

        # Annotations : Legends
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("WER (%)")
        ax.set_title(f"WER per age — {model_name}")
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)

        plt.tight_layout()
        os.makedirs("./figures", exist_ok=True)
        out_path = f"./figures/{scores_file[:-4]}_{model_name.replace(' ', '_')}_age_boxplot.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"*** Saved: {out_path} ***")



if __name__ == "__main__":
    #TO DO
    print("=== Starting ... ===")
    
    scores_files = [("../data/scores_all_speakers.txt", True),("../data/scores_speaker_selected.txt", True) ,("../data/scores_one_speaker.txt", False)]

    for scores in scores_files:
        display_plot(scores[0], scores[1])
        boxplot_gender(scores[0], scores[1])
        boxplot_accent(scores[0], scores[1])
        
    print("=== Completed. ===")