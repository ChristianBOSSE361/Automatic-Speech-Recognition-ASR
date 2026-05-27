#!../../miniconda3/envs/wx/bin/python3
import torch
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import whisperx
from whisperx.diarize import DiarizationPipeline
import json


# === The different devices ===
# Put different devices if the current gpu memory is low and there more than one gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
diarize_device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu")

# === Load model and processor ===
processor = WhisperProcessor.from_pretrained("aihpi/FrWhisper")
model = WhisperForConditionalGeneration.from_pretrained("aihpi/FrWhisper")
model.to(device)

# === Hugging Face Key (choose a better way after to protect my key) ===
HF_TOKEN = "..."

# === Word by word Alignement (we choose whisperx alignment to have close timestamp) ===
model_a, metadata = whisperx.load_align_model(
            language_code="fr",
            device=device)

# === Diarization model (we also choose diarization from Whisperx to do that) ===
diarize_model = DiarizationPipeline(use_auth_token=HF_TOKEN, device=diarize_device)

# === Batch size ===
batch_size = 70

# === Process audio ===
def transcribe_french(audios_path, output_path,limit=10000):

    """
    Transcribe the different segments of an audio using FrWhisper. 

    Parameters:
        audios_path (str) : folder of segments of the same audio. Ex: ./segmentation/audio/FRA2019_0028
        output_path (str) : folder to store the transcription.
        limit (int)       : Maximum number of audio segments to segment. usefull to test and see how it work with transcribing everything. The default value of 10000, so all the segments. 
    
    Returns:
        None
    """

    audios_file = os.listdir(audios_path)
    folder_name = audios_path.split("/")[-1]
    transcriptions = {"segments":[] , "word_segments":[]}
    count = 1

    # === Creation of the folder to regroup the segments ===
    os.makedirs(output_path,exist_ok=True)
    
    # We check if the transcription has been already made to avoid doing it again x)
    if os.path.exists(f"{output_path}/{folder_name}.json"):
        print(f"*** File {output_path}/{folder_name}.json already exist ***")
        return

    # Main loop
    for i in range(0, len(audios_file), batch_size):
        batch_file = audios_file[i:i+batch_size] if i+batch_size<len(audios_file) else audios_file[i:]

        # Creation of a batch of audios
        audios = []
        for file in batch_file:
            audio_path = f"{audios_path}/{file}"
            audio, sr = librosa.load(audio_path, sr=16000)
            audios.append(audio)
        
        # Process with the model
        inputs = processor(audios,sampling_rate=16000,return_tensors="pt",padding=True)
        input_features = inputs.input_features.to(device)

        # Generate transcription
        with torch.no_grad():
            predicted_ids = model.generate(input_features, language="fr", task="transcribe")
        
        # Decode results
        texts = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        
        # Align, we did it one by one
        for file , text, audio in zip(batch_file, texts, audios):
            
            # We take the start time from the file name
            start_time = int(("".join(os.path.splitext(file)[0])).split("_")[-2])/1000

            duration = len(audio) / sr
            segments = [{"text": text,
                        "start": 0.0,
                        "end"  : duration}]

            result_aligned = whisperx.align(
                segments,
                model_a,
                metadata,
                audio,
                device=device
            )
            
            diarize_segments = diarize_model(audio)
            result_final   = whisperx.assign_word_speakers(diarize_segments, result_aligned)

            try:
                # we modifie the timestamps in each segment in "segments" 
                for segment in result_final["segments"]:
                    
                    # modification of the timestamp of the segment
                    segment["start"] = float(format( segment["start"] + start_time, ".3f"))
                    segment["end"]   = float(format( segment["end"] + start_time, ".3f"))

                    # print(f"Segment: {segment["text"]}: {segment['start']:.3f}s --> {segment['end']:.3f}s ")
                    # print("Words:")
                    # modification of the timestamp in "words"
                    for word in segment["words"]:

                        word['start'] = float(format(word['start'] + start_time, ".3f")) # format just to take 3digits after the dot
                        word['end']   = float(format(word['end'] + start_time, ".3f"))
                
                # we do not need to modify the timest

                # adding in the global transcription
                transcriptions["segments"].extend(result_final["segments"])
                transcriptions["word_segments"].extend(result_final["word_segments"])
                
                print(f">>> Transcription {count}/{len(audios_file)} made. Text:{result_final["segments"][0]["text"]}" )
                
            except Exception as e:
                print(f">>> Transcription {count}/{len(audios_file)} not made because: {e}. Text:{result_final["segments"][0]["text"]} ")
            
            #incrementation 
            count+=1
        
        #if we want to stop the process faster
        if count>= limit: print("*** BREAK: Limit of iteration reached ***");break

    # Saving
    folder_name = audios_path.split("/")[-1]
    
    with open(f"{output_path}/{folder_name}.json", 'w') as f:
        json.dump(transcriptions, f, indent=4, ensure_ascii=False)
    
    print("Transcription Completed.")



if __name__ == "__main__":
    print("=== Starting... ===")

    # === Part 1 : Segmentation using all speakers ===
    input_file  = "../segmentation/audio_all_interviews_2024_2025"
    output_path = "../segmentation/transcriptions_all_interviews_2024_2025"

    for dir_file in os.listdir(input_file):
        folder_of_segments = f"{input_file}/{dir_file}"
        print(f"*** Transcription of {folder_of_segments} **** ")
        
        transcribe_french(folder_of_segments, output_path)

    # === Part 2 : Segmentation using only on speaker (the "main" one) ===
    # input_file  = "../segmentation/audio_one_speaker"
    # output_path = "../segmentation/transcriptions_one_speaker"

    # for dir_file in os.listdir(input_file):
    #     folder_of_segments = f"{input_file}/{dir_file}"
    #     print(f"*** Transcription of {folder_of_segments} ***")
    #     transcribe_french(folder_of_segments, output_path)

    # transcribe_french("../segmentation/audio/FRA2019_0028")

    print("=== Completed ===")
