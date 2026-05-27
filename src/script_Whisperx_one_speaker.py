#!../../miniconda3/envs/wx/bin/python3
"""
This python file aims to made the transcription of the different audios using WhisperX.
Otherwise, the difference between "script_folder" is the fact that the input is a folder of folders containing audio files.
It uses for the transcription with only one speaker in the audio.
"""
import json
import os
import glob
import torch
from pydub import AudioSegment
import whisperx
from whisperx.diarize import DiarizationPipeline
#import gc #added

# Local fix to use ffmpeg on server
os.environ["PATH"] = "/home/getalp/bossec/cfpr/ffmpeg-7.0.2-amd64-static:" + os.environ["PATH"]
AudioSegment.converter = "/home/getalp/bossec/cfpr/ffmpeg-7.0.2-amd64-static/ffmpeg"
AudioSegment.ffprobe = "/home/getalp/bossec/cfpr/ffmpeg-7.0.2-amd64-static/ffprobe"
print("ffmpeg path:", AudioSegment.converter)
print("ffprobe path:", AudioSegment.ffprobe)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Local fix : patch torch.load to use weights_only=False for all loads
_original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = patched_load

# Variables
device = "cuda"
whisper_device_index=0
diarize_device = "cuda:0"

input_folder = "/home/getalp/bossec/cfpr/segmentation/audio_one_speaker/"  # Folder containing folders of audio files
output_folder = "/home/getalp/bossec/cfpr/output_one_speaker/"  # Folder for output JSON files

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

batch_size = 64 # modified (was previously 16)
compute_type = "float32"
HF_TOKEN = "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"

# Supported audio formats
audio_extensions = ['*.mp3', '*.wav', '*.ogg', '*.flac', '*.m4a']

# Load models once (outside the loop for efficiency)
print("Loading Whisper model...")
model = whisperx.load_model("large-v2", device, device_index=whisper_device_index, compute_type=compute_type)

print("Loading diarization model...")
diarize_model = DiarizationPipeline(use_auth_token=HF_TOKEN, device=diarize_device)

# Loop over all folder containing the different segment of speach
for folder in os.listdir(input_folder):
    print(f"=== Starting transcription folder : {folder} ===")

    # Get all audio files in the folder
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(input_folder,folder,ext)))

    print(f"Found {len(audio_files)} audio file(s) to process")

    # create the folder
    os.makedirs(os.path.join(output_folder,folder), exist_ok=True)
    
    # Process each audio file
    for idx, audio_file in enumerate(audio_files, 1):
        try:
            print(f"\n[{idx}/{len(audio_files)}] Processing: {os.path.basename(audio_file)}")
            
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            
            # # Just to check if the transcription was already done
            if os.path.exists(os.path.join(output_folder,folder ,base_name + "_transcription_output.json") ):
                print(f"*** Files for {os.path.join(output_folder,folder, base_name)} aready exist. ***")
                continue  

            # 1. Transcribe with original whisper (batched)
            print("*** Transcribing ***")
            audio = whisperx.load_audio(audio_file)
            result = model.transcribe(audio, batch_size=batch_size)
            
            # 2. Align whisper output
            print("*** Aligning ***")
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
            
            # Delete alignment model to free memory
            del model_a
            #gc.collect() #added: delete unreferenced object in memory
            torch.cuda.empty_cache() 
            #torch.cuda.synchronize() #added: wait until every gpus finish his work
            
            # 3. Assign speaker labels
            print("*** Diarizing ***")
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            

            # 4.
            start_time= int(base_name.split("_")[2])/1000
            # we modifie the timestamps in each segment in "segments"
            # I notice that oddly that also modify the timestamp in the "word_segments" part
            for segment in result["segments"]:
                
                #modification of the timestamp of the segment
                segment["start"] = float(format( segment["start"] + start_time, ".3f"))
                segment["end"]   = float(format( segment["end"] + start_time, ".3f"))

                # modification of the timestamp in the words
                # print("Words:")
                for word in segment["words"]:

                    word['start'] = float(format(word['start'] + start_time, ".3f")) # format just to take 3digits after the dot
                    word['end']   = float(format(word['end'] + start_time, ".3f")) 


            # Save to JSON files in output folder
            output_file = os.path.join(output_folder,folder,base_name + "_transcription_output.json")

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            segments_file = os.path.join(output_folder, folder,base_name + "_segments.json")
            with open(segments_file, 'w', encoding='utf-8') as f:
                json.dump(result["segments"], f, indent=2, ensure_ascii=False)
            
            sentence_file = os.path.join(output_folder,folder,base_name + "_sentences_only.json")
            segments_without_words = [
                {k: v for k, v in segment.items() if k != "words"}
                for segment in result["segments"]
            ]
            with open(sentence_file, 'w', encoding='utf-8') as f:
                json.dump(segments_without_words, f, indent=2, ensure_ascii=False)
            
            print(f"  COMPLETED: {base_name}")
            
        except Exception as e:
            print(f"  ERROR processing {audio_file}: {str(e)}")
            continue
    
    print(f"=== Completed folder :{folder} === ")

print(f"\n=== Processing complete! ===")
print(f"Processed {len(audio_files)} file(s)")
print(f"Output saved to: {output_folder}")