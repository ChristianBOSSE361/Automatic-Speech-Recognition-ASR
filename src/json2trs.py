import json
import os
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom

def print_usage():
    usage = """
Usage: python script_name.py <input_file.json>

Description:
    Converts a JSON transcription file into a .trs (Transcriber) XML format.
    The output file will be saved in the same directory with a .trs extension.
    """
    print(usage)

def generate_trs(input_path):
    # 1. Basic File Checks
    if not os.path.exists(input_path):
        print(f"Error: The file '{input_path}' does not exist.")
        return

    if not input_path.lower().endswith('.json'):
        print("Error: Input file must be a .json file.")
        return

    # 2. Load JSON Data
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON. Ensure the file is valid.")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    segments = data.get("segments", [])
    if not segments:
        print("Error: No 'segments' found in the JSON data.")
        return

    # 3. Process Data (Internal Structure & Stats)
    all_speakers = set()
    total_segments = len(segments)
    speaker_None = False
    
    for seg in segments:
        # Check segment speaker
        if "speaker" in seg:
            all_speakers.add(seg["speaker"])
        # Check individual word speakers (for overlapping/multi-speaker turns)
        for word in seg.get("words", []):
            if "speaker" in word:
                if word["speaker"]!=None:all_speakers.add(word["speaker"])
                else: speaker_None = True

    sorted_speakers = sorted(list(all_speakers))
    print(f"Stats: Found {len(sorted_speakers)} unique speaker(s) and {total_segments} segment(s).")

    # we add 'None' in the sorted speakers for segment with no speaker
    if speaker_None : sorted_speakers.append(None)

    # Constante pour représenter un speaker inconnu/null dans le TRS
    NONE_SPEAKER_ID = "unknown"

    # 4. Generate TRS XML Structure
    audio_filename = os.path.splitext(os.path.basename(input_path))[0] + ".wav"

    trans = ET.Element("Trans", audio_filename=audio_filename)

    topics = ET.SubElement(trans, "Topics")
    ET.SubElement(topics, "Topic", id="to0", desc="")

    # Speakers — on remplace None par NONE_SPEAKER_ID pour l'attribut XML
    speakers_tag = ET.SubElement(trans, "Speakers")
    for spk in sorted_speakers:
        spk_id = spk if spk is not None else NONE_SPEAKER_ID
        ET.SubElement(speakers_tag, "Speaker", id=spk_id, name=spk_id)

    episode = ET.SubElement(trans, "Episode")
    section_end = segments[-1]["end"] if segments else 0.0
    section = ET.SubElement(episode, "Section", type="report", startTime="0.0",
                             endTime=str(section_end), topic="to0")

    for seg in segments:
        seg_speakers = []
        for w in seg.get("words", []):
            spk = w.get("speaker")
            # CORRECTION : on inclut explicitement None (remplacé par NONE_SPEAKER_ID)
            spk_id = spk if spk is not None else NONE_SPEAKER_ID
            if spk_id not in seg_speakers:
                seg_speakers.append(spk_id)

        # Fallback sur le speaker du segment (qui peut aussi être None)
        if not seg_speakers:
            seg_spk = seg.get("speaker")
            seg_speakers = [seg_spk if seg_spk is not None else NONE_SPEAKER_ID]

        turn = ET.SubElement(section, "Turn",
                             speaker=" ".join(seg_speakers),
                             startTime=str(seg["start"]),
                             endTime=str(seg["end"]))

        if len(seg_speakers) > 1:
            current_speaker = None
            speaker_to_idx = {spk: i for i, spk in enumerate(seg_speakers)}

            for word in seg.get("words", []):
                w_spk_raw = word.get("speaker")

                w_spk = w_spk_raw if w_spk_raw is not None else NONE_SPEAKER_ID

                if w_spk not in speaker_to_idx:
                    # Fallback sur le speaker du segment si toujours introuvable
                    seg_spk_raw = seg.get("speaker")
                    w_spk = seg_spk_raw if seg_spk_raw is not None else NONE_SPEAKER_ID
                    print(f"Speaker introuvable, fallback vers : {w_spk}")

                if w_spk != current_speaker:
                    current_speaker = w_spk
                    ET.SubElement(turn, "Sync", time=str(word["start"]))
                    ET.SubElement(turn, "Who", nb=str(speaker_to_idx[w_spk]))

                last_child = turn[-1]
                last_child.tail = (last_child.tail or "") + word["word"] + " "
        else:
            sync = ET.SubElement(turn, "Sync", time=str(seg["start"]))
            sync.tail = seg.get("text", "").strip()

    # 5. Save Output
    output_path = os.path.splitext(input_path)[0] + ".trs"
    
    # Convert to pretty XML string
    xml_str = ET.tostring(trans, encoding='utf-8')
    parsed_xml = minidom.parseString(xml_str)
    
    # Add Doctype manually as ElementTree is limited here
    header = '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE Trans SYSTEM "trans-14.dtd">\n'
    pretty_xml = header + parsed_xml.documentElement.toprettyxml(indent="    ")

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(pretty_xml)
        print(f"Success: File generated at '{output_path}'")
    except Exception as e:
        print(f"Error writing output file: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print_usage()
    else:
        generate_trs(sys.argv[1])
