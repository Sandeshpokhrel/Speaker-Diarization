import os
import random
from pydub import AudioSegment


def merge_audio_files(audio_files, output_path, file_format):
    combined = AudioSegment.empty()
    for file in audio_files:
        print(f"Processing file: {file}")
        audio = AudioSegment.from_file(file, format=file_format)
        if audio.channels > 1:
            audio = audio.set_channels(1)
        combined += audio
    combined.export(output_path, format=file_format)


def prepare_dataset(unmerged_dir, merged_dir, output_count, num_spk, file_format="flac"):
    audio_dir = os.path.join(unmerged_dir)
    merged_audio_dir = os.path.join(merged_dir, "audio")
    details_dir = os.path.join(merged_dir, "details")
    merge_file_path = os.path.join(details_dir, "merge")
    wav_scp_file_path = os.path.join(details_dir, "wav.scp")
    
    os.makedirs(merged_audio_dir, exist_ok=True)
    os.makedirs(details_dir, exist_ok=True)

    all_files = [
        os.path.join(subdir, file)
        for subdir, _, files in os.walk(audio_dir)
        for file in files if file.endswith(("flac", "wav"))
    ]
    
    if len(all_files) < num_spk:
        raise ValueError("Not enough audio files to merge.")

    merged_files = []
    merge_lines = []
    wav_scp_lines = []

    for i in range(output_count):
        selected_files = random.sample(all_files, num_spk)
        output_file = os.path.join(merged_audio_dir, f"{i:08d}.{file_format}")
        merge_audio_files(selected_files, output_file, file_format)
        
        merged_files.append(output_file)
        merge_file_name = os.path.basename(output_file).replace(f".{file_format}", "")
        merge_lines.append(f"{merge_file_name} {' '.join(os.path.basename(f).replace(f'.{file_format}', '') for f in selected_files)}")
        wav_scp_lines.append(f"{merge_file_name} {output_file}")

        print("Done:", i, "Audio file")

    with open(merge_file_path, 'w') as f:
        f.write("\n".join(merge_lines) + '\n')

    with open(wav_scp_file_path, 'w') as f:
        f.write("\n".join(line.replace("\\", "/") for line in wav_scp_lines) + '\n')



if __name__ == "__main__":
    # modify which_dataset = (train,validation,test), output_count, num_spk
    which_dataset = "train"
    output_count = 15
    num_spk = 3

    unmerged_dir = f"va_filtered_audio/audio/{which_dataset}"
    merged_dir = f"merged_audio/{which_dataset}"
    prepare_dataset(unmerged_dir, merged_dir, output_count, num_spk)