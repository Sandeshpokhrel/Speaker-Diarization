import os
import soundfile as sf

def generate_utt2spk(root_dir, output_file):
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, 'w') as f:
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.flac') or file.endswith('.wav') or file.endswith('.m4a') or file.endswith('.mp3'):
                    # make utt_id from subdir separeted by '/'
                    utt_id = subdir.split('/')[-1]
                    file_path = os.path.join(subdir, file)
                    file_path = file_path.replace("\\", "/")

                    parent_dir, file_name = os.path.split(file_path)
                    utterance = os.path.join(os.path.basename(parent_dir), os.path.splitext(file_name)[0])

                    f.write(f'{utterance} {utt_id}\n')

if __name__ == "__main__":
    audio_path = "dataset/va_arranged_audio/audio"
    utt2spk_path = "dataset/va_arranged_audio/details/utt2spk"

    generate_utt2spk(audio_path, utt2spk_path)