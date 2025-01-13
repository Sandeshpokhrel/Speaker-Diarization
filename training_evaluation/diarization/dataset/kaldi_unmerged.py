import os
import soundfile as sf

def generate_wav_scp(root_dir, output_file):
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, 'w') as f:
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.flac') or file.endswith('.wav'):
                    utt_id = file.split('.')[0]
                    file_path = os.path.join(subdir, file)
                    file_path = file_path.replace("\\", "/")
                    f.write(f'{utt_id} {file_path}\n')


def generate_utt2spk(root_dir, output_file):
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    speaker_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    with open(output_file, 'w') as f:
        for spk_id in speaker_dirs:
            spk_dir = os.path.join(root_dir, spk_id)
            for file in os.listdir(spk_dir):
                if file.endswith('.flac') or file.endswith('.wav'):
                    utt_id = file.split('.')[0]
                    f.write(f'{utt_id} {spk_id}\n')


def compute_duration(file_path):
    try:
        with sf.SoundFile(file_path) as audio_file:
            duration = len(audio_file) / audio_file.samplerate
            return duration
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return 0


def generate_reco2dur(wav_scp_file, reco2dur_file):
    """ Generate the reco2dur file from wav.scp. """
    with open(wav_scp_file, 'r') as wav_scp, open(reco2dur_file, 'w') as reco2dur:
        for line in wav_scp:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                print(f"Skipping malformed line: {line.strip()}")
                continue

            utt_id, file_path = parts
            duration = compute_duration(file_path)
            reco2dur.write(f'{utt_id} {duration:.3f}\n')



if __name__ == "__main__":
    audio_path = "va_filtered_audio/audio"
    wav_scp_path = "va_filtered_audio/details/wav.scp"
    utt2spk_path = "va_filtered_audio/details/utt2spk"
    reco2dur_path = "va_filtered_audio/details/reco2dur"

    generate_wav_scp(audio_path, wav_scp_path)
    generate_utt2spk(audio_path, utt2spk_path)
    generate_reco2dur(wav_scp_path, reco2dur_path)
