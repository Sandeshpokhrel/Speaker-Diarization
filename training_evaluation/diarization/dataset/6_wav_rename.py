type_dataset = 'train'
input_file = f'dataset/merged_audio/{type_dataset}/details/wav.scp'
output_file = f'dataset/merged_audio/{type_dataset}/details/n_wav.scp'

new_base_dir = f'/kaggle/input/dataseteend/merged_audio/{type_dataset}/audio/'

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        utt_id, old_path = line.strip().split(maxsplit=1)
        new_path = old_path.replace(f'dataset/merged_audio/{type_dataset}/audio/', new_base_dir)
        outfile.write(f'{utt_id} {new_path}\n')

print(f"Updated wav.scp file saved to {output_file}")
