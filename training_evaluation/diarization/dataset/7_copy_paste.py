import shutil
import os

def copy_utt2spk_file(source_path, destination_path):
    try:
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"The source file '{source_path}' does not exist.")
        
        dest_dir = os.path.dirname(destination_path)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        shutil.copy(source_path, destination_path)
        return f"File 'utt2spk' successfully copied to '{destination_path}'."
    
    except Exception as e:
        return f"An error occurred: {e}"


type_dataset = 'train'
source = "dataset/va_arranged_audio/details/utt2spk"
destination = f"dataset/merged_audio2/{type_dataset}/details/utt2spk"
print(copy_utt2spk_file(source, destination))

