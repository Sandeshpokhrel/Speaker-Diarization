import os
import shutil

def move_flac_files(base_dir):
    for id_dir in os.listdir(base_dir):
        id_path = os.path.join(base_dir, id_dir)

        if os.path.isdir(id_path):
            for subdir in os.listdir(id_path):
                subdir_path = os.path.join(id_path, subdir)

                if os.path.isdir(subdir_path):
                    for file_name in os.listdir(subdir_path):
                        if file_name.endswith(".flac"):
                            src_file = os.path.join(subdir_path, file_name)
                            dest_file = os.path.join(id_path, file_name)
                            shutil.move(src_file, dest_file)
                            print(f"Moved: {src_file} -> {dest_file}")
                    
                    try:
                        shutil.rmtree(subdir_path)
                        print(f"Removed directory: {subdir_path}")
                    except Exception as e:
                        print(f"Failed to remove directory {subdir_path}: {e}")

if __name__ == "__main__":
    base_directory = "train-clean-360" #LibriSpeech dataset
    move_flac_files(base_directory)
