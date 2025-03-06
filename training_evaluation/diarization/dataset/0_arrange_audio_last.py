import shutil
import os

# Define source folders
source_folders = [
    "dataset/arranged_audio_LibriSpeech",
    "dataset/arranged_audio_VoxCeleb",
    "dataset/arranged_audio_nphi"
]

# Define destination folder
destination_folder = "dataset/arranged_audio"

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Copy subdirectories from each source folder to destination folder
for folder in source_folders:
    if os.path.exists(folder):
        for sub_dir in os.listdir(folder):
            src_path = os.path.join(folder, sub_dir)
            dest_path = os.path.join(destination_folder, sub_dir)

            # Ensure we copy only directories
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
                print(f"Copied folder: {src_path} -> {dest_path}")

print("All folders copied successfully.")
