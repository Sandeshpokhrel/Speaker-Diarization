# Install ffmpeg - sudo apt install ffmpeg
# pip install -r requirements.txt


step1: run VAD_audio.py 
    Process each audio file in the input directory to remove non-voice segments. 
    The resulting audio will only contain sections with detected voice activity.

step2: run kaldi_unmerged.py
    generating voice activityy filtered audio metadata in Kaldi format
    (before merging them to generate 2-3-4 speaker audio)

step3: run train_valid_test.py
    Split unprocessed audio data into train, validation, and test directories.
    Ensuring that audio from the same person does not appear in multiple sets.

step4: run merge_multiple.py (main task is to modify this file)
    merges multiple audio files into single multi-speaker audio in respective train, validation and test directories.

step5: run kaldi_merged.py
    create kaldi files with metada of merged audio.
