### Install ffmpeg - sudo apt install ffmpeg
### pip install -r requirements.txt

### step0: run arrange_audio.py
    Arrange librispeech audio from 345/4553/345-4553-0001.flac to 345/345-4553-0001.flac
    (for all audio)

### step1: run VAD_audio.py 
    Process each audio file in the input directory to remove non-voice segments. 
    The resulting audio will only contain sections with detected voice activity.

### step2: run kaldi_unmerged.py
    generating voice activityy filtered audio metadata in Kaldi format
    (before merging them to generate 2-3-4 speaker audio)

### step3: run train_valid_test.py
    Split unprocessed audio data into train, validation, and test directories.
    Ensuring that audio from the same person does not appear in multiple sets.

### step4: run improved_merged.py
    merges multiple audio files into single multi-speaker audio in respective train, validation and test directories.

### step5: run data_visual.py
    visualize each train/validation/test dataset
    (how they are merged)

### step6: run wav_rename.py
    rename wav.scp for the location of audio data
    (necessary if we train in Kaggle or change the location)