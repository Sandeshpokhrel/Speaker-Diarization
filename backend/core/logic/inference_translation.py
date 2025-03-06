from pydub import AudioSegment
import os
import speech_recognition as sr
import glob


# Split audio into 10-second chunks
def split_audio(input_file):
    audio = AudioSegment.from_file(input_file, format="flac").set_channels(1)
    chunk_length = 10 * 1000  # 10 seconds in milliseconds
    chunks = []
    
    for i in range(0, len(audio), chunk_length):
        chunk = audio[i:i + chunk_length]
        chunk_name = f"{len(chunks)+1}.flac"
        chunk.export(chunk_name, format="flac")
        chunks.append(chunk_name)
        # print(f"Created chunk: {chunk_name}")
    
    return chunks

# Recognize text from chunks
def recognize_chunks(chunk_files):
    r = sr.Recognizer()
    texts = []
    
    for chunk in chunk_files:
        try:
            with sr.AudioFile(chunk) as source:
                audio_data = r.record(source)
                text = r.recognize_google(audio_data, language='ne-NP')
                texts.append(text)
                # print(f"Processed {chunk}: {text[:50]}...")
        except Exception as e:
            texts.append(" ") # Append empty string if error
            # print(f"Error processing {chunk}: {e}")
    
    return texts



def np_speech_text_translation(user_input_path):

    wav_files = glob.glob(f"{user_input_path}/*.flac")
    rttm_files = glob.glob(f"{user_input_path}/*.rttm")
    audio_path = wav_files[0]
    rttm_path = rttm_files[0]

    audio = AudioSegment.from_file(audio_path, format="flac")

    # Read RTTM file and extract segments
    segments = []
    with open(rttm_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 5:
                start_time = float(parts[3])
                duration = float(parts[4])
                end_time = start_time + duration
                segments.append((start_time, end_time))

    # Process and save each segment
    for idx, (start, end) in enumerate(segments, start=1):
        segment = audio[int(start * 1000):int(end * 1000)]

        os.makedirs(f"{user_input_path}/separate", exist_ok=True)
        
        filename = f"{user_input_path}/separate/{idx}.flac"
        segment.export(filename, format="flac")

    # load audio files
    directory = f"{user_input_path}/separate"
    audio_files = [f for f in os.listdir(directory) if f.endswith(".flac")]

    text_list = []
    for audio_file in audio_files:
        chunk_files = split_audio(f"{directory}/{audio_file}")
        results = recognize_chunks(chunk_files)
        final_text = " ".join(results)
        text_list.append(final_text)

        for chunk in chunk_files:
            os.remove(chunk)
    
    os.remove(f"{user_input_path}/separate/{idx}.flac") if os.path.exists(f"{user_input_path}separate/{idx}.flac") else None

    return text_list