# import speech_recognition as sr
# from io import BytesIO
# from pydub import AudioSegment
# import soundfile as sf
# import tempfile
# import torch
# from transformers import AutoProcessor, AutoModelForCTC
# from scipy.io import wavfile
# import numpy as np
# import os
# import librosa
# # Define the model path
# # model_path = "core/logic/diarization/model/wav2vec2/wav2vec2_model.pth"  # Adjust the path as necessary

# # # Load the model architecture
# # processor = AutoProcessor.from_pretrained("anish-shilpakar/wav2vec2-nepali")
# # model = AutoModelForCTC.from_pretrained("anish-shilpakar/wav2vec2-nepali")

# # # Load the saved state dictionary
# # model.load_state_dict(torch.load(model_path))

# # # Set the model to evaluation mode
# # model.eval()

# # # Load the tokenizer (if needed)
# # tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")  # or your specific tokenizer
# recognizer = sr.Recognizer()

# def recognize_speech(audio_data):
#     try:
#         # Convert the audio to PCM WAV in-memory
#         audio_segment = AudioSegment.from_file(audio_data)
        
#         # Ensure audio is of sufficient quality
#         if len(audio_segment) < 500:  # Minimum audio length in milliseconds
#             print(f"Audio segment too short: {len(audio_segment)} ms")
#             return None
        
#         # Normalize and preprocess audio
#         audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
        
#         # Increase volume if too quiet
#         audio_segment = audio_segment + 6  # Increase volume by 6 dB
        
#         wav_io = BytesIO()
#         audio_segment.export(wav_io, format="wav")
#         wav_io.seek(0)

#         # Use BytesIO object as the audio source
#         with sr.AudioFile(wav_io) as source:
#             # Adjust for ambient noise with a longer duration
#             recognizer.adjust_for_ambient_noise(source, duration=1)
#             audio = recognizer.record(source)

#         # Attempt recognition with multiple fallback methods
#         try:
#             # Try Google Speech Transcription first
#             text = recognizer.recognize_google(audio, language="ne-IN")  # Nepali language
#         except sr.UnknownValueError:
#             try:
#                 # Fallback to Sphinx (offline recognition)
#                 text = recognizer.recognize_sphinx(audio)
#             except Exception:
#                 print(f"Speech transcription failed for audio segment")
#                 return None

#         # Check if recognized text is meaningful
#         if text and len(text.strip()) > 0:
#             return text
#         else:
#             print("Recognized text is empty")
#             return None

#     except Exception as e:
#         print(f"Error in speech recognition: {e}")
#         return None

# def transcribe_audio():
#     directory = "core/logic/diarization/user_input/user_0"

#     # Find the .flac file
#     flac_file = next((f for f in os.listdir(directory) if f.endswith(".flac")), None)
#     rttm_file = next((f for f in os.listdir(directory) if f.endswith(".rttm")), None)
    
#     if flac_file and rttm_file:
#         flac_path = os.path.join(directory, flac_file)
#         rttm_path = os.path.join(directory, rttm_file)
        
#         rttm_list = []
#         with open(rttm_path, 'r') as file:
#             for line in file:
#                 if line.strip():
#                     parts = line.split()
#                     speaker = parts[7]
#                     start_time = float(parts[3])
#                     duration = float(parts[4])
#                     end_time = start_time + duration
#                     rttm_list.append([speaker, start_time, end_time])
        
#         transcription = []
#         for speaker, start_time, end_time in rttm_list:
#             # Load audio segment
#             y, sr = librosa.load(flac_path, sr=16000, offset=start_time, duration=end_time-start_time)
            
#             # Normalize audio safely
#             if np.max(np.abs(y)) > 0:
#                 sound_data = y / np.max(np.abs(y))
#             else:
#                 sound_data = y

#             # Create temporary audio file
#             with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
#                 sf.write(temp_audio.name, sound_data, sr)
#                 temp_audio_path = temp_audio.name
            
#             # Attempt speech recognition
#             text = recognize_speech(temp_audio_path)
            
#             # Remove temporary file
#             os.unlink(temp_audio_path)
            
#             # Add text to transcription list
#             transcription.append([speaker, start_time, end_time, text])
        
#         return transcription
#     else:
#         print("No .flac file found in the directory.")
#         return []
