import openai

API_KEY = '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'
model_id = 'whisper-1'

media_file_path = r'C:\Users\WIN\Desktop\ROBOTATAL\raspberrypi\1.wav'

# Open the WAV file
with open(media_file_path, 'rb') as media_file:
    # Transcribe the audio
    response = openai.Audio.transcribe(
        api_key=API_KEY,
        model=model_id,
        file=media_file
    )

# Print transcription
print(response['text'])
