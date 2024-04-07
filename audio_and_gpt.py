import openai


# Set your OpenAI API key here
api_key = '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'


# Function to call OpenAI API and get response
def call_openai_api(input_text):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use GPT-3.5 "turbo" model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_text}
        ]

    )
    return response['choices'][0]['message']['content']


# Function to transcribe audio file
def transcribe_audio(audio_file_path):
    openai.api_key = api_key
    response = openai.Audio.transcribe(
        api_key=api_key,
        model="whisper-1",  # Specify the model for audio transcription
        file=open(audio_file_path, 'rb')
    )
    return response['text']


# Main function
def main():
    # Transcribe audio file
    audio_file_path = r'C:\Users\WIN\Desktop\ROBOTATAL\raspberrypi\2.wav'  # Provide the path to your audio file
    transcription = transcribe_audio(audio_file_path)

    # Call OpenAI API with transcribed text
    response = call_openai_api(transcription)
    print("AI: ", response)


if __name__ == "__main__":
    main()
