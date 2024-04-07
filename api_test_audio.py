import openai
import sounddevice as sd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("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")


# Function to call OpenAI API and get response
def call_openai_api(input_text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_text}
        ]
    )
    return response['choices'][0]['message']['content']


# Function to record audio from microphone
def record_audio():
    duration = 5  # Record for 5 seconds
    fs = 16000  # Sample rate
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    print("Recording audio...")
    sd.wait()  # Wait until recording is finished
    print("Recording completed.")
    return myrecording


# Function to transcribe audio using rev.ai
def transcribe_audio(audio, rev_ai_api_key):
    # Implement rev.ai transcription logic here
    # For now, let's just return the audio as text
    return "".join([str(x) for x in audio])


# Main function
def main():
    while True:
        # Record audio from microphone
        audio = record_audio()

        # Transcribe audio using rev.ai
        rev_ai_api_key = os.getenv("02LR7pfNZPqD8x7VXnK-xgY4qqzJDjId1qXS-dpY0OC90ijDX-zU_5sDjX6xdLgCyHuO1qN3HVXkgaTgq_nxpOtO-BiR8")
        transcript = transcribe_audio(audio, rev_ai_api_key)

        # Generate response using OpenAI API
        response = call_openai_api(transcript)
        print("AI: ", response)

        # Ask user if they want to continue
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break


if __name__ == "__main__":
    main()
