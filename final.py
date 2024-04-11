
import os
import sounddevice as sd
import soundfile as sf
import tempfile
from openai import OpenAI
import numpy as np
import librosa
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

# Set your OpenAI API key here
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Parameters for wake word detection
WAKE_WORD = "Computer"
WAKE_WORD_THRESHOLD = 1  # Adjust this threshold as needed
FRAME_SIZE = 2048
HOP_LENGTH = 512
DURATION = 10  # Adjust the duration of audio recording


def detect_wake_word(audio_data):
    mfccs = librosa.feature.mfcc(y=np.squeeze(audio_data), sr=44100, n_mfcc=13)
    delta_mfccs = librosa.feature.delta(mfccs)
    combined_features = np.vstack((mfccs, delta_mfccs))

    # Spectral flux
    spectral_flux = np.sum(np.diff(combined_features) > 0, axis=0)

    # Peak picking
    struct = generate_binary_structure(1, 1)
    neighborhood = generate_binary_structure(1, 5)  # Adjusted neighborhood size
    local_max = maximum_filter(spectral_flux, footprint=neighborhood) == spectral_flux
    detected_peaks = np.logical_and(spectral_flux > 1, local_max)
    detected_peaks = binary_erosion(detected_peaks, structure=struct)

    # Check if peaks are detected
    if np.any(detected_peaks):
        # Find the maximum peak
        max_peak = np.max(spectral_flux[detected_peaks])

        if max_peak > WAKE_WORD_THRESHOLD:
            return True

    return False


def record_audio():
    print("Listening for wake word...")
    while True:
        audio_data = sd.rec(44100, samplerate=44100, channels=1, dtype='float32')
        sd.wait()
        if detect_wake_word(audio_data):
            print("Wake word detected. Recording started.")
            break

    frames = []

    # Capture audio for a fixed duration
    print(f"Recording for {DURATION} seconds...")
    audio_data = sd.rec(int(44100 * DURATION), samplerate=44100, channels=1, dtype='float32')
    sd.wait()
    frames.append(audio_data)

    audio_data = np.concatenate(frames)
    return audio_data


def transcribe_audio(audio_data):
    client = OpenAI(api_key=OPENAI_API_KEY)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio_file:
        tmp_audio_file_path = tmp_audio_file.name
        sf.write(tmp_audio_file_path, audio_data, 44100, subtype='PCM_16')
        with open(tmp_audio_file_path, 'rb') as file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=file,
                response_format="json"
            )
    os.remove(tmp_audio_file_path)
    return response.text


def call_openai_api(input_text):
    client = OpenAI(api_key=OPENAI_API_KEY)
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Use GPT-3.5 "turbo" model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_text}
        ]
    )
    return completion.choices[0].message.content


def text_to_speech(text, voice="echo"):
    client = OpenAI(api_key=OPENAI_API_KEY)
    valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    if voice not in valid_voices:
        print(f"Invalid voice option: {voice}. Using 'echo' instead.")
        voice = "echo"

    speech_file_path = f"{os.path.splitext(os.path.basename(__file__))[0]}.wav"
    response = client.audio.speech.create(
        model="tts-1-hd",
        voice=voice,
        input=text
    )
    response.stream_to_file(speech_file_path)
    return speech_file_path


def play_audio(file_path):
    data, samplerate = sf.read(file_path)
    sd.play(data, samplerate)
    sd.wait()


def main():
    while True:
        print("Starting audio recording...")
        audio_data = record_audio()
        print("Audio recorded. Transcribing...")
        transcription = transcribe_audio(audio_data)
        print("Transcription:", transcription)

        print("Sending transcription to ChatGPT...")
        gpt_response = call_openai_api(transcription)
        print("ChatGPT response:", gpt_response)

        print("Converting ChatGPT response to speech...")
        speech_file_path = text_to_speech(gpt_response)
        print(f"Speech saved to: {speech_file_path}")

        print("Playing the audio...")
        play_audio(speech_file_path)


if __name__ == "__main__":
    main()
