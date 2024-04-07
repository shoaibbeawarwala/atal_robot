import openai
import sounddevice as sd
import soundfile as sf
import tempfile
import threading

API_KEY = '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'
model_id = 'whisper-1'


def record_audio():
    print("Recording... Press Enter to stop recording.")
    audio_data = sd.rec(frames=44100 * 5, samplerate=44100, channels=1, dtype='int16')
    return audio_data


def save_audio(audio_data, file_path):
    sf.write(file_path, audio_data, 44100)


def transcribe_audio(file_path):
    with open(file_path, 'rb') as audio_file:
        response = openai.Audio.transcribe(
            api_key=API_KEY,
            model=model_id,
            file=audio_file
        )
    return response['text']


def main():
    input("Press Enter to start recording...")
    recording_thread = threading.Thread(target=record_audio)
    recording_thread.start()
    input("Press Enter to stop recording...")
    sd.stop()

    # Save the recorded audio to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio_file:
        tmp_audio_file_path = tmp_audio_file.name
        save_audio(sd.rec(frames=44100 * 5, samplerate=44100, channels=1, dtype='int16'), tmp_audio_file_path)
        print("Audio saved as", tmp_audio_file_path)

        # Transcribe the audio
        transcription = transcribe_audio(tmp_audio_file_path)
        print("Transcription:", transcription)


if __name__ == "__main__":
    main()
