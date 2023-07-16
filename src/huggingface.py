import torch
import torchaudio
import pyaudio as py
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
import speech_recognition as sr


# Preprocessing the datasets.
# We need to read the audio files as arrays
def speech_file_to_array_fn(audio_file):
    resampler = torchaudio.transforms.Resample(48_000, 16_000)
    speech_array, sampling_rate = torchaudio.load(audio_file)
    processed_audio = resampler(speech_array).squeeze().numpy()
    return processed_audio

def recognize(test_file):
    processor = Wav2Vec2Processor.from_pretrained("mohammed/wav2vec2-large-xlsr-arabic")
    model = Wav2Vec2ForCTC.from_pretrained("mohammed/wav2vec2-large-xlsr-arabic")

    # test_file = "data/test1.wav"

    test_file = speech_file_to_array_fn(test_file)
    inputs = processor(test_file, sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_speech = processor.batch_decode(predicted_ids)
    print("The predicted sentence is: ", predicted_speech[0][::-1])

if __name__ == "__main__":
    test_file = "data/test1.wav"
    # recognize(test_file)
    r = sr.Recognizer()
    pa = py.PyAudio()
    microphone_index = pa.get_default_input_device_info()['index']

    #Defining the microphone
    mic = sr.Microphone(device_index=microphone_index, chunk_size=4)
    with mic as source:
        counter = 0
        while 1:
            print("say something >>>>>>")
            file_name = "test_file" + str(counter) + ".wav"
            audio = r.listen(source, 50, 5)
            with open(file_name, "wb") as f:
                f.write(audio.get_wav_data())
            recognize(file_name)

        
