import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf


# Preprocessing the datasets.
# We need to read the audio files as arrays
def speech_file_to_array_fn(audio_file):
    resampler = torchaudio.transforms.Resample(48_000, 16_000)
    speech_array, sampling_rate = torchaudio.load(audio_file)
    processed_audio = resampler(speech_array).squeeze().numpy()
    return processed_audio

def recognizer():
    processor = Wav2Vec2Processor.from_pretrained("mohammed/wav2vec2-large-xlsr-arabic")
    model = Wav2Vec2ForCTC.from_pretrained("mohammed/wav2vec2-large-xlsr-arabic")

    test_file = "data/test1.wav"

    test_file = speech_file_to_array_fn(test_file)
    inputs = processor(test_file, sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)

    print("The predicted sentence is: ", processor.batch_decode(predicted_ids))
if __name__ == "__main__":
    recognizer()
        
