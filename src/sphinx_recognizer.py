from textblob import TextBlob
import pyaudio as py
import logging
import speech_recognition as sr 
import requests
from pydub import AudioSegment
from pydub.silence import split_on_silence


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Creating a recognizer
r = sr.Recognizer()

pa = py.PyAudio()

def recognize_english(source):
    audio = r.listen(source)
    text = ""
    logger.info('end of speech, translating >>>>>')
    try:
        text = r.recognize_google(audio)
    except sr.UnknownValueError:
        print("Unable to catch the speech !")
    except sr.RequestError as e:
        print("Couldn't request the result from google service {0}".format(e))
        
    if len(text) < 1:
        return 'No Recognition Found'
    return text


def analyze_text(text, language):
    if language == "da":
        text = text.translate(from_lang="da", to='en')
        
    textblob_text = TextBlob(text)
    if textblob_text.sentiment.polarity < 0:
        sentiment = "Negative"
    elif textblob_text.sentiment.polarity > 0:
        sentiment = "Positive"
    else:
        sentiment = "Neutral"
    return textblob_text.sentiment, sentiment



# Define a function to normalize a chunk to a target amplitude.
def match_target_amplitude(aChunk, target_dBFS):
    ''' Normalize given audio chunk '''
    change_in_dBFS = target_dBFS - aChunk.dBFS
    return aChunk.apply_gain(change_in_dBFS)

# Load your audio.
song = AudioSegment.from_wav("mine.wav")
print(f'The number of channels is: {song.channels}')
print(f'The length of the audio file is {len(song)}')
print(f'Duration in seconds is: {audio_file.duration_seconds}')

logger.info('Calculating total silence time >>>>>>')

silence = silence.detect_silence(audio_file, min_silence_len=1000, silence_thresh=-40)
#convert to sec
silence = [((start/1000),(stop/1000)) for start,stop in silence] 

# Split track where the silence is 2 seconds or more and get chunks using 
# the imported function.
chunks = split_on_silence(
    # Use the loaded audio.
    song,
    # Specify that a silent chunk must be at least 2 seconds or 2000 ms long.
    min_silence_len = 1000,
    # Consider a chunk silent if it's quieter than -16 dBFS.
    # (You may want to adjust this parameter.)
    silence_thresh = -40,
    keep_silence=500
)

print(chunks)
# Process each chunk with your parameters
for i, chunk in enumerate(chunks):
    # Create a silence chunk that's 0.5 seconds (or 500 ms) long for padding.
    silence_chunk = AudioSegment.silent(duration=500)

    # Add the padding chunk to beginning and end of the entire chunk.
    audio_chunk = silence_chunk + chunk + silence_chunk

    # Normalize the entire chunk.
    normalized_chunk = match_target_amplitude(audio_chunk, -20.0)
    print("Exporting chunk{0}.wav.".format(i))
    normalized_chunk.export(
        ".//chunk{0}.wav".format(i),
        bitrate = "192k",
        format = "wav"
    )
    # Export the audio chunk with new bitrate.
    print("Translating chunk{0}.wav.".format(i))
    with sr.AudioFile("chunk"+str(i)+".wav") as source:
        print(source)
        audio = r.listen(source)
    transcribed_text = r.recognize_sphinx(audio)
    sentiment = analyze_text(transcribed_text, "en")
    print(f'Text is: {transcribed_text}')
    logger.info('Showing sentiment result >>>>>>')
    print(f'The sentiment result is: {sentiment}')