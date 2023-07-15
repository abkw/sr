from textblob import TextBlob
import pyaudio as py
import logging
import speech_recognition as sr 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#Creating a recognizer
r = sr.Recognizer()

#The list of the default microphone devices
pa = py.PyAudio()
microphone_index = pa.get_default_input_device_info()['index']

#Defining the microphone
mic = sr.Microphone(device_index=microphone_index)

#Listening through the mic (recognizng english language)
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

#Listening through the mic (recognizng danish language)
        
def recognize_danish():
    audio = r.listen(source)
    text = ""
    logger.info('end of speech, translating >>>>>')
    try:
        text = r.recognize_google(audio, language = "da-DK")
    except sr.UnknownValueError:
        print("Unable to catch the speech !")
    except sr.RequestError as e:
        print("Couldn't request the result from google service {0}".format(e))
        
    if len(text) < 1:
        return 'No Recognition Found'
    return text

#Analyzing the text sentiment
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


if __name__ == '__main__':
    print("say me")
    r = sr.Recognizer()
    with mic as source:
        while 1:
            transcribed_text = recognize_english(source)
            sentiment = analyze_text(transcribed_text, "en")
            print(f'Text is: {transcribed_text}')
            logger.info('Showing sentiment result >>>>>>')
            print(f'The sentiment result is: {sentiment}')