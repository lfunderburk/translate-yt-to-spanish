import openai
import os 
from dotenv import load_dotenv
import unidecode

def transcribe_audio(api_key, audio_file):
    """Transcribe the audio file using Whisper ASR."""
    with open(audio_file, "rb") as f:
        response = openai.Audio.transcribe("whisper-1", f, api_key=api_key)
    return response["text"]

def translate_text(text):
    """Translate English text to Spanish using GPT."""
    system  = "You are a helpful assistant and you translate English text to Spanish text.\
                Please discard tildes and other special characters from your translation. \
                You have in your language the following special words: JupySQL \
                (sometimes incorrectly spelled as GPSQL or gpsql); \
                Ploomber (incorrectly spelled as Plumber or Plumer)\
                LLMs (el-ell-ems). \
                You are also adept at separating information into\
                paragraphs and sentences."
    user = text
    completion = openai.ChatCompletion.create( 
                    model="gpt-3.5-turbo", 
                    messages=[ 
                        {"role": "system", "content": system}, 
                        {"role": "user", "content":user} 
                    ] 
                    ) 

    return completion.choices[0].message['content']

if __name__ == "__main__":
    load_dotenv(".env")

    # Initialize OpenAI API
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Transcribe the MP3 file
    audio_file = "src/queries.m4a"
    transcription = transcribe_audio(openai.api_key, audio_file)
    clean_transcription = unidecode.unidecode(transcription)
    print("Original Transcription: ", transcription)

    # Translate the transcription to Spanish
    spanish_transcription = translate_text(transcription)
    print("Translated Transcription: ", spanish_transcription)