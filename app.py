import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
from deepgram import DeepgramClient, PrerecordedOptions, FileSource

load_dotenv()

DG_API_KEY = os.getenv("DG_API_KEY")
OA_API_KEY = os.getenv("OPENAI_API_KEY")

def transcribe_file(audio_bytes):
    try:
        # Create a Deepgram client using the API key
        deepgram = DeepgramClient(DG_API_KEY)

        # Create payload with the audio bytes
        payload: FileSource = {"buffer": audio_bytes}

        # Configure Deepgram options for audio analysis
        options = PrerecordedOptions(
            model="nova-2",
            language="ja",
            smart_format=True,
            punctuate=True,
            diarize=True,
        )

        # Call the transcription.prerecorded method with the payload and options
        response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)
        return response
    except Exception as e:
        print(f"Exception: {e}")
        return None
    
def summary(text):
    # 文字起こしされたテキストを議事録化
    prompt = f"次のテキストを要点をトピックごとにまとめて、議事録を作成してください：\n{text}"
    # chatGPTに質問する
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    summary = completion.choices[0].message.content.strip().split("\n")
    return summary

def main():
    st.title('オーディオファイルをアップロードしてください')
    uploaded_file = st.file_uploader("ファイルを選択してください", type=['mp3', 'wav', 'm4a'])

    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        response = transcribe_file(audio_bytes)

        if response is not None:
            # Display the Deepgram response
            transcription = response.to_dict()
            transcript_text = transcription["results"]["channels"][0]["alternatives"][0]["transcript"]
            st.write("文字起こし結果：")
            st.write(transcript_text)

            # Generate summary using OpenAI API
            summarized_text = summary(transcript_text)
            st.write("議事録：")
            for item in summarized_text:
                st.write(f"- {item}")
        else:
            st.error('エラーが発生しました。')

if __name__ == "__main__":
    main()
