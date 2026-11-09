import os
import base64
import requests
import json
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama.llms import OllamaLLM


load_dotenv()
openai_api_key= os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OpenAI API 키가 없습니다. 한번 더 확인 부탁드립니다.")

def generate_text_summaries(texts, tables, summarize_texts=False):
    """
    텍스트 및 표 데이터를 요약하여 검색에 활용할 수 있는 요약본을 생성합니다.
    texts: 텍스트 리스트
    tables: 표 리스트
    summarize_texts: 텍스트 요약을 활성화할지 여부를 결정하는 불리언 값
    """

    # Prompt 한국어 버전
    prompt_text_kor = """당신은 표와 텍스트를 요약하여 검색에 활용할 수 있도록 돕는 도우미 입니다.
    이 요약본들은 임베딩되어 원본 텍스트나 표 요소를 검색하는데 사용될 것입니다.
    주어진 표나 텍스트의 내용을 검색에 최적화된 간결한 요약으로 작성해주세요.
    요약할 표 또는 텍스트: {element}"""

    # Prompt 영어 버전
    prompt_text_eng = """You are an assistant that summarizes tables and text so they can be used for retrieval.
    These summaries will be embedded and used to retrieve the original text or table elements.
    Write a concise summary of the given table or text, optimized for retrieval.
    Respond in Korean.
    Table or text to summarize: {element}"""


    prompt = ChatPromptTemplate.from_template(prompt_text_eng)

    model = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    summarize_chain = {"element" : lambda x: x} | prompt | model | StrOutputParser()

    text_summaries = []
    table_summaries = []

    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
    elif texts:
        text_summaries = texts

    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

    return text_summaries, table_summaries


def encode_image(image_path):
    """이미지를 base64 문자열로 변환"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_summarize(img_base64, prompt):
    """이미지 요약 생성"""
    chat = ChatOpenAI(model="gpt-4o-mini", max_tokens=1024)

    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"ure":f"data:image/jpeg;base64,{img_base64}"},
                },
                ]
            )
        ]
    )
    return msg.content


def generate_img_summaries(path):
    img_base64_list = []
    image_summaries = []

    prompt_kr="""You are an assistant tasked with summarizing images for retrieval.
    These summaries will be embedded and used to retrieve the raw image.
    Provide a concise summary of the image that is well optimized for retrieval.
    The summary should be written in Korean (Hangul)."""

    for img_file in sorted(os.listdir(path)):
        if img_file.endwith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt_kr))

    return img_base64_list, image_summaries