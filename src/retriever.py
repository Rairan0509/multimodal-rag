import os
import re
import uuid
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatMessagePromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


try:
    from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
except ImportError:
    print("MultiVectorRetriever 클래스가 필요합니다")

try:
    from langchain_core.stores import InMemoryStore
except ImportError:
    print("InMemoryStore 클래스가 필요합니다")

vectorstore = Chroma(
    collection_name="multimodal_rag", embedding_function= OpenAIEmbeddings(model="text-embedding-3-small")
)

store = InMemoryStore()

id_key = "doc_id"

retriever = MultiVectorRetriever(
    vectorstore=vectorstore, docstore=store, id_key=id_key
)

def add_documents(retriever, doc_summaries, doc_contents):
    """요약본은 벡터 DB에, 원본은 docstore에 '같은 id'로 저장합니다."""
    doc_ids = []

    doc_ids = [str(uuid.uuid4()) for _ in doc_contents]

    summary_docs = [
        Document(page_content= s, metadata={id_key:doc_ids[i]})
        for i, s in enumerate(doc_summaries)
    ]

    retriever.vectorstore.add_documents(summary_docs)

    retriever.docstore.mset(list(zip(doc_ids, doc_contents)))


def split_image_text_types(docs):
    b64_images = []
    texts_out = []

    for doc in docs:
        doc = str(doc.page_content if isinstance(doc, Document) else doc)
        if re.match("^[A-Za-z0-9+/]{100,}={0,2}$", doc):
            b64_images.append(doc)
        else:
            texts_out.append(doc)
    return {"images": b64_images, "texts": texts_out}


def img_prompt_func(data_dict):
    messages = []

    for image in data_dict["context"]["images"]:
        messages.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image}"}
        })
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages.append({
        "type": "text",
        "text": ("자료를 근거로 질문에 한국어로 답하세요\n"
                 "자료에 없는 내용은 모른다고 답하세요\n\n"
                 f"질문: {data_dict['question']}\n\n"
                 f"참고 텍스트 : \n {formatted_texts}"
                 ),
    })
    return [HumanMessage(content=messages)]


model = ChatOpenAI(temperature=0, model="gpt-4o-mini", max_tokens=1024)

chain = (
    {
        "context" : retriever | RunnableLambda(split_image_text_types),
        "question": RunnablePassthrough(),
    } | RunnableLambda(img_prompt_func)
    | model
    | StrOutputParser()
)