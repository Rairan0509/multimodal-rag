import os

from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf


def extract_pdf_elements(path, fname):
    """
    PDF 파일에서 이미지, 테이블, 텍스트 블록을 추출합니다.
    path: 파일 경로 (추출된 이미지 파일이 저장될 위치)
    fname: 파일 이름
    """
    return partition_pdf(
        filename=path+fname,
        strategy="hi_res",      # 레이아웃 분석 모델로 정밀 추출(fast로 하니 추출된 요소가 0개 나왔음)
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=path
    )


def categorize_elements(raw_pdf_elements):
    """
    PDF에서 추출한 요소들을 테이블과 텍스트로 분류합니다.
    raw_pdf_elements: unstructured.documents.elements 리스트
    """
    tables = []
    texts = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))     # 테이블 요소를 저장
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))
    return texts, tables


# 현재 작업 디렉터리 경로 반환
current_directory = os.getcwd()

# fname: 처리할 PDF 파일 이름 / fpath: PDF가 저장된 폴더 경로
fname = "assignment.pdf"
fpath = os.path.join(os.path.dirname(current_directory), "data/")

raw_pdf_elements = extract_pdf_elements(fpath, fname)
texts, tables = categorize_elements(raw_pdf_elements)

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=100
)

joined_texts = " ".join(texts)
texts_2k_token = text_splitter.split_text(joined_texts)