import os
import time
import re
import io
from typing import List, Dict

import pdfplumber
from PIL import Image
from dotenv import load_dotenv
from supabase import create_client, Client
import google.generativeai as genai


# =========================================
# 0. 환경 설정 (.env 필요)
#   SUPABASE_URL
#   SUPABASE_SERVICE_ROLE
#   GOOGLE_API_KEY
# =========================================
load_dotenv()

SUPABASE_URL = "https://wzafalbctqkylhyzlfej.supabase.co"
SUPABASE_KEY = os.getenv("supbase_service_role")
GOOGLE_API_KEY = os.getenv("google_api")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GOOGLE_API_KEY)


# =========================================
# 1. 공통 유틸: 텍스트 정리
# =========================================
def clean_text_basic(text: str) -> str:
    """기본 개행/공백 정리"""
    text = text.replace("\r\n", "\n")
    text = text.replace("\t", " ")
    text = re.sub(r" {2,}", " ", text)      # 연속 공백 축소
    text = re.sub(r"\n{3,}", "\n\n", text)  # 빈 줄 2줄까지만
    return text.strip()


def page_to_markdown(page_text: str) -> str:
    """
    아주 심플한 Markdown 정리.
    필요하면 bullet/헤더 인식 규칙을 점점 추가하면 됨.
    """
    text = clean_text_basic(page_text)
    text = text.replace("•", "- ").replace("◦", "- ")
    return text


# =========================================
# 2. 에러코드 표 페이지 감지 + Vision 파싱
# =========================================
def is_error_table_page(raw_text: str) -> bool:
    """
    이 페이지가 '고장 신고 전 확인 사항' 에러코드 표인지 판별하는 간단한 규칙.
    필요하면 키워드 추가/수정해서 쓰면 됨.
    """
    keywords = ["고장 신고 전 확인 사항", "표시부 알림", "해결책", "원인"]
    return any(k in raw_text for k in keywords)


def extract_page_image(page) -> Image.Image:
    """pdfplumber page → PIL 이미지로 변환"""
    pil_img = page.to_image(resolution=200).original
    return pil_img


def parse_error_table_with_gemini(page_image: Image.Image) -> List[Dict]:
    """
    에러코드 표 페이지 전체 이미지를 Gemini Vision에 보내서
    row 단위 JSON으로 파싱.

    기대 응답 형식:
    [
      {
        "code": "UE",
        "symptom": "...",
        "cause": "...",
        "solution": "..."
      },
      ...
    ]
    """
    buf = io.BytesIO()
    page_image.save(buf, format="PNG")
    buf.seek(0)

    prompt = """
다음 이미지는 세탁기 사용설명서의 '고장 신고 전 확인 사항' 표이다.
각 행에는
- 표시부 알림(디스플레이에 보이는 코드 또는 메시지)
- 원인
- 해결책
이 있다.

이 표를 JSON 배열로 추출하라. 형식은 다음과 같다.

[
  {
    "code": "UE",
    "symptom": "탈수 시 진동, 소음이 요란합니다.",
    "cause": "세탁물이 한쪽으로 치우쳐 있습니다.",
    "solution": "세탁물을 고르게 펴십시오."
  },
  ...
]

설명이 없는 칸은 빈 문자열로 둔다.
"""

    model = genai.GenerativeModel("gemini-1.5-pro")
    resp = model.generate_content(
        [prompt, buf.getvalue()],
    )

    import json
    rows = json.loads(resp.text)
    return rows


def make_error_sections_from_rows(
    rows: List[Dict],
    page_number: int,
) -> List[Dict]:
    """
    에러코드 row JSON → manual_sections용 섹션 텍스트로 변환
    force_category/force_title은 이후 메타 생성 단계에서 그대로 사용
    """
    sections: List[Dict] = []
    for row in rows:
        code = (row.get("code") or "").strip()
        symptom = (row.get("symptom") or "").strip()
        cause = (row.get("cause") or "").strip()
        solution = (row.get("solution") or "").strip()

        content = f"""에러코드: {code}

증상: {symptom}

원인: {cause}

해결책: {solution}
"""
        sections.append({
            "page_number": page_number,
            "content_markdown": content,
            "force_category": "error",
            "force_title": f"{code} 오류" if code else "에러코드 안내",
        })
    return sections


def extract_pages_and_error_sections(pdf_path: str) -> (List[Dict], List[Dict]):
    """
    PDF 전체를 돌면서
    - 일반 페이지 텍스트 목록
    - 에러코드 표에서 뽑은 섹션 목록
    을 동시에 만들어 반환.

    normal_pages: [{page_number, raw_text}, ...]
    error_sections: [{
        page_number,
        content_markdown,
        force_category,
        force_title
    }, ...]
    """
    normal_pages: List[Dict] = []
    error_sections: List[Dict] = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            raw_text = (page.extract_text() or "").strip()

            if is_error_table_page(raw_text):
                img = extract_page_image(page)
                rows = parse_error_table_with_gemini(img)
                secs = make_error_sections_from_rows(rows, page_number=i)
                error_sections.extend(secs)
            else:
                normal_pages.append({
                    "page_number": i,
                    "raw_text": raw_text,
                })

    return normal_pages, error_sections


# =========================================
# 3. 일반 페이지 → Markdown 섹션 청크
# =========================================
def split_markdown_into_sections(
    pages: List[Dict],
    max_chars: int = 1200,
) -> List[Dict]:
    """
    normal_pages 리스트를 받아 섹션(청크) 리스트로 변환.

    return:
    [
      {
        "page_number": int,
        "content_markdown": str,
        "force_category": None,
        "force_title": None
      },
      ...
    ]
    """
    sections: List[Dict] = []

    for page in pages:
        page_num = page["page_number"]
        md_text = page_to_markdown(page["raw_text"])

        if not md_text.strip():
            continue

        paragraphs = [p.strip() for p in md_text.split("\n\n") if p.strip()]

        current_chunk = ""
        for para in paragraphs:
            candidate = (
                (current_chunk + "\n\n" + para).strip()
                if current_chunk else para
            )

            if len(candidate) <= max_chars:
                current_chunk = candidate
            else:
                if current_chunk:
                    sections.append({
                        "page_number": page_num,
                        "content_markdown": current_chunk,
                        "force_category": None,
                        "force_title": None,
                    })
                current_chunk = para

        if current_chunk:
            sections.append({
                "page_number": page_num,
                "content_markdown": current_chunk,
                "force_category": None,
                "force_title": None,
            })

    return sections


# =========================================
# 4. 섹션 메타데이터 (제목/카테고리) + 임베딩
# =========================================
def analyze_section_with_gemini(content: str) -> Dict:
    """
    섹션 텍스트를 넣으면:
    - section_title (15자 이내)
    - category     (button / course / error / maintenance / other)
    를 돌려줌.
    """
    prompt = f"""
다음은 세탁기 사용설명서의 한 섹션 내용이다.

이 텍스트를 보고 아래 정보를 JSON 형식으로 만들어라.
- section_title: 이 섹션을 잘 대표하는 제목 (15자 이내, 한국어)
- category: 다음 중 하나
  - "button": 버튼/조작부 설명
  - "course": 세탁 코스/프로그램 설명
  - "error": 오류코드/에러 설명
  - "maintenance": 관리/청소/안전 주의
  - "other": 위에 해당하지 않는 기타

텍스트:
\"\"\"{content}\"\"\"
"""

    model = genai.GenerativeModel("gemini-1.5-pro")
    resp = model.generate_content(prompt)
    text = resp.text.strip()

    import json
    meta = json.loads(text)

    return {
        "section_title": meta.get("section_title", "")[:50],
        "category": meta.get("category", "other"),
    }


def get_embedding(text: str) -> str:
    """
    Gemini 임베딩을 구하고 JSON 문자열로 반환.
    (DB에는 text 컬럼으로 저장하고, 나중에 파싱해서 사용)
    """
    emb_resp = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
    )
    embedding = emb_resp["embedding"]  # [float, float, ...]
    import json
    return json.dumps(embedding)


# =========================================
# 5. Supabase insert 함수들
# =========================================
def insert_manual_document(
    model_id: str,
    title: str,
    version: str,
    file_url: str,
) -> int:
    """
    manual_documents 에 한 줄 넣고 manual_id 리턴
    """
    data = {
        "model_id": model_id,
        "title": title,
        "version": version,
        "file_url": file_url,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    res = supabase.table("manual_documents").insert(data).execute()
    manual_id = res.data[0]["manual_id"]
    return manual_id


def insert_manual_sections(manual_id: int, sections: List[Dict]) -> None:
    """
    섹션 리스트를 manual_sections 테이블에 일괄 insert.
    - force_category/force_title 이 있으면 그대로 사용
    - 없으면 Gemini로 메타 생성
    """
    rows = []
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    for sec in sections:
        content = sec["content_markdown"]
        page_num = sec["page_number"]
        force_cat = sec.get("force_category")
        force_title = sec.get("force_title")

        if force_cat or force_title:
            section_title = force_title or ""
            category = force_cat or "other"
        else:
            meta = analyze_section_with_gemini(content)
            section_title = meta["section_title"]
            category = meta["category"]

        embedding_json = get_embedding(content)

        rows.append({
            "manual_id": manual_id,
            "section_title": section_title,
            "content_text": content,
            "page_number": page_num,
            "category": category,
            "embedding_vector": embedding_json,
            "created_at": now,
        })

    if rows:
        supabase.table("manual_sections").insert(rows).execute()


# =========================================
# 6. 전체 파이프라인
# =========================================
def process_manual_pdf(
    pdf_path: str,
    model_id: str,
    manual_title: str,
    manual_version: str,
    file_url: str,
    max_chars: int = 1200,
):
    """
    1) manual_documents insert
    2) PDF에서 일반 페이지 + 에러코드 섹션 분리
    3) 일반 페이지 → Markdown 섹션 청크
    4) 에러 섹션과 합치기
    5) Gemini 메타/임베딩 + manual_sections insert
    """
    manual_id = insert_manual_document(
        model_id=model_id,
        title=manual_title,
        version=manual_version,
        file_url=file_url,
    )
    print(f"[INFO] manual_id={manual_id} created")

    normal_pages, error_sections = extract_pages_and_error_sections(pdf_path)
    print(f"[INFO] normal_pages={len(normal_pages)}, error_sections={len(error_sections)}")

    normal_sections = split_markdown_into_sections(
        normal_pages,
        max_chars=max_chars,
    )
    print(f"[INFO] normal_sections={len(normal_sections)}")

    # 일반 섹션 + 에러 섹션 합치기
    all_sections: List[Dict] = []
    all_sections.extend(normal_sections)
    all_sections.extend(error_sections)
    print(f"[INFO] total sections={len(all_sections)}")

    insert_manual_sections(manual_id, all_sections)
    print("[INFO] all sections inserted into manual_sections")


# =========================================
# 7. 예시 실행
# =========================================
if __name__ == "__main__":
    PDF_FILE_PATH = "통돌이 설명서.pdf"
    MODEL_ID = "TA25GZ9"
    MANUAL_TITLE = "TA25GZ9 상세 매뉴얼"
    MANUAL_VERSION = "v1.0"
    
    if os.path.exists(PDF_FILE_PATH):
        process_manual_pdf(
            pdf_path=PDF_FILE_PATH,
            model_id=MODEL_ID,
            manual_title=MANUAL_TITLE,
            manual_version=MANUAL_VERSION
        )
    else:
        print(f"❌ 파일을 찾을 수 없습니다: {PDF_FILE_PATH}")
