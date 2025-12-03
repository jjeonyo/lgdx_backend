import os
import time
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from supabase import create_client, Client
import google.generativeai as genai
from google.api_core import exceptions

# ==========================================
# 1. 환경 설정 및 초기화
# ==========================================
load_dotenv()

# API 키 및 URL 로드
SUPABASE_URL = "https://wzafalbctqkylhyzlfej.supabase.co"
SUPABASE_KEY = os.getenv("supbase_service_role")
GOOGLE_API_KEY = os.getenv("google_api")
FIREBASE_KEY_PATH = "C:\dxfirebasekey\serviceAccountKey.json"

if not all([SUPABASE_URL, SUPABASE_KEY, GOOGLE_API_KEY]):
    raise ValueError("❌ 환경변수(.env) 설정을 확인해주세요.")

# 1-1. 파이어베이스 초기화
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(FIREBASE_KEY_PATH)
        firebase_admin.initialize_app(cred)
        print("🔥 파이어베이스 연결 성공!")
    except Exception as e:
        print(f"❌ 파이어베이스 키 파일 오류: {e}")
        pass

db = firestore.client()

# 1-2. Supabase & Gemini 초기화
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GOOGLE_API_KEY)

# 모델 설정 (최신 모델 적용)
EMBEDDING_MODEL = "models/text-embedding-004"
# 사용자님이 원하신 2.5 버전이 아직 정식 배포 전이라면 2.0-flash-exp 사용 권장
# 만약 2.5 접근 권한이 있으시면 "gemini-2.5-flash"로 바꾸세요.
GENERATION_MODEL_ID = "gemini-2.5-flash" 
GENERATION_MODEL = genai.GenerativeModel(GENERATION_MODEL_ID)

print(f"🚀 AI 모델 로드 완료: {GENERATION_MODEL_ID}")


# ==========================================
# 2. 헬퍼 함수들
# ==========================================

def save_to_firebase(user_id: str, sender: str, text: str, msg_type: str = "TEXT"):
    try:
        doc_ref = db.collection("chat_rooms").document(f"room_{user_id}").collection("messages")
        doc_ref.add({
            "sender": sender,
            "text": text,
            "message_type": msg_type,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        print(f"💾 [Firebase] {sender}: {text[:10]}...")
    except Exception as e:
        print(f"⚠️ Firebase 저장 실패: {e}")

def get_embedding(text: str):
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']
    except Exception as e:
        print(f"❌ 임베딩 생성 실패: {e}")
        return None

def optimize_search_query(original_query: str) -> str:
    """사용자 질문을 검색용 키워드로 변환 (쿼리 확장)"""
    try:
        prompt = f"""
        규칙: 문장이 아닌 **키워드 나열** 형태. LG 세탁기 용어 적극 활용.
        
        사용자: "{original_query}"
        변환:
        """
        response = GENERATION_MODEL.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"⚠️ 쿼리 확장 실패: {e}")
        return original_query


# ==========================================
# 3. FastAPI 서버 설정
# ==========================================
app = FastAPI()

class ChatRequest(BaseModel):
    user_message: str
    user_id: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]

# -------------------------------------------------------
# [API 1] 텍스트 챗봇 (하이브리드 검색 적용)
# -------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    print(f"📩 [요청 도착] ID: {req.user_id}, 내용: {req.user_message}")
    
    try:
        # 1. 사용자 질문 저장
        save_to_firebase(req.user_id, "user", req.user_message, "user")  # message_type을 sender와 동일하게

        # 2. 쿼리 확장 (키워드 검색용)
        search_keyword = optimize_search_query(req.user_message)
        print(f"✨ [쿼리 확장] '{req.user_message}' -> '{search_keyword}'")

        # 3. 임베딩 생성 (벡터 검색용)
        query_vector = get_embedding(search_keyword)
        if not query_vector: raise Exception("임베딩 실패")

        # 🔥 [핵심] 하이브리드 검색 RPC 호출
        # (Supabase에 hybrid_search 함수가 만들어져 있어야 함)
        rpc_response = supabase.rpc("hybrid_search", {
            "query_text": search_keyword,    # 텍스트 매칭용
            "query_embedding": query_vector, # 의미 검색용
            "match_threshold": 0.1,          # 기준 점수
            "match_count": 5,                # 가져올 개수
            "w_vector": 0.9,                 # 벡터 가중치 (0.0~1.0)
            "w_keyword": 0.1                 # 키워드 가중치 (0.0~1.0)
        }).execute()
        
        search_results = rpc_response.data
        
        if not search_results:
            final_answer = "죄송합니다. 매뉴얼에서 관련 내용을 찾을 수 없습니다. 고객센터에 문의해주세요."
            source_titles = []
        else:
            # 5. 프롬프트 구성 (하이브리드 결과 사용)
            context_list = []
            for item in search_results:
                # hybrid_search 함수는 'content_text'로 리턴함
                text = item.get('content_text') or item.get('content') or ""
                title = item.get('section_title') or "정보"
                context_list.append(f"- {text} (출처: {title})")
            
            context_text = "\n\n".join(context_list)
            source_titles = list(set([item.get('section_title', '제목없음') for item in search_results]))

            prompt = f"""
            당신은 LG전자 가전제품 전문 상담원 'ThinQ 봇'입니다.
            사용자의 질문에 대해 아래 제공된 [매뉴얼 데이터]를 기반으로 친절하고 정확하게 답변해 주세요.
            세탁방법에 대해 물었는데 메뉴얼에 없다면 다른 특정 세탁기의 기능은 말하지 말고 특정 세탁기가 없어도 누구나 적용가능한 방법을 너가 알고 있는 최대한 정확한 지식으로 친절하게 답변해줘
            
            [지침]
            1. 표 내용은 문장으로 자연스럽게 풀어서 설명하세요.
            2. 답변 끝에는 참고한 페이지 번호나 섹션을 언급해주세요.
            3. 사용자가 '통돌이', '드럼' 등 구어체를 써도, 매뉴얼의 해당 제품군 내용으로 답변하세요.
            4. 질문에 '띵큐'가 있다면 답변할 때 'LG ThinQ'로 바꿔서 말해주세요.
            
            [매뉴얼 데이터]:
            {context_text}
            
            [사용자 질문]: {req.user_message}
            (참고: '{search_keyword}' 관련 내용을 검색했습니다.)
            
            [답변]:
            """
            
            # 6. 답변 생성
            gen_resp = GENERATION_MODEL.generate_content(prompt)
            final_answer = gen_resp.text

        # 7. 답변 저장
        save_to_firebase(req.user_id, "ai", final_answer, "ai")  # message_type을 sender와 동일하게
        print(f"✅ [답변 완료] {final_answer[:30]}...")

        return ChatResponse(
            answer=final_answer,
            sources=source_titles
        )

    except Exception as e:
        print(f"❌ 서버 에러: {e}")
        return ChatResponse(
            answer=f"죄송합니다. 오류가 발생했습니다. ({str(e)})",
            sources=[]
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)