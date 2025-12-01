from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import google.generativeai as genai

# ==========================================
# 1. 설정 및 초기화
# ==========================================
load_dotenv()
SUPABASE_URL = "https://wzafalbctqkylhyzlfej.supabase.co"
SUPABASE_KEY = os.getenv("supbase_service_role")
GOOGLE_API_KEY = os.getenv("google_api")

# 클라이언트 직접 생성
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GOOGLE_API_KEY)

# 모델 설정
embedding_model = "models/text-embedding-004"
generation_model = genai.GenerativeModel("gemini-2.5-flash")

# ==========================================
# 2. FastAPI 서버 설정
# ==========================================
app = FastAPI()

class ChatRequest(BaseModel):
    user_message: str
    user_id: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]

# 🔥 핵심 로직: 랭체인 없이 직접 구현 (에러 원천 차단)
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    print(f"📩 [Spring -> Python] 요청 도착: {req.user_message}")
    
    try:
        # 1. 질문을 벡터(숫자)로 변환
        emb_result = genai.embed_content(
            model=embedding_model,
            content=req.user_message,
            task_type="retrieval_query"
        )
        query_vector = emb_result['embedding']

        # 2. DB 함수 직접 호출 (여기서 에러가 안 나게 됨!)
        #    아까 만든 SQL 함수의 파라미터 이름과 정확히 일치시킵니다.
        rpc_response = supabase.rpc("match_manual_sections", {
            "query_embedding": query_vector,
            "match_threshold": 0.3,
            "match_count": 3
        }).execute()
        
        # 3. 검색 결과 정리
        search_results = rpc_response.data
        
        if not search_results:
            print("⚠️ 검색 결과 없음")
            return ChatResponse(answer="매뉴얼에서 관련 내용을 찾을 수 없습니다.", sources=[])

        # 검색된 내용을 하나로 합침
        context_text = "\n\n".join([f"- {item['content']}" for item in search_results])
        source_titles = list(set([item['section_title'] for item in search_results]))

        # 4. 프롬프트 구성
        prompt = f"""
        당신은 LG ThinQ 봇입니다. 아래 [매뉴얼 내용]을 바탕으로 사용자의 질문에 친절하게 답하세요.
        
        [매뉴얼 내용]:
        {context_text}
        
        [질문]: {req.user_message}
        
        [답변]:
        """

        # 5. 답변 생성
        gen_response = generation_model.generate_content(prompt)
        final_answer = gen_response.text

        print(f"✅ 답변 생성 완료: {final_answer[:30]}...")

        return ChatResponse(
            answer=final_answer,
            sources=source_titles
        )

    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        # 에러 내용을 그대로 보여줘서 디버깅을 돕습니다.
        return ChatResponse(
            answer=f"서버 에러 발생: {str(e)}",
            sources=[]
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)