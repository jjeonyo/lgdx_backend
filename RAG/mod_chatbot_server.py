import os
from pathlib import Path
import time
import subprocess
import sys
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
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
FIREBASE_KEY_PATH = "/Users/harry/LG DX SCHOOL/lgdx_backend/serviceAccountKey.json"

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
        room_id = f"room_{user_id}"
        doc_ref = db.collection("chat_rooms").document(room_id).collection("messages")
        message_data = {
            "sender": sender,
            "text": text,
            "content": text,  # vision/test.py와 통일을 위해 content 필드도 추가
            "message_type": "chat_bot",  # 메시지 타입: 'chat_bot' (텍스트 챗봇)
            "timestamp": firestore.SERVER_TIMESTAMP
        }
        doc_ref.add(message_data)
        print(f"💾 [Firebase] 저장 완료 - room: {room_id}, sender: {sender}, text: {text[:30]}...")
        print(f"💾 [Firebase] 저장된 데이터: {message_data}")
    except Exception as e:
        print(f"❌ [Firebase] 저장 실패: {e}")
        import traceback
        traceback.print_exc()

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
from fastapi.staticfiles import StaticFiles
import asyncio
import socket

app = FastAPI()

# 정적 파일 서빙 설정 (assets_generate 폴더를 /assets 경로로 노출)
assets_path = Path(__file__).parent.parent / "generate" / "assets_generate"
assets_path.mkdir(parents=True, exist_ok=True) # 폴더가 없으면 생성
app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")

# [서버 IP 가져오기 함수]
def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Google DNS 서버에 접속 시도하여 내 IP 확인 (실제 접속은 안함)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"

SERVER_IP = get_host_ip()
print(f"🌐 Server IP: {SERVER_IP}")


# [비디오 감시 태스크]
# assets 폴더를 감시하다가 새 비디오가 생기면 Firestore에 메시지를 남깁니다.
# 이렇게 하면 Firebase Storage 없이도 앱에서 비디오가 뜹니다.
processed_files = set()

async def watch_new_videos():
    print("👀 Video Watcher Started...")
    
    # 초기 상태: 이미 있는 파일은 처리된 것으로 간주 (원하면 제거 가능)
    if assets_path.exists():
        for f in assets_path.glob("*.mp4"):
            processed_files.add(f.name)
            
    while True:
        try:
            if assets_path.exists():
                # 현재 모든 mp4 파일
                current_files = list(assets_path.glob("*.mp4"))
                
                for file_path in current_files:
                    if file_path.name not in processed_files:
                        # 새 파일 발견!
                        print(f"🎬 New Video Detected: {file_path.name}")
                        
                        # 파일이 완전히 써질 때까지 잠시 대기 (옵션)
                        await asyncio.sleep(2)
                        
                        # 1. 로컬 URL 생성
                        # 예: http://192.168.0.x:8000/assets/filename.mp4
                        video_url = f"http://{SERVER_IP}:8000/assets/{file_path.name}"
                        
                        # 2. Firestore에 메시지 강제 저장
                        # (데모용: 가장 최근 방이나 기본 방에 저장)
                        # 실제로는 generate.py에서 session_id를 파일명에 넣거나 별도 전달해야 정확함
                        # 여기서는 'room_user_001' 등 고정값 또는 가장 최근 수정된 방을 찾음
                        
                        target_room_id = "room_user_001" # Default
                        
                        # [고급] 가장 최근 대화가 있었던 방 찾기
                        try:
                            # 최근 메시지가 있는 방 찾기 (복잡하므로 생략하거나 간단히 구현)
                            # 여기서는 간단히 고정 ID 사용하되, 필요시 로직 추가
                            pass
                        except: pass

                        print(f"📤 Sending video message to {target_room_id}...")
                        
                        # DB 저장
                        doc_ref = db.collection("chat_rooms").document(target_room_id).collection("messages")
                        doc_ref.add({
                            "sender": "ai",
                            "text": "솔루션 영상을 생성했습니다. (Local Server)",
                            "video_url": video_url,
                            "message_type": "VIDEO",
                            "timestamp": firestore.SERVER_TIMESTAMP
                        })
                        
                        print(f"✅ Saved video message: {video_url}")
                        
                        # 처리 목록에 추가
                        processed_files.add(file_path.name)
                        
        except Exception as e:
            print(f"⚠️ Watcher Error: {e}")
            
        await asyncio.sleep(3) # 3초마다 확인

@app.on_event("startup")
async def startup_event():
    # 백그라운드 태스크로 감시 시작
    asyncio.create_task(watch_new_videos())


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
    print(f"📩 [Python] 요청 도착 - userId: {req.user_id}, message: {req.user_message}")
    
    try:
        # 1. 사용자 질문 저장
        print(f"💾 [Python] 사용자 메시지 Firebase 저장 시작...")
        save_to_firebase(req.user_id, "user", req.user_message)

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
            메뉴얼에 없는 내용은 메뉴얼에 없는 내용이라고 말하지말고 자연스럽게 너가 알고 있는 지식으로 친절하게 답변해줘
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
        print(f"💾 [Python] AI 답변 Firebase 저장 시작...")
        save_to_firebase(req.user_id, "ai", final_answer)
        print(f"✅ [Python] 답변 완료 및 저장 완료: {final_answer[:30]}...")

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

# --- 비디오 상태 확인용 글로벌 변수 ---
# 실제로는 DB나 Redis를 써야 하지만, 간단한 데모를 위해 메모리에 상태 저장
# key: video_id (또는 user_id), value: {'status': '...', 'url': '...'}
video_generation_status = {}

@app.post("/generate-video")
async def generate_video_endpoint():
    try:
        # Current file directory: lgdx_backend/RAG
        current_dir = Path(__file__).parent
        # Target script: lgdx_backend/generate/generate.py
        script_path = current_dir.parent / "generate" / "generate.py"
        
        print(f"🎥 실행 요청: {script_path}")
        
        if not script_path.exists():
             raise HTTPException(status_code=404, detail=f"Script not found at {script_path}")

        # 상태를 'processing'으로 설정
        # 실제 앱에서는 user_id 등을 받아야 함. 여기선 'demo_video'라는 고정 ID 사용
        video_generation_status['demo_video'] = {'status': 'processing'}

        # Run the script asynchronously using subprocess
        # 스크립트가 완료되면 파일을 생성하거나 DB를 업데이트한다고 가정
        # 여기서는 단순히 스크립트를 실행하고, 폴링 시 파일 존재 여부를 확인할 수도 있음
        subprocess.Popen([sys.executable, str(script_path)])
        
        return {"status": "started", "message": "Video generation started in background"}
    except Exception as e:
        print(f"❌ 실행 실패: {e}")
        video_generation_status['demo_video'] = {'status': 'failed'}
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/check-video-status")
async def check_video_status():
    # 1. 생성된 파일이 있는지 확인하는 로직
    # lgdx_backend/generate/assets_generate/ 폴더 확인
    try:
        base_dir = Path(__file__).parent.parent / "generate" / "assets_generate"
        
        # 가장 최근에 생성된 mp4 파일 찾기
        if not base_dir.exists():
             return {"status": "processing"}
             
        mp4_files = list(base_dir.glob("*.mp4"))
        if not mp4_files:
            return {"status": "processing"}
            
        # 최신 파일 찾기
        latest_file = max(mp4_files, key=os.path.getctime)
        
        # 파일이 생성된지 얼마 안 되었으면(예: 1분 이내) 완료로 간주
        # 실제로는 generate.py가 완료 신호를 어딘가(DB/파일)에 남기는 게 정확함
        # 여기서는 파일 존재만으로 체크
        
        # 클라이언트에서 접근 가능한 URL로 변환 필요
        # 지금은 로컬 파일 경로를 리턴하거나, 별도 정적 파일 서빙 설정 필요
        # 데모용: 파일명 리턴 (외부 접속을 위해 0.0.0.0 또는 호스트 IP 사용 권장, 여기선 예시로 localhost 유지하나 실제론 앱에서 접근 가능한 주소여야 함)
        # 앱에서 접근하려면 실행 서버의 IP가 필요함. 
        
        # (임시) 서버 IP를 알 수 없으면 상대 경로만 리턴하고 앱에서 Base URL 붙여서 쓰게 할 수도 있음
        return {
            "status": "completed", 
            "video_url": f"/assets/{latest_file.name}" 
        }
        
    except Exception as e:
        print(f"Check status error: {e}")
        return {"status": "processing"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)