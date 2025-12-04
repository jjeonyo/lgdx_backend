import asyncio
import os
import cv2
import pathlib
import sys
import time
from datetime import datetime
import pyaudio
import warnings
import traceback
import threading
import queue
from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import uvicorn
from typing import List
import base64
import json
import asyncio


# [추가] RAG & Supabase 관련
try:
    import firebase_admin
    from firebase_admin import credentials, firestore  # pyright: ignore[reportMissingImports]
except ImportError:
    print("❌ supabase 또는 google-generativeai 라이브러리가 없습니다.")
    print("   pip install supabase google-generativeai")
    sys.exit(1)


# [수정] google.genai에서 types 임포트
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("❌ google-genai 라이브러리가 설치되지 않았습니다.")
    sys.exit(1)

# [Supabase 라이브러리 추가]
try:
    from supabase import create_client, Client
except ImportError:
    print("❌ supabase 라이브러리가 설치되지 않았습니다. 'pip install supabase'를 실행하세요.")
    sys.exit(1)

# [설정] 경고 메시지 숨기기
warnings.filterwarnings("ignore")

# ==========================================
# .env 파일 로드 (프로젝트 루트에서 찾기)
# ==========================================
def load_environment():
    try:
        # 프로젝트 루트 경로 찾기
        project_root = pathlib.Path(__file__).parent.parent.absolute()
        env_path = project_root / ".env"
        
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            print(f"✅ .env 파일 로드 완료: {env_path}")
        else:
            print(f"⚠️ .env 파일을 찾을 수 없습니다: {env_path}")
    except Exception as e:
        print(f"⚠️ .env 파일 로드 중 오류: {e}")

load_environment()

# ==========================================
API_KEY = os.getenv("GEMINI_API_KEY")

# Firebase 키 경로 설정 (환경변수 우선, 없으면 현재 디렉토리의 FirebaseAdmin.json 사용)
project_root = pathlib.Path(__file__).parent.parent.absolute()
current_dir = pathlib.Path(__file__).parent.absolute()

# 우선순위 1: 환경변수
# 우선순위 2: vision 폴더 내 FirebaseAdmin.json
# 우선순위 3: 프로젝트 루트의 serviceAccountKey.json
default_key_path = "C:\dxfirebasekey\serviceAccountKey.json"
#FIREBASE_KEY_PATH = os.getenv("FIREBASE_KEY_PATH", str(default_key_path))

# Realtime Database URL (Firestore 사용 시 불필요하지만 참고용으로 남김/삭제 가능)
FIREBASE_DATABASE_URL = "https://team-dxproject-default-rtdb.asia-southeast1.firebasedatabase.app/"

MODEL_ID = "gemini-2.5-flash-native-audio-preview-09-2025"

# [오디오 설정]
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
INPUT_RATE = 16000
OUTPUT_RATE = 24000
CHUNK_SIZE = 512

# ==========================================

def load_environment():
    try:
        current_dir = pathlib.Path(__file__).parent.absolute()
        env_path = None
        for parent in [current_dir] + list(current_dir.parents):
            check_path = parent / ".env"
            if check_path.exists():
                env_path = check_path
                break

        if env_path:
            load_dotenv(dotenv_path=env_path)
        else:
            print("⚠️ .env 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"❌ .env 로드 오류: {e}")

load_environment()

API_KEY = os.getenv("GEMINI_API_KEY")

# Firebase 키 경로 설정 로직 개선
project_root = pathlib.Path(__file__).parent.parent.absolute()
default_firebase_path = project_root / "serviceAccountKey.json"
FIREBASE_KEY_PATH = "C:\dxfirebasekey\serviceAccountKey.json"

if not API_KEY:
    print("❌ API 키가 없습니다. .env 파일을 확인해주세요.")
    sys.exit(1)

# RAG용 추가 키
SUPABASE_URL = "https://wzafalbctqkylhyzlfej.supabase.co"
SUPABASE_KEY = os.getenv("supbase_service_role") or os.getenv("SUPABASE_SERVICE_ROLE")

if not SUPABASE_KEY:
    print("⚠️ Supabase 키가 없습니다. RAG 기능이 제한될 수 있습니다.")


MODEL_ID = "gemini-2.5-flash-native-audio-preview-09-2025"
#MODEL_ID = "gemini-2.5-flash"
#MODEL_ID = "gemini-2.5-flash-preview-09-2025"
#MODEL_ID = "gemini-2.0-flash"
#MODEL_ID = "gemini-2.0-flash-exp"

# [오디오 설정]
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
INPUT_RATE = 16000
OUTPUT_RATE = 24000
CHUNK_SIZE = 512




async def perform_summarization(client, session_id):
    """Firebase에서 대화를 가져와 요약하고 결과를 DB에 저장"""
    print(f"\n🔔 [Command Received] 요약 요청을 받았습니다. (Session: {session_id})")
    
    try:
        db_client = firestore.client()
        # 1. 대화 로그 가져오기
        # Firestore: sessions/{session_id}/messages 컬렉션 조회
        messages_ref = db_client.collection('sessions').document(session_id).collection('messages')
        # created_at 기준 정렬
        docs = messages_ref.order_by('created_at').stream()
        
        messages_list = []
        for doc in docs:
            messages_list.append(doc.to_dict())

        if not messages_list:
            print("   ⚠️ 대화 내용이 없습니다.")
            return

        # 2. 텍스트 변환
        chat_context = ""
        for msg in messages_list:
            sender = msg.get('sender', 'unknown')
            content = msg.get('content', '')
            chat_context += f"[{sender}]: {content}\n"

        # 3. Gemini에게 요약 요청 (가벼운 모델 사용)
        prompt = f"""
        아래는 가전제품 수리 AI와 사용자의 대화 로그입니다.
        현재 사용자가 겪고 있는 '문제점'과 '증상'을 
        기술적인 관점에서 명확하게 1문장으로 요약해 주세요.
        
        [대화 로그]
        {chat_context}
        """

        # Gemini 호출
        resp = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        summary_text = resp.text.strip()
        print(f"   📝 요약 완료: {summary_text}")

        # 4. 결과 DB 저장 및 명령어 초기화
        # summary 필드에 결과 저장
        db_client.collection('sessions').document(session_id).update({
            'summary': summary_text,
            'command': None  # 명령 수행 완료 후 초기화 (중요)
        })

    except Exception as e:
        print(f"   ❌ 요약 중 에러 발생: {e}")



# ==========================================
# [클래스] Firebase Logger (Firestore 사용)
# ==========================================
class FirebaseLogger:
    def __init__(self):
        self.session_ref = None
        self.current_turn_text = ""
        self.last_user_text = ""  # 중복 저장 방지용
        self.db = None
        self._init_firebase()
        self._start_session()

    def _init_firebase(self):
        # 이미 앱이 초기화되어 있는지 확인 (중복 초기화 방지)
        if not firebase_admin._apps:
            try:
                if not os.path.exists(FIREBASE_KEY_PATH):
                    print(f"❌ 키 파일을 찾을 수 없습니다: {FIREBASE_KEY_PATH}")
                    sys.exit(1)
                    
                cred = credentials.Certificate(FIREBASE_KEY_PATH)
                firebase_admin.initialize_app(cred)
                print(f"🔥 Firebase 연결 성공!")
            except Exception as e:
                print(f"❌ Firebase 초기화 오류: {e}")
                sys.exit(1)
        
        self.db = firestore.client()

    def _start_session(self):
        try:
            # 'sessions' 컬렉션에 새 세션 생성 (add)
            update_time, self.session_ref = self.db.collection('sessions').add({
                'start_time': int(time.time() * 1000),  # timestamp (ms)
                'model_id': MODEL_ID,
                'status': 'active'
            })
            print(f"📄 새 세션 ID: {self.session_ref.id}")
        except Exception as e:
            print(f"❌ 세션 생성 실패: {e}")

    def log_message(self, sender, text):
        if not self.session_ref:
            print(f"⚠️ [Firebase] session_ref가 None입니다. 저장할 수 없습니다.")
            return
        if not text or not text.strip():
            print(f"⚠️ [Firebase] 빈 텍스트입니다. 저장하지 않습니다.")
            return
        try:
            # 현재 시간 정보 생성
            current_timestamp = int(time.time() * 1000)  # 밀리초 단위 타임스탬프
            current_datetime = datetime.now()  # 현재 날짜/시간 객체
            
            # 읽기 쉬운 날짜/시간 형식 (한국 시간대 기준)
            # 예: "2024-01-15 14:30:25"
            formatted_time = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            
            # 시간대 정보 (한국 표준시)
            timezone = "KST"
            
            print(f"💾 [Firebase] 저장 시도 - sender: {sender}, text: {text[:50]}..., 시간: {formatted_time}")
            # 해당 세션의 'messages' 컬렉션에 대화 추가
            doc_ref = self.session_ref.collection('messages').add({
                'sender': sender,      # 'user' or 'gemini'
                'content': text,       # 메시지 내용 (주 필드)
                'text': text,          # RAG/mod_chatbot_server.py와 통일을 위해 text 필드도 추가
                'message_type': 'live',  # 메시지 타입: 'live' (실시간 대화)
                'created_at': current_timestamp,  # 밀리초 단위 타임스탬프 (정렬/쿼리용)
                'timestamp': formatted_time,      # 읽기 쉬운 날짜/시간 형식
                'timezone': timezone              # 시간대 정보
            })
            print(f"✅ [Firebase] 저장 성공! - sender: {sender}, text 길이: {len(text)}, 시간: {formatted_time}")
        except Exception as e:
            print(f"❌ [Firebase] 로그 저장 실패: {e}")
            import traceback
            traceback.print_exc()

    def append_text(self, text):
        """스트리밍되는 텍스트 조각을 임시 버퍼에 추가"""
        if text:
            print(f"📝 [버퍼] 텍스트 추가: '{text[:50]}...' (현재 버퍼 길이: {len(self.current_turn_text)})")
            self.current_turn_text += text
        else:
            print(f"⚠️ [버퍼] 빈 텍스트가 append_text에 전달됨")

    def flush_model_turn(self):
        """버퍼에 모인 텍스트를 한 번에 로그로 저장하고 초기화"""
        if self.current_turn_text.strip():
            print(f"💾 [Firebase] AI 응답 저장 시도 - 길이: {len(self.current_turn_text)}")
            self.log_message('gemini', self.current_turn_text)
            self.current_turn_text = ""
        else:
            print(f"⚠️ [Firebase] AI 응답 버퍼가 비어있습니다.")


# ==========================================
# [클래스] Supabase RAG Engine
# ==========================================

# ==========================================
# [클래스] Supabase Hybrid RAG Engine (텍스트 + 벡터)
# ==========================================

# ==========================================
# [수정됨] Supabase Hybrid RAG Engine
# ==========================================
class SupabaseRAG:
    def __init__(self, gemini_client):
        self.gemini_client = gemini_client
        # 다른 파일들과 동일한 Supabase URL 사용 (기본값)
        # .env 파일에서 SUPABASE_URL이 있으면 사용, 없으면 기본값 사용
        self.supabase_url = os.getenv("SUPABASE_URL", "https://wzafalbctqkylhyzlfej.supabase.co")
        # .env 파일에서 supbase_service_role 키 가져오기
        self.supabase_key = os.getenv("supbase_service_role") 
        self.client = None
        
        if self.supabase_key:
            try:
                self.client = create_client(self.supabase_url, self.supabase_key)
                print(f"🔥 Supabase 하이브리드 엔진 연결 성공!")
                print(f"   URL: {self.supabase_url}")
            except Exception as e:
                print(f"❌ Supabase 초기화 오류: {e}")
                print(f"   ⚠️ Supabase 없이 계속 진행합니다.")
        else:
            print("⚠️ Supabase Key(supbase_service_role)를 .env 파일에서 찾을 수 없습니다.")
            print("   ⚠️ Supabase RAG 기능은 비활성화되지만, 다른 기능은 계속 작동합니다.")
            print(f"   .env 파일 위치: {pathlib.Path(__file__).parent.parent.absolute() / '.env'}")

    def get_embedding(self, text):
        if not self.gemini_client: return None
        try:
            # 텍스트 임베딩 생성 (Gemini)
            response = self.gemini_client.models.embed_content(
                model="text-embedding-004",
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY"
                )
            )
            if hasattr(response, 'embeddings') and response.embeddings:
                return response.embeddings[0].values
            return None
        except Exception as e:
            print(f"⚠️ 임베딩 생성 실패 (텍스트 검색만 시도): {e}")
            return None

    def search(self, query, k=3):
        if not self.client: return []
        
        # 1. 벡터 생성
        embedding = self.get_embedding(query)
        
        # 임베딩 실패 시 0으로 채운 더미 벡터 사용
        if not embedding: 
            embedding = [0.0] * 768 

        # 2. 하이브리드 검색 요청
        # (SQL 함수 파라미터 이름과 정확히 일치해야 합니다)
        params = {
            "query_text": query,          
            "query_embedding": embedding, 
            "match_threshold": 0.45,      
            "match_count": k              
        }
        
        try:
            # RPC 호출: hybrid_search
            response = self.client.rpc("hybrid_search", params).execute()
            
            results = []
            seen_content = set()
            
            data = response.data if response.data else []
            
            for row in data:
                content = row.get('content_text', '')
                if content and content not in seen_content:
                    results.append(content)
                    seen_content.add(content)
            
            return results
        except Exception as e:
            print(f"❌ Supabase 검색 실패: {e}")
            return []

# ==========================================
# [함수] 설정 및 페르소나 로드
# ==========================================

def get_config():
    current_dir = pathlib.Path(__file__).parent.absolute()
    persona_path = current_dir / "persona/persona_세탁기사용법.txt"
    
    system_instruction = ""
    if persona_path.exists():
        try:
            system_instruction = persona_path.read_text(encoding="utf-8")
            print(f"🎭 페르소나 로드됨: {persona_path.name}")
        except Exception:
            pass
    else:
        system_instruction = "너는 도움이 되는 AI 어시스턴트야. 실시간으로 대화해."

    # 툴 정의 (매뉴얼 검색)
    tools = [
        {
            "function_declarations": [
                {
                    "name": "search_manual",
                    "description": "LG 전자 제품 매뉴얼에서 문제 해결 방법, 에러 코드, 사용법 등을 검색합니다. 사용자가 기술적인 질문을 하거나 도움이 필요할 때 사용하세요.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "query": {
                                "type": "STRING",
                                "description": "검색할 질문 내용 (예: 'OE 에러 해결법', '세탁기 청소 방법')"
                            }
                        },
                        "required": ["query"]
                    }
                }
            ]
        }
    ]

    return {
        "response_modalities": ["AUDIO"],  # 오디오만 받기 (텍스트는 output_audio_transcription에서 추출)
        "input_audio_transcription": {},  # 입력 오디오를 텍스트로 변환
        "output_audio_transcription": {},  # 출력 오디오를 텍스트로 변환 (AI 응답)
        "speech_config": {
            "voice_config": {
                "prebuilt_voice_config": {
                    "voice_name": "Laomedeia" # 목소리 바꾸기
                }
            }
        },
        "system_instruction": system_instruction
    }


# ==========================================
# [API 설정] FastAPI & Chat Endpoint
# ==========================================
app = FastAPI()
chat_client = None
chat_rag_engine = None

class ChatRequest(BaseModel):
    user_id: str
    user_message: str

class ChatResponse(BaseModel):
    answer: str

@app.on_event("startup")
async def startup_event():
    global chat_client, chat_rag_engine
    # API용 클라이언트 별도 초기화
    chat_client = genai.Client(api_key=API_KEY)
    chat_rag_engine = SupabaseRAG(chat_client)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    print(f"📩 [Spring -> Python] 요청 도착: {req.user_message}")
    
    context_text = ""
    if chat_rag_engine:
        # RAG 검색 실행
        results = chat_rag_engine.search(req.user_message, k=3)
        if results:
            context_text = "\n\n".join(results)
            print(f"   ✅ 검색 성공: {len(results)}건")
        else:
            print("   ⚠️ 검색 결과 없음")
    
    prompt = f"""
    당신은 LG전자 가전제품 수리 및 사용법을 안내하는 AI 어시스턴트입니다.
    아래 [매뉴얼 정보]를 바탕으로 사용자의 질문에 친절하고 명확하게 답변해 주세요.
    매뉴얼에 관련 정보가 없다면, 일반적인 지식을 활용하되 "매뉴얼에는 없는 내용이지만..."이라고 언급해 주세요.

    [매뉴얼 정보]
    {context_text}

    [사용자 질문]
    {req.user_message}
    """

    try:
        response = chat_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        return ChatResponse(answer=response.text)
    except Exception as e:
        print(f"❌ 답변 생성 오류: {e}")
        return ChatResponse(answer="죄송합니다. 현재 답변을 생성할 수 없습니다.")

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("📱 Flutter Client Connected")
    
    # Gemini Live 세션 준비
    config = get_config()
    client = genai.Client(api_key=API_KEY)
    
    # 큐 생성
    video_queue = asyncio.Queue()
    audio_queue = asyncio.Queue()
    
    async with client.aio.live.connect(model=MODEL_ID, config=config) as session:
        print("✅ Gemini Live Session Started")

        # [Task 1] WebSocket -> Gemini (Receive from Flutter)
        async def receive_from_flutter():
            try:
                while True:
                    # 텍스트(JSON)로 수신 (이미지/오디오는 Base64 인코딩됨)
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    if message['type'] == 'audio':
                        # Base64 -> Bytes -> Gemini
                        audio_bytes = base64.b64decode(message['data'])
                        await session.send_realtime_input(
                            audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000")
                        )
                    elif message['type'] == 'image':
                        # Base64 -> Bytes -> Gemini
                        image_bytes = base64.b64decode(message['data'])
                        await session.send_realtime_input(
                            video=types.Blob(data=image_bytes, mime_type="image/jpeg")
                        )
                    elif message['type'] == 'text':
                        # 텍스트 메시지 (RAG 검색 등에 활용 가능)
                        pass
                        
            except WebSocketDisconnect:
                print("🔌 Client Disconnected")
            except Exception as e:
                print(f"Receive Error: {e}")

        # [Task 2] Gemini -> WebSocket (Send to Flutter)
        async def send_to_flutter():
            try:
                while True:
                    async for response in session.receive():
                        if response.server_content:
                            model_turn = response.server_content.model_turn
                            if model_turn:
                                for part in model_turn.parts:
                                    # 오디오 데이터
                                    if part.inline_data:
                                        audio_b64 = base64.b64encode(part.inline_data.data).decode('utf-8')
                                        await websocket.send_json({
                                            "type": "audio",
                                            "data": audio_b64
                                        })
                                    
                                    # 텍스트 데이터
                                    if part.text:
                                        await websocket.send_json({
                                            "type": "text",
                                            "data": part.text
                                        })
                                        
                                    # 턴 종료 시그널 (필요하면 전송)
                                    # if getattr(response.server_content, "turn_complete", False): ...

            except Exception as e:
                print(f"Send Error: {e}")

        # 태스크 실행
        await asyncio.gather(receive_from_flutter(), send_to_flutter())


# ==========================================
# [클래스] DB 로그 저장 (Firebase Realtime Database)
# ==========================================
class DatabaseLogger:
    def __init__(self, cred_path=None, database_url=None):
        # 1. 우선순위: 인자 -> 환경변수 -> 기본값(serviceAccountKey.json)
        self.cred_path = cred_path or os.getenv("FIREBASE_CRED_PATH")
        
        # 기본값 설정 로직
        if not self.cred_path:
            # 현재 파일 위치 기준
            current_dir = pathlib.Path(__file__).parent.absolute()
            possible_path = current_dir / "serviceAccountKey.json"
            if possible_path.exists():
                self.cred_path = str(possible_path)
            else:
                # 프로젝트 루트 등 다른 위치 시도 (필요시)
                # 여기서는 vision 폴더 루트 가정
                 possible_path_root = current_dir.parents[1] / "serviceAccountKey.json" # flask/기능/실시간비전 -> vision/
                 if possible_path_root.exists():
                     self.cred_path = str(possible_path_root)
        
        self.database_url = database_url or os.getenv("FIREBASE_DB_URL")
        
        self.buffer = []
        self.session_id = None
        self._init_firebase()
        self._start_session()

    def _init_firebase(self):
        """Firebase 초기화"""
        try:
            # 이미 초기화되었는지 확인
            if not firebase_admin._apps:
                cred = credentials.Certificate(self.cred_path)
                firebase_admin.initialize_app(cred, {
                    'databaseURL': self.database_url
                })
                print("🔥 Firebase 연결 성공!")
            else:
                print("🔥 Firebase 이미 연결됨")
        except Exception as e:
            print(f"❌ Firebase 초기화 오류: {e}")
            print("⚠️ firebase_key.json 파일과 database_url을 확인해주세요.")

    def _start_session(self):
        """새로운 대화 세션 시작"""
        try:
            # 세션 ID 생성 (타임스탬프 기반)
            self.session_id = str(int(time.time()))
            session_ref = db.reference(f'sessions/{self.session_id}')
            
            session_data = {
                'start_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                'model_id': MODEL_ID
            }
            session_ref.set(session_data)
            print(f"💾 Firebase 세션 시작됨: ID {self.session_id}")
        except Exception as e:
            print(f"❌ 세션 시작 오류: {e}")

    def append_text(self, text):
        self.buffer.append(text)

    def log_user_message(self, text):
        """사용자 메시지 저장"""
        try:
            if self.session_id:
                messages_ref = db.reference(f'sessions/{self.session_id}/messages')
                new_message_ref = messages_ref.push() # 고유 키 생성
                
                message_data = {
                    'sender': 'user',
                    'content': text,
                    'created_at': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                new_message_ref.set(message_data)
        except Exception as e:
            print(f"\n⚠️ Firebase 저장 실패 (User): {e}")

    def flush_model_turn(self):
        """모델 응답 저장"""
        if not self.buffer: return
        
        full_text = "".join(self.buffer)
        
        try:
            if self.session_id:
                messages_ref = db.reference(f'sessions/{self.session_id}/messages')
                new_message_ref = messages_ref.push()
                
                message_data = {
                    'sender': 'gemini',
                    'content': full_text,
                    'created_at': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                new_message_ref.set(message_data)
        except Exception as e:
            print(f"\n⚠️ Firebase 저장 실패 (Gemini): {e}")
            
        self.buffer = []
    
    def save_feedback(self, score):
        """피드백 저장"""
        try:
            if self.session_id:
                session_ref = db.reference(f'sessions/{self.session_id}')
                session_ref.update({
                    'feedback': score
                })
                print("✅ Firebase에 피드백 저장 완료!")
        except Exception as e:
            print(f"❌ 피드백 저장 오류: {e}")

# ==========================================
# [클래스] STT 처리기 (백그라운드 스레드)
# ==========================================
class SpeechTranscriber:
    def __init__(self, logger, shared_state=None):
        self.logger = logger
        self.shared_state = shared_state
        self.audio_queue = queue.Queue()
        self.running = True
        self.recognizer = sr.Recognizer()
        
        # STT 설정
        self.energy_threshold = 1000  # 음성 감지 임계값 (조절 필요)
        self.pause_threshold = 0.8    # 말 끊김 간주 시간 (초)
        self.sample_rate = 16000
        self.sample_width = 2         # 16-bit = 2 bytes

        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
    
    def add_audio(self, data):
        if self.running:
            self.audio_queue.put(data)
            
    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)

    def _process_loop(self):
        print("👂 STT 리스너 시작 (한국어)")
        
        audio_buffer = bytearray()
        silence_frames = 0
        has_voice = False
        
        # 1 프레임(청크) 당 시간 계산
        # CHUNK_SIZE(512) / RATE(16000) = 0.032초
        chunk_duration = 512 / 16000
        pause_frame_count = int(self.pause_threshold / chunk_duration)
        
        while self.running:
            try:
                # 큐에서 오디오 청크 가져오기 (타임아웃 1초)
                data = self.audio_queue.get(timeout=1.0)
                
                # 에너지(소리 크기) 계산
                rms = audioop.rms(data, self.sample_width)
                
                if rms > self.energy_threshold:
                    has_voice = True
                    silence_frames = 0
                else:
                    if has_voice:
                        silence_frames += 1
                
                # 버퍼에 데이터 추가
                if has_voice:
                    audio_buffer.extend(data)
                
                # 말이 끝났다고 판단되면 (일정 시간 침묵)
                if has_voice and silence_frames > pause_frame_count:
                    # 인식 수행
                    self._recognize(audio_buffer)
                    
                    # 초기화
                    audio_buffer = bytearray()
                    silence_frames = 0
                    has_voice = False
                    
                # 버퍼가 너무 커지면 (예: 15초 이상) 강제 인식 (메모리 보호)
                if len(audio_buffer) > 16000 * 2 * 15:
                    self._recognize(audio_buffer)
                    audio_buffer = bytearray()
                    silence_frames = 0
                    has_voice = False

            except queue.Empty:
                continue
            except Exception as e:
                print(f"STT 루프 오류: {e}")
                
    def _recognize(self, audio_data):
        if len(audio_data) < 16000 * 2 * 0.5: # 0.5초 미만은 무시
            return
            
        try:
            # Raw PCM 데이터를 AudioData 객체로 변환
            audio_source = sr.AudioData(bytes(audio_data), self.sample_rate, self.sample_width)
            
            # Google Web Speech API 호출 (동기)
            text = self.recognizer.recognize_google(audio_source, language="ko-KR")
            if text.strip():
                print(f"\n[🗣️ User]: {text}")
                self.logger.log_user_message(text)
                
                # shared_state 접근이 어려우므로 로거를 통해 우회하거나 전역 변수 고려
                # 여기서는 간단히 전역 shared_state가 없으므로 생략하거나 
                # SpeechTranscriber에 shared_state 참조를 넘겨주는 것이 좋음
                if hasattr(self, 'shared_state') and self.shared_state:
                     self.shared_state["display_text"] = "..."
                
        except sr.UnknownValueError:
            # 인식 실패 (잡음 등) - 조용히 넘어감
            pass
        except sr.RequestError as e:
            print(f"STT API 오류: {e}")
        except Exception as e:
            print(f"STT 처리 중 오류: {e}")

# ==========================================
# [메인] 실행 루프
# ==========================================

async def main():
    try:
        client = genai.Client(api_key=API_KEY)
        config = get_config()
        
        # Supabase RAG 초기화
        rag_engine = SupabaseRAG(client)
        rag_queue = asyncio.Queue()
        
        p = pyaudio.PyAudio()
        
        input_stream = None
        output_stream = None

        try:
            output_stream = p.open(format=AUDIO_FORMAT, channels=CHANNELS, rate=OUTPUT_RATE, output=True)
            input_stream = p.open(format=AUDIO_FORMAT, channels=CHANNELS, rate=INPUT_RATE, input=True, 
                                  input_device_index=MIC_DEVICE_INDEX, frames_per_buffer=CHUNK_SIZE)
        except Exception as e:
            print(f"❌ 오디오 초기화 오류: {e}")
            return

        # USB 웹캠 초기화 (Windows DirectShow 백엔드 사용)
        cap = None
        camera_index = 0
        max_cameras = 5  # 최대 5개까지 시도
        
        print("📹 USB 웹캠 연결 시도 중...")
        for i in range(max_cameras):
            # Windows에서 DirectShow 백엔드 사용 (USB 웹캠에 더 안정적)
            try:
                # DirectShow 백엔드 사용 (Windows 전용)
                test_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            except:
                # CAP_DSHOW가 없으면 기본 백엔드 사용
                test_cap = cv2.VideoCapture(i)
            
            if test_cap.isOpened():
                # 프레임 읽기 테스트
                ret, frame = test_cap.read()
                if ret and frame is not None:
                    # USB 웹캠인지 확인 (일반적으로 외부 웹캠은 인덱스 1 이상)
                    # 또는 첫 번째로 성공한 웹캠 사용
                    cap = test_cap
                    camera_index = i
                    print(f"✅ USB 웹캠 연결 성공! (인덱스: {i})")
                    # 웹캠 이름 정보 출력 (가능한 경우)
                    try:
                        backend = test_cap.getBackendName()
                        print(f"   백엔드: {backend}")
                    except:
                        pass
                    break
                else:
                    test_cap.release()
            else:
                test_cap.release()
        
        if cap is None or not cap.isOpened():
            print("❌ USB 웹캠을 찾을 수 없습니다.")
            print("   가능한 해결 방법:")
            print("   1. USB 웹캠이 연결되어 있는지 확인")
            print("   2. 다른 프로그램에서 웹캠을 사용 중인지 확인 (Zoom, Teams, 카메라 앱 등)")
            print("   3. 웹캠 권한을 확인 (Windows 설정 > 개인 정보 > 카메라)")
            print("   4. USB 포트를 다른 포트로 변경해보세요")
            print("   5. 웹캠 드라이버가 설치되어 있는지 확인 (장치 관리자)")
            sys.exit(1)
        
        # 웹캠 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        
        # 실제 설정된 해상도 확인
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"📐 웹캠 해상도: {actual_width}x{actual_height}")
        
        # 웹캠이 실제로 프레임을 읽을 수 있는지 테스트
        print("🔍 웹캠 프레임 읽기 테스트 중...")
        time.sleep(0.5)  # 웹캠 초기화 대기
        test_ret, test_frame = cap.read()
        if not test_ret or test_frame is None:
            print("❌ 웹캠에서 프레임을 읽을 수 없습니다.")
            print("   가능한 해결 방법:")
            print("   1. 다른 프로그램에서 웹캠을 사용 중인지 확인 (Zoom, Teams, 카메라 앱 등)")
            print("   2. 웹캠을 다시 연결해보세요")
            print("   3. Windows 설정 > 개인 정보 > 카메라 권한 확인")
            cap.release()
            sys.exit(1)
        else:
            print(f"✅ 웹캠 프레임 읽기 성공! (프레임 크기: {test_frame.shape})")

        if not cap.isOpened():
            print("❌ 웹캠을 찾을 수 없습니다.")
            return

        shared_state = {
            "latest_frame": None, 
            "running": True
        }

        # [중요] Firebase 로거 초기화
        logger = FirebaseLogger()
        
        # [중요] Supabase RAG 초기화
        rag_engine = SupabaseRAG(client)
        rag_queue = asyncio.Queue()

        print(f"\n🚀 모델({MODEL_ID}) 연결 중...")

        async with client.aio.live.connect(model=MODEL_ID, config=config) as session:
            print("✅ 연결 성공! (종료: q)")
            
            # [Task 1] 화면 표시 (Clean View)
            async def display_loop():
                frame_error_count = 0
                max_errors = 10
                
                while shared_state["running"]:
                    ret, frame = cap.read()
                    
                    if not ret or frame is None:
                        frame_error_count += 1
                        if frame_error_count > max_errors:
                            print("❌ 웹캠 프레임 읽기 실패가 계속됩니다. 웹캠을 확인해주세요.")
                            shared_state["running"] = False
                            break
                        await asyncio.sleep(0.1)
                        continue
                    
                    # 프레임 읽기 성공 시 에러 카운터 리셋
                    frame_error_count = 0
                    
                    shared_state["latest_frame"] = frame.copy()
                    cv2.imshow('Gemini Live Vision', frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        shared_state["running"] = False
                        break
                    await asyncio.sleep(0.01)

            # [Task 3] 오디오 입력 (User)
            async def send_audio():
                while shared_state["running"]:
                    try:
                        data = await asyncio.to_thread(input_stream.read, CHUNK_SIZE, exception_on_overflow=False)
                        
                        # [수정] 봇이 말하고 있을 때는 마이크 입력을 모델에 보내지 않음 (Self-Interruption 방지)
                        if audio_player.is_playing:
                            continue

                        await session.send_realtime_input(audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000"))
                    except Exception: break

# [Task 4] 응답 수신 (생각 프로세스 숨기기 적용)
            async def receive_response():
                print("   👂 응답 대기 중...")
                response_count = 0
                while shared_state["running"]:
                    try:
                        async for response in session.receive():
                            response_count += 1
                            
                            # [추가] 음성 인식 이벤트 처리 - 사용자 음성을 텍스트로 저장
                            # Gemini Live API의 response 구조 확인
                            speech_recognition = None
                            
                            # 방법 1: response의 직접 속성
                            if hasattr(response, 'speech_recognition_event'):
                                speech_recognition = response.speech_recognition_event
                            
                            # 방법 2: server_content 안에 있을 수도 있음
                            if speech_recognition is None:
                                server_content_temp = getattr(response, 'server_content', None)
                                if server_content_temp:
                                    if hasattr(server_content_temp, 'speech_recognition_event'):
                                        speech_recognition = server_content_temp.speech_recognition_event
                            
                            # 방법 3: response 객체 전체 구조 확인 (처음 몇 번만)
                            if response_count <= 3:
                                print(f"🔍 [디버그 #{response_count}] response 타입: {type(response)}")
                                response_attrs = [attr for attr in dir(response) if not attr.startswith('_')]
                                print(f"🔍 [디버그] response 속성: {response_attrs[:10]}...")  # 처음 10개만
                            
                            if speech_recognition:
                                print(f"🔍 [디버그] speech_recognition 이벤트 발견! 타입: {type(speech_recognition)}")
                                # transcript 속성 확인 (다양한 가능한 속성 이름 시도)
                                recognized_text = None
                                
                                # 가능한 속성 이름들 시도
                                for attr_name in ['transcript', 'text', 'content', 'message']:
                                    if hasattr(speech_recognition, attr_name):
                                        attr_value = getattr(speech_recognition, attr_name)
                                        if attr_value:
                                            recognized_text = str(attr_value)
                                            print(f"🔍 [디버그] '{attr_name}' 속성에서 텍스트 발견: {recognized_text[:50]}")
                                            break
                                
                                # 속성을 찾지 못했다면 모든 속성 출력
                                if recognized_text is None:
                                    speech_attrs = [attr for attr in dir(speech_recognition) if not attr.startswith('_')]
                                    print(f"🔍 [디버그] speech_recognition 속성: {speech_attrs}")
                                    # 값이 있는 속성만 출력
                                    for attr in speech_attrs:
                                        try:
                                            value = getattr(speech_recognition, attr)
                                            if value and not callable(value):
                                                print(f"   - {attr}: {value}")
                                        except:
                                            pass
                                
                                if recognized_text and recognized_text.strip():
                                    # is_final이 True일 때만 최종 인식된 텍스트를 저장
                                    is_final = getattr(speech_recognition, 'is_final', False)
                                    print(f"🔍 [디버그] 인식된 텍스트: '{recognized_text}', is_final: {is_final}")
                                    if is_final:
                                        print(f"\n🎤 [사용자 음성 인식] {recognized_text}")
                                        logger.log_message('user', recognized_text.strip())
                                    else:
                                        # 중간 인식 결과는 화면에만 표시 (저장하지 않음)
                                        print(f"\r🎤 [인식 중...] {recognized_text}", end="", flush=True)

                            server_content = response.server_content
                            if server_content is None:
                                continue

                            # server_content의 모든 속성 확인 (처음 몇 번만)
                            if response_count <= 3:
                                server_attrs = [attr for attr in dir(server_content) if not attr.startswith('_')]
                                print(f"🔍 [디버그 #{response_count}] server_content 속성: {server_attrs}")
                                # 주요 속성들의 값 확인
                                for attr in ['transcript', 'model_turn', 'turn_complete', 'speech_recognition_event', 'input_transcription', 'output_transcription']:
                                    if hasattr(server_content, attr):
                                        value = getattr(server_content, attr)
                                        print(f"   - {attr}: {type(value)} = {str(value)[:100] if value else 'None'}")

                            # [핵심] output_audio_transcription에서 AI 응답 텍스트 추출
                            output_transcription = getattr(server_content, 'output_transcription', None)
                            if output_transcription:
                                transcript_text = getattr(output_transcription, 'text', None)
                                if transcript_text:
                                    print(f"🔍 [디버그] output_transcription.text 발견: '{transcript_text[:50]}...'")
                                    print(transcript_text, end="", flush=True)
                                    logger.append_text(transcript_text)

                            # [핵심] input_audio_transcription에서 사용자 음성 텍스트 추출
                            input_transcription = getattr(server_content, 'input_transcription', None)
                            if input_transcription:
                                input_text = getattr(input_transcription, 'text', None)
                                # is_final 속성 확인 (최종 결과만 저장)
                                is_final = getattr(input_transcription, 'is_final', True)  # 기본값은 True
                                
                                if input_text and input_text.strip() and is_final:
                                    # 중복 저장 방지 (같은 텍스트가 연속으로 오는 경우)
                                    if input_text.strip() != logger.last_user_text:
                                        print(f"🔍 [디버그] input_transcription.text 발견: '{input_text[:50]}...'")
                                        print(f"\n🎤 [사용자 음성 인식] {input_text}")
                                        logger.log_message('user', input_text.strip())
                                        logger.last_user_text = input_text.strip()
                                    else:
                                        print(f"🔍 [디버그] input_transcription.text 중복 (저장 생략): '{input_text[:50]}...'")
                                elif input_text and input_text.strip() and not is_final:
                                    # 중간 인식 결과는 화면에만 표시
                                    print(f"\r🎤 [인식 중...] {input_text}", end="", flush=True)

                            model_turn = server_content.model_turn
                            if model_turn:
                                parts = getattr(model_turn, 'parts', [])
                                print(f"🔍 [디버그 #{response_count}] model_turn 발견! parts 개수: {len(parts)}")
                                
                                for idx, part in enumerate(parts):
                                    # [핵심 수정] "생각(Thought)" 데이터면 출력하지 않고 건너뜀
                                    # google-genai 최신 버전에서는 part.thought 속성으로 구분 가능
                                    is_thought = getattr(part, "thought", False)
                                    if is_thought:
                                        print(f"🔍 [디버그] part[{idx}]는 생각(thought)이므로 건너뜀")
                                        continue

                                    # 1. 텍스트 데이터 처리 (우선 처리 - 텍스트가 있으면 저장)
                                    part_text = getattr(part, 'text', None)
                                    if part_text:
                                        print(f"🔍 [디버그] part[{idx}] 텍스트 발견: '{part_text[:50]}...'")
                                        print(part_text, end="", flush=True)
                                        logger.append_text(part_text)

                                    # 2. 오디오 데이터 처리
                                    inline_data = getattr(part, 'inline_data', None)
                                    if inline_data:
                                        print(f"🔍 [디버그] part[{idx}] 오디오 데이터 발견 (크기: {len(inline_data.data)} bytes)")
                                        audio_player.add_audio(inline_data.data)
                                    
                                    # 텍스트도 오디오도 없는 경우 - 모든 속성 확인
                                    if not part_text and not inline_data:
                                        print(f"🔍 [디버그 #{response_count}] part[{idx}]에는 텍스트와 오디오가 모두 없음")
                                        # part의 모든 속성 확인
                                        part_attrs = [attr for attr in dir(part) if not attr.startswith('_')]
                                        print(f"   part 속성: {part_attrs}")
                                        # 각 속성의 값 확인
                                        for attr in part_attrs[:15]:  # 처음 15개만
                                            try:
                                                value = getattr(part, attr)
                                                if not callable(value):
                                                    print(f"   - {attr}: {type(value)} = {str(value)[:80] if value else 'None'}")
                                            except:
                                                pass
                            else:
                                # model_turn이 없는 경우도 로그
                                if response_count <= 5:
                                    print(f"🔍 [디버그 #{response_count}] model_turn이 없습니다. server_content 속성 재확인:")
                                    server_attrs = [attr for attr in dir(server_content) if not attr.startswith('_')]
                                    for attr in server_attrs:
                                        try:
                                            value = getattr(server_content, attr)
                                            if not callable(value) and value is not None:
                                                print(f"   - {attr}: {type(value)} = {str(value)[:80]}")
                                        except:
                                            pass

                            # 3. 턴 종료 신호 처리
                            if server_content.turn_complete:
                                print(f"\n🔍 [디버그] turn_complete 신호 수신! (response_count: {response_count}, 버퍼 길이: {len(logger.current_turn_text)})")
                                # turn_complete 전까지 받은 모든 응답 요약
                                if len(logger.current_turn_text) == 0:
                                    print(f"⚠️ [경고] turn_complete까지 총 {response_count}개의 응답을 받았지만 버퍼가 비어있습니다!")
                                    print(f"   - transcript: {transcript is not None}")
                                    print(f"   - model_turn: {model_turn is not None}")
                                    if model_turn:
                                        parts = getattr(model_turn, 'parts', [])
                                        print(f"   - parts 개수: {len(parts)}")
                                print("\n") 
                                logger.flush_model_turn()

                    except Exception as e:
                        print(f"⚠️ 응답 수신 루프 에러: {e}")
                        await asyncio.sleep(1)


            # [Task 5] RAG 검색 및 컨텍스트 주입
            async def rag_loop():
                while shared_state["running"]:
                    try:
                        # 큐에서 텍스트 꺼내기 (없으면 대기하지 않고 넘어감 -> timeout)
                        # wait for user input
                        try:
                            text = await asyncio.wait_for(rag_queue.get(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue

                        print(f"   ... 🔎 매뉴얼 검색 중: {text[:20]}...")
                        # Supabase 검색 (동기 함수이므로 스레드로 실행)
                        results = await asyncio.to_thread(rag_engine.search, text)
                        
                        if results:
                            context_text = "\n".join(results)
                            msg = f"참고 매뉴얼 정보 (User Question: {text}):\n{context_text}"
                            print(f"   ✅ 검색 성공 ({len(results)}건) -> 모델에 주입")
                            
                            # 모델에게 텍스트로 정보 전달 (end_of_turn=False로 설정하여 답변 강제 트리거 방지)
                            # 하지만 Live API에서는 텍스트를 보내면 모델이 읽고 반응할 수 있음
                            await session.send(input=msg, end_of_turn=False)
                        else:
                            print("   ⚠️ 검색 결과 없음")
                            
                    except Exception as e:
                        print(f"RAG Loop Error: {e}")
                    
                    await asyncio.sleep(0.1)

            # [Task 6] Command Watcher
            async def command_watcher():
                # 세션이 생성되지 않았으면 종료
                if not logger.session_ref:
                    print("⚠️ 세션이 생성되지 않아 command_watcher를 시작할 수 없습니다.")
                    return

                current_session_id = logger.session_ref.id
                # Firestore 참조
                db_client = firestore.client()
                session_doc_ref = db_client.collection('sessions').document(current_session_id)
                
                while shared_state["running"]:
                    try:
                        # polling 방식으로 1초마다 확인 (Listen보다 async 충돌 위험이 적음)
                        doc = session_doc_ref.get()
                        command = None
                        if doc.exists:
                            command = doc.to_dict().get('command')
                        
                        if command == "summarize":
                            # 요약 로직 실행 (비동기)
                            await perform_summarization(client, current_session_id)
                        
                        await asyncio.sleep(1.0) # 1초 대기
                    except Exception as e:
                        print(f"Command Watcher Error: {e}")
                        await asyncio.sleep(1.0)                    

            # [Task 7] FastAPI Server (Spring Boot 연동)
            config = uvicorn.Config(app=app, host="0.0.0.0", port=8000, log_level="info")
            server = uvicorn.Server(config)

            tasks = [
                asyncio.create_task(display_loop()),
                asyncio.create_task(send_video()),
                asyncio.create_task(send_audio()),
                asyncio.create_task(receive_response()),
                asyncio.create_task(rag_loop()),
                asyncio.create_task(command_watcher()),
                asyncio.create_task(server.serve())
            ]
            
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending: task.cancel()
            
            # [종료 시퀀스] 사용자 피드백 수집
            print("\n" + "="*40)
            print("👋 상담이 종료되었습니다.")
            try:
                feedback = input("💡 이번 상담이 도움이 되셨나요? (y/n): ").strip().lower()
                feedback_score = 1 if feedback == 'y' else 0
                
                # 마지막 세션 ID 가져오기 및 피드백 업데이트
                if logger.session_id:
                    logger.save_feedback(feedback_score)
            except Exception as e:
                print(f"피드백 저장 오류: {e}")
            print("="*40 + "\n")

            if cap.isOpened(): cap.release()
            if input_stream: input_stream.stop_stream(); input_stream.close()
            if output_stream: output_stream.stop_stream(); output_stream.close()
            if p: p.terminate()
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"\n❌ 메인 오류: {e}")
        input("엔터를 누르면 종료합니다...")

if __name__ == "__main__":
    asyncio.run(main())