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


# [Firebase 라이브러리 추가]
try:
    import firebase_admin
    from firebase_admin import credentials, firestore  # pyright: ignore[reportMissingImports]
except ImportError:
    print("❌ firebase-admin이 설치되지 않았습니다. 'pip install firebase-admin'을 실행하세요.")
    sys.exit(1)

# [Gemini 라이브러리]
try:
    from google import genai
    from google.genai import types  # pyright: ignore[reportMissingImports]
except ImportError:
    print("❌ google-genai 라이브러리가 설치되지 않았습니다.")
    sys.exit(1)

# [Supabase 라이브러리 추가]
try:
    from supabase import create_client, Client
except ImportError:
    print("❌ supabase 라이브러리가 설치되지 않았습니다. 'pip install supabase'를 실행하세요.")
    sys.exit(1)

warnings.filterwarnings("ignore")

# ==========================================
API_KEY = os.getenv("GEMINI_API_KEY")

# Firebase 키 경로 설정 (환경변수 우선, 없으면 현재 디렉토리의 FirebaseAdmin.json 사용)
project_root = pathlib.Path(__file__).parent.parent.absolute()
current_dir = pathlib.Path(__file__).parent.absolute()

# 우선순위 1: 환경변수
# 우선순위 2: vision 폴더 내 FirebaseAdmin.json
# 우선순위 3: 프로젝트 루트의 serviceAccountKey.json
default_key_path = current_dir / "FirebaseAdmin.json"
FIREBASE_KEY_PATH = os.getenv("FIREBASE_KEY_PATH", str(default_key_path))

# Realtime Database URL (Firestore 사용 시 불필요하지만 참고용으로 남김/삭제 가능)
# FIREBASE_DATABASE_URL = "https://lgdx-6054d-default-rtdb.asia-southeast1.firebasedatabase.app/"

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
    except Exception:
        pass

load_environment()

API_KEY = os.getenv("GEMINI_API_KEY")

# Firebase 키 경로 설정 로직 개선
project_root = pathlib.Path(__file__).parent.parent.absolute()
default_firebase_path = project_root / "serviceAccountKey.json"
FIREBASE_KEY_PATH = '/Users/harry/LG DX SCHOOL/lgdx_backend/vision/serviceAccountKey.json'

if not API_KEY:
    print("❌ GEMINI_API_KEY가 없습니다. .env 파일을 확인해주세요.")
    sys.exit(1)

if not FIREBASE_KEY_PATH:
    print(f"❌ Firebase 키 파일을 찾을 수 없습니다.")
    print(f"   검색 위치 1: {current_dir / 'FirebaseAdmin.json'}")
    print(f"   검색 위치 2: {project_root / 'Firebase.json'}")
    sys.exit(1)

MODEL_ID = "gemini-2.5-flash-native-audio-preview-09-2025"

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
        if not self.session_ref: return
        try:
            # 해당 세션의 'messages' 컬렉션에 대화 추가
            self.session_ref.collection('messages').add({
                'sender': sender,      # 'user' or 'gemini'
                'content': text,
                'created_at': int(time.time() * 1000)
            })
        except Exception as e:
            print(f"⚠️ 로그 저장 실패: {e}")

    def append_text(self, text):
        """스트리밍되는 텍스트 조각을 임시 버퍼에 추가"""
        self.current_turn_text += text

    def flush_model_turn(self):
        """버퍼에 모인 텍스트를 한 번에 로그로 저장하고 초기화"""
        if self.current_turn_text.strip():
            self.log_message('gemini', self.current_turn_text)
            self.current_turn_text = ""

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
        # .env에서 로드할 키 이름을 사용자 설정에 맞춤
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("supbase_service_role") 
        self.client = None
        
        if self.supabase_url and self.supabase_key:
            try:
                self.client = create_client(self.supabase_url, self.supabase_key)
                print(f"🔥 Supabase 하이브리드 엔진 연결 성공!")
            except Exception as e:
                print(f"❌ Supabase 초기화 오류: {e}")
        else:
            print("❌ Supabase URL 또는 Key(supbase_service_role)를 찾을 수 없습니다.")

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
# [클래스] 비동기 오디오 플레이어
# ==========================================
class AsyncAudioPlayer:
    def __init__(self):
        self.queue = queue.Queue()
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=AUDIO_FORMAT,
            channels=CHANNELS,
            rate=OUTPUT_RATE,
            output=True
        )
        self.running = True
        self.is_playing = False
        self.thread = threading.Thread(target=self._play_loop, daemon=True)
        self.thread.start()

    def _play_loop(self):
        while self.running:
            try:
                data = self.queue.get(timeout=0.05)
                self.is_playing = True
                self.stream.write(data)
            except queue.Empty:
                self.is_playing = False
                continue
            except Exception:
                pass

    def add_audio(self, data):
        self.queue.put(data)

    def close(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

# ==========================================
# [설정] Config
# ==========================================
def get_config():
    current_dir = pathlib.Path(__file__).parent.absolute()
    persona_path = current_dir / "persona/persona_세탁법.txt"
    
    system_instruction = "너는 도움이 되는 LG전자의 AI 어시스턴트야."
    if persona_path.exists():
        try:
            system_instruction = persona_path.read_text(encoding="utf-8")
        except Exception:
            pass

    return {
        "response_modalities": ["AUDIO"], 
        "speech_config": {
            "voice_config": {
                "prebuilt_voice_config": {
                    "voice_name": "Kore" # 목소리 바꾸기
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
# [메인] 실행 루프
# ==========================================
async def main():
    try:
        client = genai.Client(api_key=API_KEY)
        config = get_config()
        
        p = pyaudio.PyAudio()
        input_stream = p.open(format=AUDIO_FORMAT, channels=CHANNELS, rate=INPUT_RATE, input=True, frames_per_buffer=CHUNK_SIZE)
        audio_player = AsyncAudioPlayer()

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

        shared_state = {
            "latest_frame": None, 
            "running": True
        }

        # [중요] Firebase 로거 초기화
        logger = FirebaseLogger()
        
        # [중요] Supabase RAG 초기화
        rag_engine = SupabaseRAG(client)
        rag_queue = asyncio.Queue()

        def on_model_speak(text):
            print(f"[🤖 Gemini]: {text}")
            logger.log_message('gemini', text)
        print(f"\n🚀 모델({MODEL_ID}) 연결 중...")

        async with client.aio.live.connect(model=MODEL_ID, config=config) as session:
            print("✅ 연결 성공! (종료: q)")
            
            # [Task 1] 화면 표시 (Clean View)
            async def display_loop():
                while shared_state["running"]:
                    ret, frame = cap.read()
                    if not ret: break

                    shared_state["latest_frame"] = frame.copy()
                    cv2.imshow('Gemini Live Vision', frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        shared_state["running"] = False
                        break
                    await asyncio.sleep(0.01)

            # [Task 2] 비디오 전송
            async def send_video():
                while shared_state["running"]:
                    if shared_state["latest_frame"] is not None:
                        frame = cv2.resize(shared_state["latest_frame"], (640, 480))
                        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                        try:
                            await session.send_realtime_input(
                                video=types.Blob(data=buffer.tobytes(), mime_type="image/jpeg")
                            )
                        except Exception: pass
                    await asyncio.sleep(0.5)

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
                while shared_state["running"]:
                    try:
                        async for response in session.receive():
                            server_content = response.server_content
                            if server_content is None:
                                continue

                            model_turn = server_content.model_turn
                            if model_turn:
                                for part in model_turn.parts:
                                    
                                    # [핵심 수정] "생각(Thought)" 데이터면 출력하지 않고 건너뜀
                                    # google-genai 최신 버전에서는 part.thought 속성으로 구분 가능
                                    if getattr(part, "thought", False):
                                        continue

                                    # 1. 오디오 데이터 처리
                                    if part.inline_data:
                                        audio_player.add_audio(part.inline_data.data)

                                    # 2. 텍스트 데이터 처리 (생각이 아닌 실제 답변만 출력)
                                    if part.text:
                                        print(part.text, end="", flush=True)
                                        logger.append_text(part.text)

                            # 3. 턴 종료 신호 처리
                            if server_content.turn_complete:
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
                if not logger.session_ref:
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

    except Exception as e:
        print(f"오류 발생: {e}")
        traceback.print_exc()
    finally:
        if 'audio_player' in locals(): audio_player.close()
        if 'input_stream' in locals(): input_stream.stop_stream(); input_stream.close()
        if 'p' in locals(): p.terminate()
        if 'cap' in locals(): cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())