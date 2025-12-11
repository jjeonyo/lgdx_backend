import asyncio
import os
import cv2
import pathlib
from pathlib import Path
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
import numpy as np

# ensure project root is on sys.path when invoked via uvicorn
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


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
# .env 파일 로드 (프로젝트 루트에서 찾기)
# ==========================================
# 상위 폴더의 .env 파일 로드 (프로젝트 루트에 있는 경우)
project_root = pathlib.Path(__file__).parent.parent.absolute()
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)
# 현재 폴더의 .env도 시도 (하위 호환성)
load_dotenv()

# ==========================================
API_KEY = os.getenv("google_api")

# Firebase 키 경로 설정
project_root = pathlib.Path(__file__).parent.parent.absolute()
current_dir = pathlib.Path(__file__).parent.absolute()
FIREBASE_KEY_PATH = "C:\dxfirebasekey\serviceAccountKey.json"

# Realtime Database URL (Firestore 사용 시 불필요하지만 참고용으로 남김/삭제 가능)
# FIREBASE_DATABASE_URL = "https://lgdx-6054d-default-rtdb.asia-southeast1.firebasedatabase.app/"

MODEL_ID = "gemini-2.5-flash-native-audio-preview-09-2025"

# [오디오 설정]
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
INPUT_RATE = 16000
OUTPUT_RATE = 24000
CHUNK_SIZE = 512

# API 키 확인
if not API_KEY:
    print("❌ google_api가 없습니다. .env 파일을 확인해주세요.")
    sys.exit(1)

if not FIREBASE_KEY_PATH:
    print(f"❌ Firebase 키 파일을 찾을 수 없습니다.")
    print(f"   검색 위치 1: {current_dir / 'FirebaseAdmin.json'}")
    print(f"   검색 위치 2: {project_root / 'Firebase.json'}")
    sys.exit(1)




async def perform_summarization(client, session_id):
    """Firebase에서 대화를 가져와 요약하고 결과를 DB에 저장"""
    print(f"\n🔔 [Command Received] 요약 요청을 받았습니다. (Session: {session_id})")
    
    def _load_messages_sync():
        db_client = firestore.client()
        messages_ref = db_client.collection('sessions').document(session_id).collection('messages')
        docs = messages_ref.order_by('created_at').stream()
        messages_list = [doc.to_dict() for doc in docs]
        return messages_list

    def _update_summary_sync(summary_text: str):
        db_client = firestore.client()
        db_client.collection('sessions').document(session_id).update({
            'summary': summary_text,
            'command': None  # 명령 수행 완료 후 초기화 (중요)
        })

    try:
        # 동기 Firestore I/O는 스레드 오프로딩
        messages_list = await asyncio.to_thread(_load_messages_sync)

        if not messages_list:
            print("   ⚠️ 대화 내용이 없습니다.")
            return

        # 2. 텍스트 변환
        chat_context = ""
        for msg in messages_list:
            sender = msg.get('sender', 'unknown')
            text = msg.get('text', '')  # text 필드만 사용 (content 제거)
            chat_context += f"[{sender}]: {text}\n"

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
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        summary_text = resp.text.strip()
        print(f"   📝 요약 완료: {summary_text}")

        # 4. 결과 DB 저장 및 명령어 초기화 (동기 I/O 오프로딩)
        await asyncio.to_thread(_update_summary_sync, summary_text)

    except Exception as e:
        print(f"   ❌ 요약 중 에러 발생: {e}")



# ==========================================
# [클래스] Firebase Logger (Firestore 사용)
# ==========================================
class FirebaseLogger:
    def __init__(self):
        self.session_ref = None
        self.current_turn_text = ""
        self.last_user_text = ""  # 최근 사용자 발화 저장해 중복 로그 방지
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
            # chat_room 생성 또는 확인 (room_user_001 형식)
            user_id = 'user_001'  # 사용자 ID (나중에 실제 사용자 ID로 변경 가능)
            room_id = f'room_{user_id}'
            room_ref = self.db.collection('chat_rooms').document(room_id)
            
            # chat_room이 없으면 생성
            room_doc = room_ref.get()
            if not room_doc.exists:
                room_ref.set({
                    'user_id': user_id,
                    'created_at': firestore.SERVER_TIMESTAMP,
                    'last_message_at': firestore.SERVER_TIMESTAMP,
                })
                print(f"📄 새 chat_room 생성: {room_id}")
            else:
                print(f"📄 기존 chat_room 사용: {room_id}")
            
            # session_ref는 room_ref로 설정 (호환성을 위해 유지)
            self.session_ref = room_ref
        except Exception as e:
            print(f"❌ 세션 생성 실패: {e}")

    def log_message(self, sender, text):
        if not self.session_ref: return
        try:
            # 현재 시간 정보 생성
            current_timestamp = int(time.time() * 1000)  # 밀리초 단위 타임스탬프
            current_datetime = datetime.now()  # 현재 날짜/시간 객체
            
            # 읽기 쉬운 날짜/시간 형식 (한국 시간대 기준)
            # 예: "2024-01-15 14:30:25"
            formatted_time = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            
            # 시간대 정보 (한국 표준시)
            timezone = "KST"
            
            # chat_room ID 생성 (room_user_001 형식)
            user_id = 'user_001'  # 사용자 ID (나중에 실제 사용자 ID로 변경 가능)
            room_id = f'room_{user_id}'
            
            print(f"💾 [Firebase] 저장 시도 - sender: {sender}, text: {text[:50]}..., 시간: {formatted_time}")
            # chat_rooms/{room_id}/messages 컬렉션에 대화 추가
            # message_type: 라이브 대화는 모두 'live'로 저장
            doc_ref = self.db.collection('chat_rooms').document(room_id).collection('messages').add({
                'sender': sender,      # 'user' or 'gemini'
                'text': text,          # 메시지 내용 (통일된 필드명)
                'message_type': 'live',  # 라이브 대화는 모두 'live'로 저장
                'created_at': current_timestamp,  # 밀리초 단위 타임스탬프 (정렬/쿼리용)
                'timestamp': formatted_time,      # 읽기 쉬운 날짜/시간 형식
                'timezone': timezone              # 시간대 정보
            })
            
            # chat_room의 last_message_at 업데이트
            self.db.collection('chat_rooms').document(room_id).update({
                'last_message_at': firestore.SERVER_TIMESTAMP
            })
            
            print(f"✅ [Firebase] 저장 성공! (chat_rooms/{room_id}/messages) - sender: {sender}, text 길이: {len(text)}, 시간: {formatted_time}")
        except Exception as e:
            print(f"⚠️ 로그 저장 실패: {e}")

    def append_text(self, text):
        """스트리밍되는 텍스트 조각을 임시 버퍼에 추가"""
        self.current_turn_text += text

    def flush_model_turn(self):
        """버퍼에 모인 텍스트를 한 번에 로그로 저장하고 초기화"""
        if self.current_turn_text.strip():
            # 한국어만 포함하는지 확인 (영어만 있는 텍스트 필터링)
            if self._is_korean_text(self.current_turn_text):
                print(f"💾 [Firebase] AI 응답 저장 시도 - 길이: {len(self.current_turn_text)}")
                self.log_message('gemini', self.current_turn_text)
            else:
                print(f"⚠️ [Firebase] 영어만 포함된 AI 응답은 저장하지 않습니다: {self.current_turn_text[:50]}...")
            self.current_turn_text = ""
        else:
            # 버퍼가 비어있어도 경고만 출력하고 계속 진행 (오디오만 있고 텍스트가 없는 경우도 있음)
            print(f"⚠️ [Firebase] AI 응답 버퍼가 비어있습니다. (오디오만 전송되었을 수 있음)")
    
    def _is_korean_text(self, text):
        """한국어가 포함되어 있는지 확인 (영어만 있는 텍스트 필터링)"""
        import re
        # 한글 유니코드 범위: AC00-D7AF (가-힣), 1100-11FF (초성), 3130-318F (호환 자모)
        korean_pattern = re.compile(r'[가-힣ㄱ-ㅎㅏ-ㅣ]')
        has_korean = bool(korean_pattern.search(text))
        
        # 영어만 있는지 확인 (한국어가 없고 영어/숫자/공백/구두점만 있는 경우)
        if not has_korean:
            english_only_pattern = re.compile(r'^[a-zA-Z0-9\s\.,!?;:\-\'\"()]+$')
            if english_only_pattern.match(text.strip()):
                return False  # 영어만 있으면 저장하지 않음
        
        return has_korean  # 한국어가 포함되어 있으면 저장


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
        "response_modalities": ["AUDIO"],  # 오디오만 받기 (텍스트는 output_audio_transcription에서 추출)
        "input_audio_transcription": {},  # 입력 오디오를 텍스트로 변환 (한국어 자동 감지)
        "output_audio_transcription": {},  # 출력 오디오를 텍스트로 변환 (AI 응답)
        # 긴 답변이 중간에 끊기지 않도록 최대 토큰 상한을 넉넉히 설정
        "generation_config": {"max_output_tokens": 2048},
        "speech_config": {
            "voice_config": {
                "prebuilt_voice_config": {
                    "voice_name": "Kore" # 목소리 바꾸기
                }
            }
        },
        "system_instruction": system_instruction,
        # AutomaticActivityDetection 설정 추가
        # 참고: https://ai.google.dev/api/live?hl=ko#automaticactivitydetection
        # https://ai.google.dev/gemini-api/docs/live-guide?hl=ko
        # VAD (Voice Activity Detection) 설정으로 자연스러운 대화 유지
        "realtime_input_config": types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                # disabled: false (기본값) - 자동 활동 감지 활성화
                # 사용자가 말하는 동안 자동으로 활동을 감지하여 처리
                disabled=False,
                # prefixPaddingMs: 음성 시작이 커밋되기 전에 감지된 음성의 필수 길이 (밀리초)
                # 낮을수록 더 민감하지만 거짓양성 가능성 증가
                # 예제에서는 20ms 사용, 여기서는 300ms로 설정하여 안정성 확보
                prefix_padding_ms=300,
                # silenceDurationMs: 말의 끝이 커밋되기 전에 감지된 비언어(침묵)의 필수 시간 (밀리초)
                # 클수록 더 긴 음성 갭을 허용하지만 모델 지연 시간 증가
                # 자동 VAD가 음성 종료를 감지하려면 적절한 침묵 시간이 필요함
                # 너무 길면 응답이 늦어지고, 너무 짧으면 말하는 중에 끊길 수 있음
                # 예제에서는 100ms 사용, 여기서는 1000ms로 설정하여 자연스러운 대화 유지
                silence_duration_ms=1000  # 1초 침묵 후에 음성 종료로 간주 (자연스러운 대화를 위해)
                # start_of_speech_sensitivity와 end_of_speech_sensitivity는 기본값 사용 (enum 값이 존재하지 않을 수 있음)
            ),
            # ActivityHandling: 사용자 활동을 처리하는 방법
            # NO_INTERRUPTION: 모델의 응답이 중단되지 않음
            # 사용자가 말하는 동안에도 AI가 계속 말할 수 있음
            activity_handling=types.ActivityHandling.NO_INTERRUPTION
        )
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
            model="gemini-2.5-flash",
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
    
    # Firebase 로거 초기화 (WebSocket 세션별)
    logger = FirebaseLogger()
    
    # 최신 프레임만 유지하는 컨테이너 (프레임 드롭 전략)
    latest_image = {"data": None}
    last_send_time = {"ts": 0.0}

    # 사용자 발화 누적 버퍼 (끊어진 텍스트를 합쳐서 Firebase에 한 번에 저장)
    buffer_path = Path(__file__).parent / "user_buffer.txt"
    try:
        buffer_path.write_text("", encoding="utf-8")
    except Exception:
        pass
    user_buffer = {"text": ""}

    last_append_time = {"ts": 0.0}

    def append_user_buffer(text: str):
        text = (text or "").strip()
        if not text:
            return
        if user_buffer["text"]:
            user_buffer["text"] += " "
        user_buffer["text"] += text
        last_append_time["ts"] = time.time()
        try:
            buffer_path.write_text(user_buffer["text"], encoding="utf-8")
        except Exception:
            pass
        print(f"✅ [Buffer] 사용자 음성 텍스트 누적 완료: {user_buffer['text'][:50]}...")

    def flush_user_buffer():
        if not user_buffer["text"]:
            return
        try:
            logger.log_message('user', user_buffer["text"])
            print(f"💾 [Buffer] Firebase 저장 및 버퍼 초기화: {user_buffer['text']}")
        except Exception as e:
            print(f"⚠️ [Buffer] Firebase 저장 실패: {e}")
        user_buffer["text"] = ""
        try:
            buffer_path.write_text("", encoding="utf-8")
        except Exception:
            pass
    
    async def buffer_flush_loop():
        # 음성 입력이 끊긴 뒤 약간의 휴지기(예: 1초)가 지나면 버퍼를 Firebase에 저장
        try:
            while True:
                if user_buffer["text"] and (time.time() - last_append_time["ts"] > 1.0):
                    flush_user_buffer()
                await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f"⚠️ [Buffer] 자동 플러시 루프 오류: {e}")

    async with client.aio.live.connect(model=MODEL_ID, config=config) as session:
        print("✅ Gemini Live Session Started")

        # [Task 1] WebSocket -> Gemini (Receive from Flutter, binary 우선)
        async def receive_from_flutter():
            print("👂 [Receive] 코루틴 시작 - Flutter 바이너리/텍스트 수신 대기")
            try:
                while True:
                    try:
                        msg = await asyncio.wait_for(websocket.receive(), timeout=300.0)

                        # 바이너리(주로 JPEG 프레임) 처리: 최신 프레임 컨테이너에 덮어쓰기만
                        if msg.get("type") == "websocket.receive" and msg.get("bytes") is not None:
                            image_bytes = msg["bytes"]
                            latest_image["data"] = image_bytes
                            # 디버그: 크기/해시 확인 (과도한 로그 주의)
                            # print(f"[DBG] recv binary image bytes={len(image_bytes)}, md5={hashlib.md5(image_bytes).hexdigest()[:10]}")
                            continue

                        # 텍스트(JSON) 메시지 처리 (오디오/제어 신호)
                        if msg.get("type") == "websocket.receive" and msg.get("text") is not None:
                            data = msg["text"]
                            message = json.loads(data)

                            if message.get('type') == 'audio':
                                audio_bytes = base64.b64decode(message['data'])

                                if len(audio_bytes) < 320:  # 160 samples * 2 bytes
                                    continue

                                try:
                                    await session.send_realtime_input(
                                        audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000")
                                    )
                                    if not hasattr(receive_from_flutter, 'last_audio_log_time'):
                                        receive_from_flutter.last_audio_log_time = time.time()
                                    current_time = time.time()
                                    # if current_time - receive_from_flutter.last_audio_log_time >= 1.0:
                                    #     print(f"🎤 [Receive] 오디오 수신 및 전송: {len(audio_bytes)} bytes (16kHz PCM)")
                                    #     receive_from_flutter.last_audio_log_time = current_time
                                except Exception as e:
                                    print(f"⚠️ [Receive] 오디오 전송 실패: {e}")
                                    raise

                            elif message.get('type') == 'text':
                                # 텍스트 메시지 (필요 시 활용)
                                pass

                            elif message.get('type') == 'user_speech_end':
                                # 사용자 발화 종료 시점에 버퍼를 Firebase에 저장 후 턴 종료 신호 전송
                                try:
                                    flush_user_buffer()
                                    await session.send(input=".", end_of_turn=True)
                                    print("✅ [Receive] 사용자 말하기 종료 신호 수신 - 버퍼 저장 및 end_of_turn=True 전송 (AI 응답 시작)")
                                except Exception as e:
                                    print(f"⚠️ [Receive] end_of_turn/버퍼 처리 실패: {e}")

                            elif message.get('type') in ('close_diagnosis', 'exit_diagnosis'):
                                print("❌ [Receive] 진단 화면 종료 신호 수신 (X 버튼 클릭)")
                                try:
                                    flush_user_buffer()
                                    await session.send(input=".", end_of_turn=True)
                                    print("✅ [Receive] 턴 완료 신호 전송 (X 버튼으로 인한 강제 종료)")
                                    await websocket.send_json({"type": "turn_complete", "exit": True})
                                    print("✅ [Receive] Flutter에 종료 신호 전송 완료 (엘리홈으로 이동)")
                                except Exception as e:
                                    print(f"⚠️ [Receive] 진단 화면 종료 처리 실패: {e}")
                                    try:
                                        await websocket.send_json({"type": "turn_complete", "exit": True})
                                    except:
                                        pass
                        else:
                            # 알 수 없는 메시지 유형
                            continue

                    except asyncio.TimeoutError:
                        continue
                    except WebSocketDisconnect:
                        print("🔌 [Receive] Client Disconnected")
                        break
                    except Exception as e:
                        error_str = str(e)
                        if "1011" in error_str or "service is currently unavailable" in error_str.lower():
                            print("⚠️ [Receive] Gemini 서비스 불가(1011). 잠시 후 재시도하세요.")
                            try:
                                await websocket.close(code=1011, reason="service unavailable")
                            except Exception:
                                pass
                            break
                        if "disconnect" in error_str.lower() or "cannot call" in error_str.lower():
                            print(f"🔌 [Receive] 연결 종료 감지: {e}")
                            break
                        print(f"⚠️ [Receive] 메시지 처리 에러: {e}")
                        continue

            except WebSocketDisconnect:
                print("🔌 [Receive] Client Disconnected (외부)")
            except Exception as e:
                error_str = str(e)
                if "1011" in error_str or "service is currently unavailable" in error_str.lower():
                    print("⚠️ [Receive] Gemini 서비스 불가(1011). 잠시 후 재시도하세요.")
                    try:
                        await websocket.close(code=1011, reason="service unavailable")
                    except Exception:
                        pass
                    return
                if "disconnect" in error_str.lower() or "cannot call" in error_str.lower():
                    print(f"🔌 [Receive] 연결 종료: {e}")
                else:
                    print(f"❌ [Receive] WebSocket 에러: {e}")
                    import traceback
                    traceback.print_exc()

        # 최신 프레임만 일정 주기로 전송 (프레임 드롭 전략)
        async def image_sender_loop():
            # print("📸 [ImageLoop] 최신 프레임 전송 루프 시작")
            try:
                while True:
                    if websocket.client_state.name != "CONNECTED":
                        await asyncio.sleep(0.05)
                        continue

                    now = time.time()
                    # 최신 프레임이 있고 0.3초 이상 경과 시 전송 (약 3fps), 모션 게이팅 없이 즉시 전송
                    if latest_image["data"] is not None and (now - last_send_time["ts"] > 0.3):
                        frame = latest_image["data"]
                        latest_image["data"] = None  # 가장 최신만 남기고 나머지 드롭
                        last_send_time["ts"] = now

                        try:
                            await session.send_realtime_input(
                                video=types.Blob(data=frame, mime_type="image/jpeg")
                            )
                            # print(f"📸 프레임 전송 ({len(frame)} bytes)")
                        except Exception as e:
                            print(f"⚠️ 프레임 전송 실패: {e}")

                    await asyncio.sleep(0.01)
            except Exception as e:
                print(f"❌ ImageLoop Error: {e}")

        # [Task 2] Gemini -> WebSocket (Send to Flutter)
        # 공식 예제 패턴: response.data를 바로 큐에 넣고 스트리밍
        async def send_to_flutter():
            print("📡 [Send] 코루틴 시작 - Gemini 응답 수신 및 Flutter 전송 대기")

            # 누적 바이트 추적용 변수
            send_to_flutter.total_audio_bytes = 0
            # X 버튼 클릭 여부 추적 (턴 완료 신호를 조건부로 보내기 위해)
            send_to_flutter.should_send_turn_complete = True
            try:
                while True:
                    try:
                        # WebSocket 연결 상태 확인 (조용히 대기)
                        if websocket.client_state.name != "CONNECTED":
                            await asyncio.sleep(0.1)
                            continue
                        
                        # 공식 예제 패턴: session.receive()를 직접 사용 (오디오 스트리밍)
                        # 중요: 1011 에러를 처리하기 위해 try-except로 감싸기
                        try:
                            async for response in session.receive():
                                # WebSocket 연결 상태 재확인
                                if websocket.client_state.name != "CONNECTED":
                                    # 연결 끊김은 과도하게 로그하지 않고 짧게 표시
                                    await asyncio.sleep(0.1)
                                    continue
                                
                                # 공식 예제 패턴: response.data를 바로 전송 (24kHz PCM 오디오 스트리밍)
                                # 공식 문서: "Output is 24kHz" - response.data는 24kHz PCM 오디오
                                if response.data is not None:
                                    try:
                                        audio_b64 = base64.b64encode(response.data).decode('utf-8')
                                        send_to_flutter.total_audio_bytes += len(response.data)
                                        await asyncio.wait_for(
                                            websocket.send_json({
                                                "type": "audio",
                                                "data": audio_b64
                                            }),
                                            timeout=5.0
                                        )
                                        print(f"🔊 [Send] 오디오 전송 (24kHz PCM): {len(response.data)} bytes (누적: {send_to_flutter.total_audio_bytes} bytes)")
                                    except asyncio.TimeoutError:
                                        print(f"⚠️ [Send] 오디오 전송 타임아웃")
                                    except Exception as e:
                                        print(f"⚠️ [Send] 오디오 전송 실패: {e}")
                                    continue  # 오디오 데이터 처리 후 다음 응답으로
                                
                                # 공식 예제 패턴: response.text를 바로 출력
                                if response.text is not None:
                                    text = response.text
                                    # 한국어가 포함된 텍스트만 Firebase에 저장 및 전송
                                    if logger._is_korean_text(text):
                                        try:
                                            await asyncio.wait_for(
                                                websocket.send_json({
                                                    "type": "text",
                                                    "data": text
                                                }),
                                                timeout=5.0
                                            )
                                            logger.append_text(text)
                                            print(f"📝 [Send] 텍스트 전송: {text[:50]}...")
                                        except asyncio.TimeoutError:
                                            print(f"⚠️ [Send] 텍스트 전송 타임아웃")
                                        except Exception as e:
                                            print(f"⚠️ [Send] 텍스트 전송 실패: {e}")
                                    else:
                                        print(f"⚠️ [Send] 영어만 포함된 텍스트는 전송하지 않습니다: {text[:50]}...")
                                    continue  # 텍스트 데이터 처리 후 다음 응답으로
                                
                                # 기존 server_content 처리 (호환성 유지)
                                if response.server_content:
                                    # 사용자 음성 인식 텍스트 저장 (input_transcription) - 우선 처리
                                    input_transcription = getattr(response.server_content, 'input_transcription', None)
                                    if input_transcription:
                                        input_text = getattr(input_transcription, 'text', None)
                                        is_final = getattr(input_transcription, 'is_final', True)
                                        if input_text and input_text.strip():
                                            if is_final:
                                                print(f"🎤 [사용자 음성 인식] {input_text}")
                                                append_user_buffer(input_text.strip())
                                            else:
                                                # 중간 인식 결과도 로그
                                                print(f"🎤 [인식 중...] {input_text}")
                                    
                                    # 추가: server_content의 다른 속성에서 사용자 음성 텍스트 찾기
                                    # 일부 경우 input_transcription이 없을 수 있으므로 다른 경로도 확인
                                    if not input_transcription or not getattr(input_transcription, 'text', None):
                                        # speech_recognition_event 확인
                                        speech_recognition = getattr(response.server_content, 'speech_recognition_event', None)
                                        if speech_recognition:
                                            recognized_text = getattr(speech_recognition, 'transcript', None) or getattr(speech_recognition, 'text', None)
                                            is_final_speech = getattr(speech_recognition, 'is_final', True)
                                            if recognized_text and recognized_text.strip() and is_final_speech:
                                                print(f"🎤 [사용자 음성 인식 - speech_recognition] {recognized_text}")
                                                append_user_buffer(recognized_text.strip())
                                    
                                    # AI 응답 텍스트 수집 (output_transcription)
                                    output_transcription = getattr(response.server_content, 'output_transcription', None)
                                    if output_transcription:
                                        transcript_text = getattr(output_transcription, 'text', None)
                                        if transcript_text and transcript_text.strip():
                                            logger.append_text(transcript_text)
                                    
                                    # 기존 model_turn 처리 (호환성 유지: response.data/text가 없을 경우 fallback)
                                    model_turn = response.server_content.model_turn
                                    if model_turn:
                                        for part in model_turn.parts:
                                            # 오디오 데이터 (response.data가 없을 경우 fallback)
                                            if part.inline_data:
                                                # response.data로 이미 처리되었으면 스킵
                                                # 하지만 response.data가 없을 경우를 대비해 fallback
                                                try:
                                                    audio_b64 = base64.b64encode(part.inline_data.data).decode('utf-8')
                                                    if not hasattr(send_to_flutter, 'total_audio_bytes'):
                                                        send_to_flutter.total_audio_bytes = 0
                                                    send_to_flutter.total_audio_bytes += len(part.inline_data.data)
                                                    await asyncio.wait_for(
                                                        websocket.send_json({
                                                            "type": "audio",
                                                            "data": audio_b64
                                                        }),
                                                        timeout=5.0
                                                    )
                                                    print(f"🔊 [Send] 오디오 전송 (fallback): {len(part.inline_data.data)} bytes (누적: {send_to_flutter.total_audio_bytes} bytes)")
                                                except asyncio.TimeoutError:
                                                    print(f"⚠️ [Send] 오디오 전송 타임아웃 (fallback)")
                                                except Exception as e:
                                                    print(f"⚠️ [Send] 오디오 전송 실패 (fallback): {e}")
                                            
                                            # 텍스트 데이터 (response.text가 없을 경우 fallback)
                                            if part.text and part.text.strip():
                                                # response.text로 이미 처리되었으면 스킵
                                                # 하지만 response.text가 없을 경우를 대비해 fallback
                                                if logger._is_korean_text(part.text):
                                                    try:
                                                        await asyncio.wait_for(
                                                            websocket.send_json({
                                                                "type": "text",
                                                                "data": part.text
                                                            }),
                                                            timeout=5.0
                                                        )
                                                        logger.append_text(part.text)
                                                        print(f"📝 [Send] 텍스트 전송 (fallback): {part.text[:50]}...")
                                                    except asyncio.TimeoutError:
                                                        print(f"⚠️ [Send] 텍스트 전송 타임아웃 (fallback)")
                                                    except Exception as e:
                                                        print(f"⚠️ [Send] 텍스트 전송 실패 (fallback): {e}")
                                    
                                    # 턴 종료 시그널 처리
                                    # X 버튼을 누르기 전까지는 오디오를 정상적으로 재생하기 위해 turn_complete 신호 전송
                                    # X 버튼을 누르면 turn_complete + exit: true를 보내서 오디오 재생 중단 및 홈으로 이동
                                    if response.server_content.turn_complete:
                                        logger.flush_model_turn()
                                        # 중요: turn_complete는 AI가 말을 끝냈다는 신호
                                        # X 버튼을 누르지 않았으면 오디오를 재생하고 계속 대화할 수 있어야 하므로
                                        # 일반 turn_complete 신호를 Flutter에 보냄 (오디오 재생을 위해)
                                        try:
                                            await asyncio.wait_for(
                                                websocket.send_json({
                                                    "type": "turn_complete"
                                                    # exit: true는 X 버튼을 누를 때만 추가됨
                                                }),
                                                timeout=5.0
                                            )
                                            print(f"✅ [Send] 턴 완료 신호 전송 (총 오디오: {send_to_flutter.total_audio_bytes} bytes) - 오디오 재생 시작")
                                            # 누적 바이트 초기화 (다음 턴을 위해)
                                            send_to_flutter.total_audio_bytes = 0
                                        except asyncio.TimeoutError:
                                            print(f"⚠️ [Send] 턴 완료 신호 전송 타임아웃")
                                        except Exception as e:
                                            print(f"⚠️ [Send] 턴 완료 신호 전송 실패: {e}")
                                        
                                        # 중요: 세션은 계속 활성 상태이며, receive_from_flutter()가 사용자 입력을 계속 받을 수 있음
                                        # session.receive() 루프는 계속 실행되어야 하므로 continue로 다음 응답을 기다림
                                        print(f"✅ [Send] 세션 활성 상태 유지 - 사용자 입력 대기 중 (session.receive() 계속 실행)")
                                        
                                        # WebSocket 연결 상태 확인
                                        try:
                                            # WebSocket이 열려있는지 확인
                                            if websocket.client_state.name == "CONNECTED":
                                                print(f"🔌 [Send] WebSocket 연결 상태: ✅ CONNECTED (계속 수신 가능)")
                                            else:
                                                print(f"⚠️ [Send] WebSocket 연결 상태: {websocket.client_state.name}")
                                        except Exception as e:
                                            print(f"⚠️ [Send] WebSocket 상태 확인 실패: {e}")
                                        
                                        continue  # turn_complete 처리 후 다음 응답을 기다리기 위해 continue
                            
                            # 중요: async for 루프는 turn_complete를 받은 후에도 계속 실행되어야 함
                            # 다음 사용자 입력에 대한 응답을 받기 위해 루프가 계속 실행됨
                            # 만약 루프가 종료되면 세션이 닫힌 것이므로 에러 처리로 이동
                        except Exception as e:
                            # session.receive()에서 발생하는 에러 처리
                            error_str = str(e)
                            
                            # 1011 에러 처리 (Gemini Live API 내부 오류) -> 클라이언트에 알리고 연결 종료
                            if "1011" in error_str or "internal error" in error_str.lower():
                                print(f"⚠️ [Send] Gemini Live API 1011 에러 발생: {e}")
                                print("⚠️ [Send] 내부 오류, WebSocket을 종료하여 클라이언트가 재연결하도록 합니다.")
                                try:
                                    await websocket.send_json({
                                        "type": "error",
                                        "code": 1011,
                                        "message": "Gemini 서비스가 일시적으로 불가합니다. 재연결 후 다시 시도해 주세요."
                                    })
                                except Exception:
                                    pass
                                try:
                                    await websocket.close(code=1011, reason="Gemini API internal error")
                                except Exception:
                                    pass
                                break  # 루프 종료
                            
                            # Deadline expired 에러 처리 (타임아웃)
                            if "deadline expired" in error_str.lower() or "deadline" in error_str.lower():
                                print(f"⚠️ [Send] Deadline expired 에러 발생: {e}")
                                print("⚠️ [Send] Gemini API 응답 시간 초과입니다. 연결을 종료합니다.")
                                try:
                                    await websocket.close(code=1011, reason="Deadline expired")
                                except Exception:
                                    pass
                                break  # 루프 종료
                            
                            # disconnect 관련 에러는 루프 종료
                            if "disconnect" in error_str.lower() or "Cannot call" in error_str:
                                print(f"🔌 [Send] 연결 종료 감지: {e}")
                                break
                            
                            print(f"⚠️ [Send] session.receive() 에러: {e}")
                            import traceback
                            traceback.print_exc()
                            # 일반 에러는 루프 종료 (재시도하지 않음)
                            break
                    except Exception as e:
                        error_str = str(e)
                        
                        # 1011 에러 처리 -> 클라이언트 알림 후 연결 종료
                        if "1011" in error_str or "internal error" in error_str.lower():
                            print(f"⚠️ [Send] 외부 루프에서 1011 에러 발생: {e}")
                            print("⚠️ [Send] 내부 오류, WebSocket을 종료하여 클라이언트가 재연결하도록 합니다.")
                            try:
                                await websocket.send_json({
                                    "type": "error",
                                    "code": 1011,
                                    "message": "Gemini 서비스가 일시적으로 불가합니다. 재연결 후 다시 시도해 주세요."
                                })
                            except Exception:
                                pass
                            try:
                                await websocket.close(code=1011, reason="Gemini API internal error")
                            except Exception:
                                pass
                            break  # 루프 종료
                        
                        # Deadline expired 에러 처리
                        if "deadline expired" in error_str.lower() or "deadline" in error_str.lower():
                            print(f"⚠️ [Send] 외부 루프에서 Deadline expired 에러 발생: {e}")
                            print("⚠️ [Send] Gemini API 응답 시간 초과입니다. 연결을 종료합니다.")
                            try:
                                await websocket.close(code=1011, reason="Deadline expired")
                            except Exception:
                                pass
                            break  # 루프 종료
                        
                        # disconnect 관련 에러는 루프 종료
                        if "disconnect" in error_str.lower() or "Cannot call" in error_str:
                            print(f"🔌 [Send] 연결 종료 감지: {e}")
                            break
                        
                        print(f"⚠️ [Send] 응답 처리 에러: {e}")
                        import traceback
                        traceback.print_exc()
                        # 에러가 발생하면 루프 종료 (재시도하지 않음)
                        break

            except WebSocketDisconnect:
                print("🔌 [Send] Client Disconnected")
            except Exception as e:
                error_str = str(e)
                if "1011" in error_str or "service is currently unavailable" in error_str.lower():
                    print("⚠️ [Send] Gemini 서비스 불가(1011). 잠시 후 재시도하세요.")
                    try:
                        await websocket.close(code=1011, reason="service unavailable")
                    except Exception:
                        pass
                    return
                if "disconnect" in error_str.lower() or "Cannot call" in error_str:
                    print(f"🔌 [Send] 연결 종료: {e}")
                else:
                    print(f"❌ [Send] WebSocket 에러: {e}")
                    import traceback
                    traceback.print_exc()

        # 태스크 실행 (타임아웃 없이 계속 실행)
        print("🚀 [Main] 세 코루틴 시작 - receive_from_flutter & image_sender_loop & send_to_flutter")
        try:
            await asyncio.gather(
                receive_from_flutter(),
                image_sender_loop(),
                send_to_flutter(),
                buffer_flush_loop()
            )
        except Exception as e:
            print(f"❌ [Main] 코루틴 실행 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("🛑 [Main] 세 코루틴 종료됨")


# ==========================================
# [메인] 실행 루프
# ==========================================
async def main():
    # 단독 실행 시 uvicorn 서버만 구동 (추가 Gemini 세션 생성 없음)
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))
    config = uvicorn.Config(app=app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
