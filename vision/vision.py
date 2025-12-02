import asyncio
import os
import cv2
import pathlib
import sys
import time
import pyaudio
import warnings
import traceback
import threading
import queue
from dotenv import load_dotenv

# [Firebase 라이브러리 추가]
try:
    import firebase_admin
    from firebase_admin import credentials, db
except ImportError:
    print("❌ firebase-admin이 설치되지 않았습니다. 'pip install firebase-admin'을 실행하세요.")
    sys.exit(1)

# [Gemini 라이브러리]
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

# Realtime Database URL
FIREBASE_DATABASE_URL = "https://lgdx-6054d-default-rtdb.asia-southeast1.firebasedatabase.app/"

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
FIREBASE_KEY_PATH = '/Users/harry/LG DX SCHOOL/lgdx_backend/vision/FirebaseAdmin.json'

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

# ==========================================
# [클래스] Firebase Logger (Realtime Database 사용)
# ==========================================
class FirebaseLogger:
    def __init__(self):
        self.session_ref = None
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
                # Realtime Database는 databaseURL이 필수입니다.
                firebase_admin.initialize_app(cred, {
                    'databaseURL': FIREBASE_DATABASE_URL
                })
                print(f"🔥 Firebase 연결 성공! ({FIREBASE_DATABASE_URL})")
            except Exception as e:
                print(f"❌ Firebase 초기화 오류: {e}")
                sys.exit(1)

    def _start_session(self):
        try:
            # 'sessions' 노드 아래에 새 세션 생성 (push)
            self.session_ref = db.reference('sessions').push()
            self.session_ref.set({
                'start_time': int(time.time() * 1000),  # timestamp (ms)
                'model_id': MODEL_ID,
                'status': 'active'
            })
            print(f"📄 새 세션 ID: {self.session_ref.key}")
        except Exception as e:
            print(f"❌ 세션 생성 실패: {e}")

    def log_message(self, sender, text):
        if not self.session_ref: return
        try:
            # 해당 세션의 'messages' 리스트에 대화 추가
            self.session_ref.child('messages').push().set({
                'sender': sender,      # 'user' or 'gemini'
                'content': text,
                'created_at': int(time.time() * 1000)
            })
        except Exception as e:
            print(f"⚠️ 로그 저장 실패: {e}")

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
    persona_path = current_dir / "persona/persona_세탁기수리법.txt"
    
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

            async def receive():
                model_response_text_buffer = ""
                while shared_state["running"]:
                    try:
                        async for response in session.receive():
                            # Part 1: Gemini를 통해 사용자 음성 인식 처리
                            if event := response.speech_recognition_event:
                                if event.text and not event.is_final:
                                    print(f"\r[... User]: {event.text}", end="", flush=True)
                                if event.text and event.is_final:
                                    # 최종 인식된 텍스트로 RAG 검색 및 로깅 수행
                                    print(f"\n[🗣️ User]: {event.text}")
                                    logger.log_message('user', event.text)
                                    rag_queue.put_nowait(event.text)

                            # Part 2: 모델 응답 처리 (오디오 + 텍스트)
                            if model_turn := (response.server_content and response.server_content.model_turn):
                                for part in model_turn.parts:
                                    if part.text:
                                        model_response_text_buffer += part.text
                                    if part.inline_data:
                                        audio_player.add_audio(part.inline_data.data)

                                # 모델의 응답이 끝나면, 전체 텍스트를 한 번에 로깅
                                if response.server_content.turn_complete and model_response_text_buffer.strip():
                                    on_model_speak(model_response_text_buffer)
                                    model_response_text_buffer = ""
                    except Exception as e:
                        print(f"수신 종료: {e}")
                        # 오류 발생 시, 버퍼에 남아있는 텍스트가 있으면 로깅
                        if model_response_text_buffer.strip():
                           on_model_speak(model_response_text_buffer)
                        break

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

            tasks = [
                asyncio.create_task(display_loop()),
                asyncio.create_task(send_video()),
                asyncio.create_task(send_audio()),
                asyncio.create_task(receive()),
                asyncio.create_task(rag_loop())
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