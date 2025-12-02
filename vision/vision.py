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
        if not firebase_admin._apps:
            try:
                # 키 파일 경로 확인 필수
                if not os.path.exists(FIREBASE_KEY_PATH):
                    print(f"❌ 키 파일을 찾을 수 없습니다: {FIREBASE_KEY_PATH}")
                    return # 또는 sys.exit(1)
                    
                cred = credentials.Certificate(FIREBASE_KEY_PATH)
                firebase_admin.initialize_app(cred, {
                    'databaseURL': FIREBASE_DATABASE_URL
                })
                print(f"🔥 Firebase 연결 성공! ({FIREBASE_DATABASE_URL})")
            except Exception as e:
                print(f"❌ Firebase 초기화 오류: {e}")

    def _start_session(self):
        try:
            # sessions 노드 아래에 현재 시간으로 새로운 대화 세션 생성
            self.session_ref = db.reference('sessions').push()
            self.session_ref.set({
                'start_time': int(time.time() * 1000),
                'model': MODEL_ID,
                'status': 'active'
            })
            print(f"📄 Firebase 세션 시작: {self.session_ref.key}")
        except Exception as e:
            print(f"❌ 세션 생성 실패: {e}")

    def log_message(self, sender, text):
        """
        sender: 'user' 또는 'gemini'
        text: 대화 내용
        """
        if not self.session_ref or not text: return
        try:
            # 해당 세션의 messages 아래에 대화 추가
            self.session_ref.child('messages').push().set({
                'sender': sender,
                'content': text,
                'timestamp': int(time.time() * 1000) # 정렬을 위한 타임스탬프
            })
            # print(f"   [DB 저장 완료] {sender}: {text[:10]}...") 
        except Exception as e:
            print(f"⚠️ 로그 저장 실패: {e}")

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

async def perform_summarization(client, session_id):
    """Firebase에서 대화를 가져와 요약하고 결과를 DB에 저장"""
    print(f"\n🔔 [Command Received] 요약 요청을 받았습니다. (Session: {session_id})")
    
    try:
        # 1. 대화 로그 가져오기
        ref = db.reference(f'sessions/{session_id}/messages')
        messages_data = ref.get() # 동기 호출 (데이터가 많지 않으므로 괜찮음)

        if not messages_data:
            print("   ⚠️ 대화 내용이 없습니다.")
            return

        # 2. 텍스트 변환
        chat_context = ""
        for key, msg in messages_data.items():
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
            model="gemini-1.5-flash",
            contents=prompt
        )
        summary_text = resp.text.strip()
        print(f"   📝 요약 완료: {summary_text}")

        # 4. 결과 DB 저장 및 명령어 초기화
        # summary 필드에 결과 저장
        db.reference(f'sessions/{session_id}').update({
            'summary': summary_text,
            'command': None  # 명령 수행 완료 후 초기화 (중요)
        })

    except Exception as e:
        print(f"   ❌ 요약 중 에러 발생: {e}")

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

# ==========================================
# [수정] Config 설정을 통한 '생각 과정' 숨기기
# ==========================================
def get_config():
    current_dir = pathlib.Path(__file__).parent.absolute()
    persona_path = current_dir / "persona/persona_세탁기수리법.txt"
    
    # [핵심 수정] 시스템 지침 강화
    base_instruction = """
    Role: 당신은 LG전자의 친절하고 전문적인 AI 홈 가전 어시스턴트입니다.
    
    [Critical Output Rules]
    1. **No Internal Monologue**: 답변 생성 전이나 중간에 'Addressing...', 'Thinking...', 'Strategy:'와 같은 내부 추론 과정을 텍스트로 절대 출력하지 마십시오.
    2. **Direct Response**: 사용자의 질문에 대한 '최종 답변'만 즉시 한국어로 말하십시오.
    3. **Tone**: 친구에게 말하듯 부드럽고 정중한 구어체(해요체)를 사용하십시오.
    4. **Language**: 무조건 한국어(Korean)로만 대답하십시오. 영어를 섞어 쓰지 마십시오.
    """

    system_instruction = base_instruction
    
    # 페르소나 파일이 있다면 내용을 읽어서 뒤에 붙임
    if persona_path.exists():
        try:
            file_content = persona_path.read_text(encoding="utf-8")
            system_instruction += f"\n\n[Domain Knowledge]\n{file_content}"
        except Exception:
            pass

    return {
        "response_modalities": ["AUDIO"], 
        "speech_config": {
            "voice_config": {
                "prebuilt_voice_config": {
                    "voice_name": "Kore"
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
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)

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
                        frame = cv2.resize(shared_state["latest_frame"], (480, 640))
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

            # [Task 4] 모델 응답 수신 및 처리 (내부 함수)
            async def receive():
                model_response_buffer = ""
                last_user_text = ""

                try:
                    async for server_content in session.receive():
                        # ① 사용자 음성 인식 결과 처리
                        if transcription := server_content.input_transcription:
                            if transcription.final:
                                last_user_text = transcription.text.strip()
                                if last_user_text:
                                    logger.log_message("user", last_user_text)
                                    # RAG 검색 큐에 추가 (원래 로직 복원)
                                    rag_queue.put_nowait(last_user_text)
                            else:
                                # 중간 인식 결과 출력 (선택 사항)
                                print(f"\r[... User]: {transcription.text}", end="", flush=True)
                            continue

                        # ② 모델 응답 처리
                        if model_turn := server_content.model_turn:
                            for part in model_turn.parts:
                                # 🔥 (A) 오디오 스트림이 나온 경우 → 직전에 누적된 텍스트만 "최종 발화"로 저장
                                if hasattr(part, "inline_data") and part.inline_data:
                                    audio_player.add_audio(part.inline_data.data)
                                    
                                    clean_text = model_response_buffer.strip()
                                    # 불필요한 reasoning 제거 및 로깅
                                    if clean_text and clean_text != last_user_text:
                                        # 간단한 필터링 후 저장
                                        if not ("thinking" in clean_text.lower() or "what should i say" in clean_text.lower()):
                                            logger.log_message("gemini", clean_text)
                                            print(f"\n[🤖 Gemini]: {clean_text}")
                                    
                                    # 다음 발화를 위해 버퍼 초기화
                                    model_response_buffer = ""
                                    continue

                                # 🔥 (B) 순수 텍스트 (여기에는 reasoning 포함됨) → DB 저장 금지, 버퍼에만 임시 저장
                                if hasattr(part, "text") and part.text:
                                    text = part.text.strip()

                                    # 현실적인 방어 로직 — 사내 추론/시뮬레이션 대사 제거
                                    if (
                                        text == last_user_text                                 # 사용자 발화와 동일
                                        or text.startswith("User:")                             # 시뮬레이션 사용자 대사
                                        or text.startswith("Assistant:")                        # 시뮬레이션 모델 대사
                                        or "what should i say" in text.lower()                  # reasoning 힌트
                                        or "thinking" in text.lower()                           # chain-of-thought
                                        or text.endswith("?") and "should" in text.lower()      # self-questioning
                                    ):
                                        continue

                                    model_response_buffer += text
                            continue

                        # ③ 턴 종료 (turn_complete) — 안전하게 마무리
                        if server_content.turn_complete:
                            clean_text = model_response_buffer.strip()
                            if clean_text and clean_text != last_user_text:
                                if not ("thinking" in clean_text.lower() or "what should i say" in clean_text.lower()):
                                    logger.log_message("gemini", clean_text)
                                    print(f"\n[🤖 Gemini (Final)]: {clean_text}")
                            model_response_buffer = ""

                except Exception as e:
                    print(f"수신 중단: {e}")
                    # 에러로 끊겼을 때 버퍼에 남은 내용이 있다면 저장하고 종료
                    if model_response_buffer.strip():
                        logger.log_message('gemini', model_response_buffer)

            # [Task 5] RAG 검색 및 컨텍스트 주입
            async def rag_loop():
                while shared_state["running"]:
                    try:
                        # 큐에서 텍스트 꺼내기 (없으면 대기하지 않고 넘어감 -> timeout)
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
                            
                            # 모델에게 텍스트로 정보 전달
                            await session.send(input=msg, end_of_turn=False)
                        else:
                            print("   ⚠️ 검색 결과 없음")
                            
                    except Exception as e:
                        print(f"RAG Loop Error: {e}")
                    
                    await asyncio.sleep(0.1)

            # [Task 6] Flutter 명령 감지 루프 (추가된 부분)
            async def command_watcher():
                current_session_id = logger.session_ref.key
                last_command = None
                command_ref = db.reference(f'sessions/{current_session_id}/command')
                
                while shared_state["running"]:
                    try:
                        # polling 방식으로 1초마다 확인 (Listen보다 async 충돌 위험이 적음)
                        command = command_ref.get()
                        
                        if command == "summarize":
                            # 요약 로직 실행 (비동기)
                            await perform_summarization(client, current_session_id)
                        
                        await asyncio.sleep(1.0) # 1초 대기
                    except Exception as e:
                        print(f"Command Watcher Error: {e}")
                        await asyncio.sleep(1.0)

            tasks = [
                asyncio.create_task(display_loop()),
                asyncio.create_task(send_video()),
                asyncio.create_task(send_audio()),
                asyncio.create_task(receive()),
                asyncio.create_task(rag_loop()),
                asyncio.create_task(command_watcher()) # <--- 여기 추가
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
