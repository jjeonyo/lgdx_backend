import os
import io
import pathlib
from pathlib import Path
import datetime
from dotenv import load_dotenv
import google.genai as genai
from google.genai import types
from openai import OpenAI
from PIL import Image
import time
import socket
import sqlite3
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import base64
import urllib.request

##################### 영상 생성 1초에 천원이니까 신중하게 돌릴 것 #######################
# 1. 환경 설정 (.env 파일 로드)

project_root = Path(__file__).resolve().parents[1]
load_dotenv(project_root / ".env")
API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Firebase 설정 (vision.py와 동일한 키 사용)
# serviceAccountKey.json은 프로젝트 루트에 위치함
FIREBASE_KEY_PATH = project_root / "serviceAccountKey.json"
FIREBASE_STORAGE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET") # .env에서 버킷 이름 로드

if not FIREBASE_STORAGE_BUCKET:
    # Fallback: 프로젝트 ID 기반 기본 버킷 추정
    try:
        import json
        if FIREBASE_KEY_PATH.exists():
            with open(FIREBASE_KEY_PATH) as f:
                key_data = json.load(f)
                project_id = key_data.get("project_id")
                if project_id:
                    FIREBASE_STORAGE_BUCKET = f"{project_id}.appspot.com"
                    print(f"ℹ️ FIREBASE_STORAGE_BUCKET 환경변수가 없어 {FIREBASE_STORAGE_BUCKET}를 기본값으로 사용합니다.")
    except Exception as e:
        print(f"⚠️ 버킷 이름 추정 실패: {e}")

if not FIREBASE_KEY_PATH.exists():
    print(f"⚠️ 경고: 키 파일을 찾을 수 없습니다: {FIREBASE_KEY_PATH}")
    # Fallback: 하드코딩된 경로 시도 (필요 시)
    FIREBASE_KEY_PATH = Path("/Users/harry/LG DX SCHOOL/lgdx_backend/serviceAccountKey.json")

if not API_KEY:
    print("❌ API 키가 없습니다. .env 파일을 확인하거나 코드를 수정하세요.")
    exit()
if not OPENAI_API_KEY:
    print("❌ OPENAI_API_KEY가 없습니다. .env 파일에 OpenAI 키를 추가하세요.")
    exit()

# 클라이언트 초기화
client = OpenAI(api_key=OPENAI_API_KEY)

# 비디오 작업 식별용 환경변수 (외부 호출 시 주입)
VIDEO_JOB_ID = os.getenv("VIDEO_JOB_ID")
VIDEO_SESSION_ID = os.getenv("VIDEO_SESSION_ID")

def get_host_ip():
    """현재 서버의 로컬 IP 주소를 반환합니다."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"

def init_firebase():
    """Firebase 초기화 (이미 초기화되어 있으면 패스)"""
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(str(FIREBASE_KEY_PATH))
            options = {}
            if FIREBASE_STORAGE_BUCKET:
                options['storageBucket'] = FIREBASE_STORAGE_BUCKET
            
            firebase_admin.initialize_app(cred, options)
            print("🔥 Firebase 연결 성공!")
    except Exception as e:
        print(f"❌ Firebase 초기화 오류: {e}")

def get_latest_conversation_context(force_session_id: str | None = None):
    """
    Firebase Firestore에서 가장 최근 세션의 대화 내용을 가져옵니다.
    """
    init_firebase()
    
    try:
        db_client = firestore.client()
        if force_session_id:
            print(f"🔎 지정된 세션({force_session_id})의 대화 내역을 가져옵니다...")
            session_doc_ref = db_client.collection('chat_rooms').document(force_session_id)
            if not session_doc_ref.get().exists:
                print("❌ 지정된 세션 문서를 찾을 수 없습니다.")
                return None, None
        else:
            # 1. collection_group을 사용하여 모든 'messages' 컬렉션에서 가장 최근 메시지를 찾습니다.
            # 이 방식은 상위 문서(Ghost Document) 존재 여부와 상관없이 메시지 자체만으로 찾습니다.
            print("🔎 전체 채팅 내역에서 가장 최근 메시지를 검색합니다...")
            
            # 'messages' 컬렉션 그룹에서 timestamp 내림차순으로 1개만 가져옴
            latest_msg_query = db_client.collection_group('messages')\
                .order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1)
                
            latest_msgs = list(latest_msg_query.stream())
            if not latest_msgs:
                print("❌ 저장된 메시지가 없습니다.")
                return None, None
            
            # 가장 최근 메시지 찾음
            last_msg = latest_msgs[0]
            # 이 메시지의 부모 컬렉션(messages) -> 그 부모 문서(room_user_XXX)
            session_doc_ref = last_msg.reference.parent.parent
        
        if not session_doc_ref:
            print("❌ 세션 문서를 찾을 수 없습니다.")
            return None, None
            
        session_id = session_doc_ref.id
        print(f"📖 최근 대화 세션(ID: {session_id})을 불러옵니다...")
        
        # 2. 해당 세션의 메시지 가져오기
        messages_ref = session_doc_ref.collection('messages')
        messages_docs = messages_ref.order_by('timestamp').stream()
        
        messages_list = []
        for m in messages_docs:
            messages_list.append(m.to_dict())
            
        if not messages_list:
            print("❌ 이 세션에는 대화 내용이 없습니다.")
            return session_id, None
        
        # 3. 대화 내용 포맷팅
        conversation_text = ""
        for msg_data in messages_list:
            sender = msg_data.get('sender', 'unknown')
            content = msg_data.get('text', '')
            conversation_text += f"[{sender}]: {content}\n"
            
        return session_id, conversation_text.strip()

    except Exception as e:
        print(f"❌ Firebase 읽기 오류: {e}")
        return None, None


def create_visual_prompt(conversation_context):
    """
    대화 내용을 바탕으로 영상 생성용 프롬프트(영어)를 작성합니다.
    """
    """
    사용자의 현재 문제 : 
    """
    print(conversation_context)
    print("🤔 대화 내용 분석 및 프롬프트 작성 중...")
    
    prompt_instruction = f"""
    Analyze the following conversation history between a user and an AI assistant about a washing machine problem.
    Identify the specific problem or the solution being discussed.
    
    [Conversation History]
    {conversation_context}
    
    Based on this, create a high-quality, cinematic, and detailed visual prompt for a video generation model.
    The video should depict the solution or the maintenance step clearly.
    Focus on realistic textures, lighting, and clear action. 
    the video will be 6 seconds long.
 
    """



    
    response = client.responses.create(
        model="gpt-5-nano",
        input=prompt_instruction,
        reasoning={"effort": "low"},
    )
    
    # Responses API: output_text가 기본, 없으면 첫 content의 text를 시도
    visual_prompt = getattr(response, "output_text", None)
    if not visual_prompt:
        try:
            visual_prompt = response.output[0].content[0].text
        except Exception:
            visual_prompt = getattr(response, "text", "")
    visual_prompt = (visual_prompt or "").strip()
    print(f"📝 생성된 묘사(Prompt): {visual_prompt}")
    return visual_prompt


def generate_solution_video(visual_prompt, output_filename="solution.mp4"):
    """
    비디오 생성 후 상태를 폴링하고 완료되면 다운로드해 저장.
    """
    print("🎥 비디오 생성 중... (폴링 후 다운로드)")
    try:
        job = client.videos.create(
            model="sora-2",
            prompt=visual_prompt,
        )

        job_id = getattr(job, "id", None)
        if not job_id:
            raise RuntimeError("영상 생성 응답에 job_id가 없습니다.")

        # 상태 폴링
        while True:
            status = client.videos.retrieve(job_id)
            if status.status in ("completed", "failed", "cancelled"):
                break
            time.sleep(2)

        if status.status != "completed":
            err = getattr(status, "error", None)
            raise RuntimeError(f"영상 생성 실패/취소: {status.status} | {err}")

        # OpenAI 공식 가이드 형태: output[0].content[0].file_id를 사용해 파일 다운로드
        outputs = getattr(status, "output", []) or []
        if not outputs:
            raise RuntimeError("완료된 작업에 output이 없습니다.")

        first_output = outputs[0]
        contents = getattr(first_output, "content", []) or []
        if not contents:
            raise RuntimeError("완료된 작업에 content가 없습니다.")

        first_content = contents[0]
        file_id = getattr(first_content, "file_id", None)
        if not file_id:
            raise RuntimeError("content에 file_id가 없습니다.")

        # 파일 내용 가져오기 (스트리밍 대응)
        content = client.files.content(file_id)

        saved = False
        if hasattr(content, "write_to_file"):
            content.write_to_file(output_filename)
            saved = True
        elif hasattr(content, "read"):
            with open(output_filename, "wb") as f:
                f.write(content.read())
            saved = True
        elif isinstance(content, (bytes, bytearray)):
            with open(output_filename, "wb") as f:
                f.write(content)
            saved = True
        elif getattr(content, "data", None):
            data = content.data
            if isinstance(data, (bytes, bytearray)):
                with open(output_filename, "wb") as f:
                    f.write(data)
                saved = True

        if not saved:
            raise RuntimeError("다운로드 결과를 파일로 저장하지 못했습니다.")

        print(f"✅ Generated video saved to {output_filename}")
        return output_filename

    except Exception as e:
        print(f"❌ 비디오 생성 오류: {e}")
        return None



    reasoning={ "effort": "low" },
    
 
# === 메인실행부 ===
if __name__ == "__main__":
    # 사용자 시나리오 테스트
    print("--- 🛠️ AI 해결책 생성기 ---")
    
    # 1. 대화 내용 가져오기 (지정된 세션이 있으면 우선 사용)
    result = get_latest_conversation_context(VIDEO_SESSION_ID)
    
    if result:
        session_id, conversation_context = result
    else:
        session_id = None
        conversation_context = None
    
    if not conversation_context:
        print("대화 내용을 불러오지 못해 기본 예제로 진행합니다.")
        conversation_context = "헹굼할 때만 계속 OE 오류가 떠. 지금 안에 이미 빨래가 있어서 문도 안열리고, 배수 필터를 열었더니 물이 나와"

    # 2. 묘사 생성
    prompt = create_visual_prompt(conversation_context)
    
    # 3. 이미지/영상 생성
    if prompt:
        # 생성된사진 폴더 경로 설정
        current_dir = pathlib.Path(__file__).parent.absolute()
        output_dir = current_dir / "assets_generate"
        output_dir.mkdir(exist_ok=True)  # 폴더가 없으면 생성
        
        # 파일명 생성 (타임스탬프 포함하여 중복 방지)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        job_suffix = VIDEO_JOB_ID if VIDEO_JOB_ID else timestamp

        # 영상 생성
        video_filename = output_dir / f"result_solution_{job_suffix}.mp4"
        saved_path = generate_solution_video(prompt, str(video_filename))
        
        if saved_path:
            # 로컬 서버에서 접근 가능한 URL만 안내 (Firestore 저장/업로드는 제거)
            server_ip = get_host_ip()
            filename = pathlib.Path(saved_path).name
            video_url = f"http://{server_ip}:8000/assets/{filename}"
            print(f"🔗 로컬 비디오 URL: {video_url}")