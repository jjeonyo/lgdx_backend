import os
import io
import pathlib
from pathlib import Path
import datetime
from dotenv import load_dotenv
import google.genai as genai
from google.genai import types
from PIL import Image
import time
import socket
import sqlite3
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage

##################### 영상 생성 1초에 천원이니까 신중하게 돌릴 것 #######################
# 1. 환경 설정 (.env 파일 로드)

project_root = Path(__file__).resolve().parents[1]
load_dotenv(project_root / ".env")
API_KEY = os.getenv("google_api")

# Firebase 설정 (vision.py와 동일한 키 사용)
# serviceAccountKey.json은 프로젝트 루트에 위치함
FIREBASE_KEY_PATH = Path(r"C:\dxfirebasekey\serviceAccountKey.json")
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

# 클라이언트 초기화
client = genai.Client(api_key=API_KEY)

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

def get_latest_conversation_context():
    """
    Firebase Firestore에서 가장 최근 세션의 대화 내용을 가져옵니다.
    """
    init_firebase()
    
    try:
        db_client = firestore.client()
        # 1. collection_group을 사용하여 모든 'messages' 컬렉션에서 가장 최근 메시지를 찾습니다.
        # 이 방식은 상위 문서(Ghost Document) 존재 여부와 상관없이 메시지 자체만으로 찾습니다.
        print("🔎 전체 채팅 내역에서 가장 최근 메시지를 검색합니다...")
        
        # 'messages' 컬렉션 그룹에서 timestamp 내림차순으로 1개만 가져옴
        # 주의: 이를 위해서는 Firestore 콘솔에서 'messages' 컬렉션 그룹에 대한 복합 색인이 필요할 수 있습니다.
        # 만약 색인 에러가 나면 콘솔에 출력된 URL을 클릭해서 생성해야 합니다.
        latest_msg_query = db_client.collection_group('messages')\
            .order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1)
            
        latest_msgs = list(latest_msg_query.stream())
        
        # if not latest_msgs:
        #      # 메시지가 하나도 없으면 기존 방식대로 특정 ID 확인
        #     print("⚠️ 메시지를 찾지 못했습니다. 기본 ID('room_user_001')를 확인합니다.")
        #     doc_ref = db_client.collection('chat_rooms').document('room_user_001')
        #     doc = doc_ref.get()
        #     if doc.exists:
        #         latest_session = doc
        #         session_id = doc.id
        #         # 빈 방이라도 session_id는 반환
        #         print(f"📖 대화 내용이 없는 기본 세션(ID: {session_id})을 사용합니다.")
        #         return session_id, "" 
        #     else:
        #         print("❌ 저장된 대화 세션이 없습니다.")
        #         return None, None

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
    Focus on realistic textures, lighting, and clear action. the video will be 6 seconds long.
 
    """



    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt_instruction
    )
    
    visual_prompt = response.text.strip()
    print(f"📝 생성된 묘사(Prompt): {visual_prompt}")
    return visual_prompt


def upload_video_to_firebase(file_path):
    """생성된 비디오를 Firebase Storage에 업로드하고 URL을 반환합니다."""
    print(f"📤 Firebase Storage 업로드 시작: {file_path}")
    try:
        if not firebase_admin._apps:
            init_firebase()
            
        bucket = storage.bucket(name=FIREBASE_STORAGE_BUCKET) # 버킷 이름 명시
        
        # 1. chat_rooms 폴더 내의 기존 파일들을 스캔하여 다음 번호 결정
        blobs = list(bucket.list_blobs(prefix="chat_rooms/"))
        max_num = 0
        
        for b in blobs:
            name = b.name
            # chat_rooms/video_001.mp4 형태 파싱
            if name.startswith("chat_rooms/video_") and name.endswith(".mp4"):
                try:
                    # "chat_rooms/video_" (17글자) 이후부터 ".mp4" (-4) 이전까지 추출
                    num_part = name[17:-4]
                    num = int(num_part)
                    if num > max_num:
                        max_num = num
                except ValueError:
                    continue
                    
        next_num = max_num + 1
        new_filename = f"video_{next_num:05d}.mp4"
        storage_path = f"chat_rooms/{new_filename}"
        
        print(f"🔢 다음 파일명 결정: {storage_path}")
        
        blob = bucket.blob(storage_path)
        
        # 메타데이터 설정
        blob.metadata = {"contentType": "video/mp4"}
        
        blob.upload_from_filename(file_path)
        
        # 공개 URL 생성 (Make public)
        blob.make_public()
        print(f"✅ 업로드 완료! URL: {blob.public_url}")
        return blob.public_url
        
    except Exception as e:
        print(f"❌ 업로드 실패: {e}")
        return None

def save_video_message_to_firestore(session_id, video_url):
    """Firestore에 비디오 메시지를 저장합니다."""
    try:
        db = firestore.client()
        # 해당 세션의 messages 컬렉션에 추가
        messages_ref = db.collection('chat_rooms').document(session_id).collection('messages')
        
        messages_ref.add({
            "sender": "ai",
            "text": "솔루션 영상을 생성했습니다.",
            "video_url": video_url,
            "message_type": "VIDEO",
            "created_at": firestore.SERVER_TIMESTAMP
        })
        print(f"💾 Firestore에 비디오 메시지 저장 완료 (Session: {session_id})")
        
    except Exception as e:
        print(f"❌ Firestore 저장 실패: {e}")


def generate_solution_video(visual_prompt, output_filename="solution.mp4"):
    print("🎥 비디오 생성 중... (시간이 소요될 수 있습니다)")
    try:
        operation = client.models.generate_videos(
            model="veo-3.1-fast-generate-preview",
            prompt=visual_prompt,
            config=types.GenerateVideosConfig(
                aspect_ratio="9:16",
                duration_seconds=8,
            )
        )

        while not operation.done:
            print("Waiting for video generation to complete...")
            time.sleep(3)
            operation = client.operations.get(operation)

        # Download the generated video.
        if operation.response.generated_videos:
            generated_video = operation.response.generated_videos[0]
            client.files.download(file=generated_video.video)
            generated_video.video.save(output_filename)
            print(f"✅ Generated video saved to {output_filename}")
            return output_filename
        else:
            print("❌ 비디오가 생성되지 않았습니다.")
            return None

    except Exception as e:
        print(f"❌ 비디오 생성 오류: {e}")
        return None

    
 
# === 메인실행부 ===
if __name__ == "__main__":
    # 사용자 시나리오 테스트
    print("--- 🛠️ AI 해결책 생성기 ---")
    
    # 1. 대화 내용 가져오기
    result = get_latest_conversation_context()
    
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

        # 영상 생성
        video_filename = output_dir / f"result_solution_{timestamp}.mp4"
        saved_path = generate_solution_video(prompt, str(video_filename))
        
        # 4. Firebase 업로드 대신 로컬 URL 사용
        if saved_path and session_id:
            # video_url = upload_video_to_firebase(saved_path) # Firebase 업로드 생략
            
            # 로컬 URL 생성 (서버 IP 기반)
            server_ip = get_host_ip()
            filename = pathlib.Path(saved_path).name
            video_url = f"http://{server_ip}:8000/assets/{filename}"
            
            print(f"🔗 로컬 비디오 URL 생성: {video_url}")
            
            if video_url:
                save_video_message_to_firestore(session_id, video_url)
        elif saved_path:
            print("⚠️ 세션 ID가 없어 Firestore에 저장하지 못했습니다. (로컬 파일만 생성됨)")