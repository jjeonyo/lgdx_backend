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
import sqlite3
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

##################### 영상 생성 1초에 천원이니까 신중하게 돌릴 것 #######################
# 1. 환경 설정 (.env 파일 로드)

project_root = Path(__file__).resolve().parents[1]
load_dotenv(project_root / ".env")
API_KEY = os.getenv("GEMINI_API_KEY")

# Firebase 설정 (vision.py와 동일한 키 사용)
# serviceAccountKey.json은 프로젝트 루트에 위치함
FIREBASE_KEY_PATH = project_root / "serviceAccountKey.json"

if not FIREBASE_KEY_PATH.exists():
    print(f"⚠️ 경고: 키 파일을 찾을 수 없습니다: {FIREBASE_KEY_PATH}")
    # Fallback: 하드코딩된 경로 시도 (필요 시)
    FIREBASE_KEY_PATH = Path("/Users/harry/LG DX SCHOOL/lgdx_backend/serviceAccountKey.json")

if not API_KEY:
    print("❌ API 키가 없습니다. .env 파일을 확인하거나 코드를 수정하세요.")
    exit()

# 클라이언트 초기화
client = genai.Client(api_key=API_KEY)

def init_firebase():
    """Firebase 초기화 (이미 초기화되어 있으면 패스)"""
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(str(FIREBASE_KEY_PATH))
            firebase_admin.initialize_app(cred)
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
        # 1. 가장 최근 세션 가져오기 (start_time 기준 내림차순)
        sessions_ref = db_client.collection('sessions')
        # start_time이 없는 문서가 있을 수 있으므로 쿼리 시 유의 (일반적으로는 문제없음)
        query = sessions_ref.order_by('start_time', direction=firestore.Query.DESCENDING).limit(1)
        docs = list(query.stream())
        
        if not docs:
            print("❌ 저장된 대화 세션이 없습니다.")
            return None
            
        # 최근 세션 ID와 데이터 추출
        session_doc = docs[0]
        session_id = session_doc.id
        
        print(f"📖 최근 대화 세션(ID: {session_id})을 불러옵니다...")
        
        # 2. 해당 세션의 메시지 가져오기 (Subcollection)
        messages_ref = session_doc.reference.collection('messages')
        messages_docs = messages_ref.order_by('created_at').stream()
        
        messages_list = []
        for m in messages_docs:
            messages_list.append(m.to_dict())
            
        if not messages_list:
            print("❌ 이 세션에는 대화 내용이 없습니다.")
            return None
        
        # 3. 대화 내용 포맷팅
        conversation_text = ""
        for msg_data in messages_list:
            sender = msg_data.get('sender', 'unknown')
            content = msg_data.get('content', '')
            conversation_text += f"[{sender}]: {content}\n"
            
        return conversation_text.strip()

    except Exception as e:
        print(f"❌ Firebase 읽기 오류: {e}")
        return None


def create_visual_prompt(conversation_context):
    """
    대화 내용을 바탕으로 영상 생성용 프롬프트(영어)를 작성합니다.
    """
    print("🤔 대화 내용 분석 및 프롬프트 작성 중...")
    
    prompt_instruction = f"""
    Analyze the following conversation history between a user and an AI assistant about a washing machine problem.
    Identify the specific problem or the solution being discussed.
    
    [Conversation History]
    {conversation_context}
    
    Based on this, create a high-quality, cinematic, and detailed English visual prompt for a video generation model (like OpenAI Sora or Google Veo).
    The video should depict the solution or the maintenance step clearly.
    Focus on realistic textures, lighting, and clear action.
    Output ONLY the prompt in English.
    """
    
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt_instruction
    )
    
    visual_prompt = response.text.strip()
    print(f"📝 생성된 묘사(Prompt): {visual_prompt}")
    return visual_prompt


def generate_solution_video(visual_prompt, output_filename="solution.mp4"):
    print("🎥 비디오 생성 중... (시간이 소요될 수 있습니다)")
    try:
        operation = client.models.generate_videos(
            model="veo-3.1-fast-generate-preview",
            prompt=visual_prompt,
            config=types.GenerateVideosConfig(
                aspect_ratio="16:9",
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
        else:
            print("❌ 비디오가 생성되지 않았습니다.")

    except Exception as e:
        print(f"❌ 비디오 생성 오류: {e}")

    
 
# === 메인실행부 ===
if __name__ == "__main__":
    # 사용자 시나리오 테스트
    print("--- 🛠️ AI 해결책 생성기 ---")
    
    # 1. 대화 내용 가져오기
    conversation_context = get_latest_conversation_context()
    
    if not conversation_context:
        print("대화 내용을 불러오지 못해 기본 예제로 진행합니다.")
        conversation_context = "사용자는 세탁기 배수가 되지 않는 문제를 겪고 있음"

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
        generate_solution_video(prompt, str(video_filename))
