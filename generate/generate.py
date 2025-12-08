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
from firebase_admin import db

##################### 영상 생성 1초에 천원이니까 신중하게 돌릴 것 #######################
# 1. 환경 설정 (.env 파일 로드)

project_root = Path(__file__).resolve().parents[1]
load_dotenv(project_root / ".env")
API_KEY = os.getenv("google_api")

# Firebase 설정 (vision.py와 동일한 키 사용)
FIREBASE_KEY_PATH = "/Users/harry/LG DX SCHOOL/lgdx_backend/vision/FirebaseAdmin.json"
# Realtime Database URL도 vision.py와 동일해야 함 (환경변수나 상수로 관리 추천)
# 여기서는 예시 URL 사용 (vision.py에서 수정한 URL로 변경 필요)
FIREBASE_DB_URL = "https://lgdx-6054d-default-rtdb.asia-southeast1.firebasedatabase.app/"

if not API_KEY:
    print("❌ google_api가 없습니다. .env 파일을 확인하거나 코드를 수정하세요.")
    exit()

# 클라이언트 초기화
client = genai.Client(api_key=API_KEY)

def init_firebase():
    """Firebase 초기화 (이미 초기화되어 있으면 패스)"""
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(str(FIREBASE_KEY_PATH))
            firebase_admin.initialize_app(cred, {
                'databaseURL': FIREBASE_DB_URL
            })
            print("🔥 Firebase 연결 성공!")
    except Exception as e:
        print(f"❌ Firebase 초기화 오류: {e}")

def get_latest_conversation_context():
    """
    Firebase Realtime Database에서 가장 최근 세션의 대화 내용을 가져옵니다.
    """
    init_firebase()
    
    try:
        # 1. 모든 세션 가져오기 (세션 ID가 타임스탬프이므로 정렬 가능)
        sessions_ref = db.reference('sessions')
        sessions = sessions_ref.order_by_key().limit_to_last(1).get()
        
        if not sessions:
            print("❌ 저장된 대화 세션이 없습니다.")
            return None
            
        # 최근 세션 ID와 데이터 추출
        session_id = list(sessions.keys())[0]
        session_data = sessions[session_id]
        
        print(f"📖 최근 대화 세션(ID: {session_id})을 불러옵니다...")
        
        # 2. 해당 세션의 메시지 가져오기
        if 'messages' not in session_data:
            print("❌ 이 세션에는 대화 내용이 없습니다.")
            return None
            
        messages_dict = session_data['messages']
        
        # 메시지 정렬 (push ID 기준, 시간순)
        sorted_messages = sorted(messages_dict.items(), key=lambda x: x[0])
        
        # 3. 대화 내용 포맷팅
        conversation_text = ""
        for msg_id, msg_data in sorted_messages:
            sender = msg_data.get('sender', 'unknown')
            content = msg_data.get('content', '')
            conversation_text += f"[{sender}]: {content}\n"
            
        return conversation_text.strip()

    except Exception as e:
        print(f"❌ Firebase 읽기 오류: {e}")
        return None

# 2. [1단계: 작가 AI] 문제 상황을 시각적 묘사로 변환
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

# 3. [2단계: 화가 AI] 이미지 생성 (Imagen 3)
def generate_solution_image(visual_prompt, output_filename="solution.png"):
    """
    프롬프트를 받아 실제 이미지를 생성하고 저장합니다.
    """
    print("🎨 이미지 그리는 중... (약 5~10초 소요)")
    
    try:
        # Imagen 모델 호출
        response = client.models.generate_images(
            model='imagen-4.0-generate-001',
            prompt=visual_prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="9:16",
                person_generation="allow_adult" # 손이나 사람이 나와야 하므로 허용
            )
        )

        # 이미지 저장
        if response.generated_images:
            image_data = response.generated_images[0].image
            image = Image.open(io.BytesIO(image_data.image_bytes))
            image.save(output_filename)
            print(f"✅ 해결책 이미지가 저장되었습니다: {output_filename}")
            
            # (선택) 바로 이미지 띄우기
            # image.show()
            return output_filename
        else:
            print("❌ 이미지가 생성되지 않았습니다.")
            return None

    except Exception as e:
        print(f"❌ 이미지 생성 오류: {str(e)}")
        if "403" in str(e):
            print("Tip: 사용 중인 프로젝트가 Imagen API 사용 권한이 있는지 확인하세요.")
        return None

# 4. [보너스: 비디오 생성] (Veo 모델 접근 권한 필요)
# 현재 대부분의 계정에서 Imagen(이미지)은 되지만 Veo(영상)는 웨이트리스트인 경우가 많습니다.
# 권한이 있다고 가정했을 때의 코드 구조입니다.


def generate_solution_video(visual_prompt, output_filename="solution.mp4"):
    print("🎥 비디오 생성 중... (시간이 소요될 수 있습니다)")
    try:
        operation = client.models.generate_videos(
            model="veo-3.0-generate-001",
            prompt=visual_prompt,
            config=types.GenerateVideosConfig(
                aspect_ratio="9:16",
                duration_seconds=4,
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
    print("--- 🛠️ AI 해결책 생성기 (First 기능) ---")
    
    # 1. 대화 내용 가져오기
    conversation_context = get_latest_conversation_context()
    
    if not conversation_context:
        print("대화 내용을 불러오지 못해 기본 예제로 진행합니다.")
        conversation_context = "User: 세탁기 배수가 안돼요. 어떻게 해야 하나요?"

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
        
        # 이미지 생성
        output_filename = output_dir / f"result_solution_{timestamp}.png"
        generate_solution_image(prompt, str(output_filename))

        # 영상 생성
        video_filename = output_dir / f"result_solution_{timestamp}.mp4"
        generate_solution_video(prompt, str(video_filename))
