import os
import time
import google.generativeai as genai
from supabase import create_client, Client
from dotenv import load_dotenv

# ==========================================
# 1. 설정 정보
# ==========================================
load_dotenv()
SUPABASE_URL = "https://wzafalbctqkylhyzlfej.supabase.co"
SUPABASE_KEY = os.getenv("supbase_service_role") 
GOOGLE_API_KEY = os.getenv("google_api")        

genai.configure(api_key=GOOGLE_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==========================================
# 2. 임베딩 생성 함수
# ==========================================
def get_embedding(text):
    try:
        if not text or len(text.strip()) < 2:
            return None
            
        time.sleep(1.0) # 속도 조절
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        print(f"  ⚠️ 임베딩 에러 (잠시 대기): {e}")
        time.sleep(5)
        return None

# ==========================================
# 3. 메인 로직
# ==========================================
def process_existing_db_rows():
    print("🔄 DB에서 임베딩이 없는 데이터를 조회합니다...")

    # 🚨 수정된 부분: 'id' 대신 'section_id'를 사용합니다.
    response = supabase.table("manual_sections") \
        .select("section_id, content_text") \
        .is_("embedding_vector", "null") \
        .execute()
    
    rows = response.data
    
    if not rows:
        print("✅ 처리할 데이터가 없습니다. (모든 행에 임베딩이 이미 있습니다)")
        return

    print(f"📦 총 {len(rows)}개의 데이터를 찾아 업데이트를 시작합니다.")
    
    success_count = 0

    for idx, row in enumerate(rows):
        # 🚨 수정된 부분: 여기서도 section_id를 가져옵니다.
        current_id = row['section_id'] 
        text_content = row['content_text']
        
        print(f"[{idx+1}/{len(rows)}] ID:{current_id} 처리 중...", end="")

        vector = get_embedding(text_content)
        
        if vector:
            try:
                # 🚨 수정된 부분: 업데이트 조건도 section_id 기준입니다.
                supabase.table("manual_sections") \
                    .update({"embedding_vector": vector}) \
                    .eq("section_id", current_id) \
                    .execute()
                print(" -> ✅ 저장 완료")
                success_count += 1
            except Exception as e:
                print(f" -> ❌ DB 업데이트 실패: {e}")
        else:
            print(" -> ⚠️ 임베딩 생성 실패 (텍스트 확인 필요)")

    print(f"\n🎉 완료! 총 {success_count}개의 행이 업데이트되었습니다.")

if __name__ == "__main__":
    process_existing_db_rows()