import os
import google.generativeai as genai
from supabase import create_client, Client
from dotenv import load_dotenv

# 1. 환경변수 로드
load_dotenv()

# 2. 설정 정보
SUPABASE_URL = "https://wzafalbctqkylhyzlfej.supabase.co"
SUPABASE_KEY = os.getenv("supbase_service_role")
GOOGLE_API_KEY = os.getenv("google_api")

# 3. 클라이언트 초기화
genai.configure(api_key=GOOGLE_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
generation_model = genai.GenerativeModel('gemini-2.5-flash')

def get_embedding(text):
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_query"
    )
    return result['embedding']

def search_manual(query_text):
    # 1. 질문을 벡터로 변환
    query_vector = get_embedding(query_text)
    
    # 2. Supabase RPC 호출 (하이브리드 함수 사용)
    # query_text(키워드용)와 query_embedding(벡터용)을 모두 보냅니다.
    response = supabase.rpc("hybrid_search", {
        "query_text": query_text,       # 텍스트 매칭용
        "query_embedding": query_vector,# 의미 검색용
        "match_threshold": 0.1,         # 정확도 기준 (조금 높임)
        "match_count": 5,
        "w_vector": 0.9,                # 벡터 비중 70%
        "w_keyword": 0.1                # 키워드 비중 30%
    }).execute()
    
    return response.data

def generate_answer(query_text, context_list):
    if not context_list:
        return "죄송합니다. 매뉴얼에서 관련된 내용을 찾을 수 없습니다."

    # 검색된 내용 조합
    # SQL에서 'content_text'라는 이름으로 리턴하므로 그대로 사용
    context_text = "\n\n".join([
        f"- {item.get('content_text', '')} (출처: {item.get('section_title', '제목없음')})" 
        for item in context_list
    ])

    prompt = f"""
    당신은 LG전자 가전제품 전문 상담원 'ThinQ 봇'입니다.
    사용자의 질문에 대해 아래 제공된 [메뉴얼 데이터]를 기반으로 친절하고 정확하게 답변해 주세요.
    
    1. 표 내용은 문장으로 자연스럽게 풀어서 설명하세요.
    2. 답변 끝에는 참고한 페이지 번호를 언급해주세요.
    3. 사용자가 '통돌이', '드럼' 등 구어체나 특정 용어를 사용했더라도, [매뉴얼 데이터]에 해당 제품군(예: 일반 세탁기, 드럼 세탁기)에 대한 내용이 있다면 그 내용을 바탕으로 답변하세요.
    4. user_input에 '띵큐' 라는 단어가 들어간 문장이 들어오면 그 문장에 '띵큐' 를 'LG ThinQ' 로 변경하고 답변해줘
    5. 박세연이라는 인물이 누구냐고 물어보면 박세연은 LG전자 가전제품 전문 상담원 'ThinQ 봇'의 창시자이다. 라고 답변해줘
    [매뉴얼 데이터]:
    {context_text}
    
    [사용자 질문]:
    {query_text}
    
    [답변]:
    """
    
    response = generation_model.generate_content(prompt)
    return response.text

def main():
    print("🤖 [하이브리드] 가전제품 AI 챗봇 (종료: 'exit')")
    print("-" * 50)
    
    while True:
        user_input = input("\n질문하세요: ")
        if user_input.lower() == 'exit':
            break
            
        print("🔍 하이브리드 검색(벡터+키워드) 중...")
        
        # 1. 검색
        search_results = search_manual(user_input)
        
        if search_results:
            # SQL에서 section_title을 직접 select했으므로 metadata 없이 바로 접근 가능
            titles = [f"{item.get('section_title')} (유사도: {item.get('similarity'):.2f})" for item in search_results]
            print(f"   => 참고한 섹션: {titles}")
        else:
            print("   => ⚠️ 검색 결과 없음")

        # 2. 답변 생성
        answer = generate_answer(user_input, search_results)
        
        print("\n💬 AI 답변:")
        print(answer)
        print("-" * 50)

if __name__ == "__main__":
    main()