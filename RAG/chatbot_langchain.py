import os
from dotenv import load_dotenv

# 1) LLM & Embedding (그대로 OK)
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

# 2) VectorStore (그대로 OK)
from langchain_community.vectorstores import SupabaseVectorStore

# 3) 예전 memory, chains → classic으로 이동
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

# 4) PromptTemplate는 core로 이동
from langchain_core.prompts import PromptTemplate

# 5) Supabase 클라이언트
from supabase import create_client, Client

from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document

# 맨 위 import 근처에 추가
from langchain_core.documents import Document
from langchain_community.vectorstores import SupabaseVectorStore


def _patched_similarity_search_by_vector_with_relevance_scores(
    self,
    query,
    k,
    filter=None,           # 지금은 Supabase 함수에 filter 인자 없으니 안 씀
    postgrest_filter=None, # 이것도 무시
    score_threshold=None,
):
    # ⚠ 여기가 Supabase RPC로 날아가는 파라미터
    # 에러에서 힌트 준 그대로:
    #   match_manual_sections(match_count, match_threshold, query_embedding)
    filter_model_id = None
    if isinstance(filter, dict):
        filter_model_id = filter.get("model_id")

    match_documents_params = {
        "match_count": k,
        "match_threshold": 0.7,   # 유사도 하한(예시). 원하면 0.6~0.8 사이로 조정
        "query_embedding": query,
        "filter_model_id": filter_model_id,
    }

    # Supabase RPC 호출
    response = self._client.rpc(self.query_name, match_documents_params).execute()
    rows = response.data or []

    docs_and_scores = []

    for row in rows:
        content = row.get("content", "")
        if not content:
            continue

        metadata = row.get("metadata") or {}
        score = float(row.get("similarity", 0.0))

        # LangChain 쪽에서 score_threshold 더 주면 여기서 한 번 더 필터링
        if score_threshold is not None and score < score_threshold:
            continue

        doc = Document(page_content=content, metadata=metadata)
        docs_and_scores.append((doc, score))

    # 혹시 Supabase 함수가 match_count보다 많이 돌려줘도 상위 k개만 사용
    if k is not None and len(docs_and_scores) > k:
        docs_and_scores = docs_and_scores[:k]

    return docs_and_scores


# SupabaseVectorStore에 패치 적용
SupabaseVectorStore.similarity_search_by_vector_with_relevance_scores = (
    _patched_similarity_search_by_vector_with_relevance_scores
)



# 1. 환경변수 및 키 설정
load_dotenv() # .env 파일이 있다면 로딩
SUPABASE_URL = "https://wzafalbctqkylhyzlfej.supabase.co"
SUPABASE_KEY = os.getenv("supbase_service_role")
GOOGLE_API_KEY = os.getenv("google_api")

# 2. Supabase & Gemini 설정
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# 임베딩 모델 (눈)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY,
    task_type="retrieval_query"
)

# LLM 모델 (두뇌) - Gemini 2.5 Flash
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.1 # 매뉴얼 답변이므로 창의성 낮춤
)

# 3. Vector Store 연결 (랭체인이 우리 DB를 쓸 수 있게 함)
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="manual_sections",
    query_name="match_manual_sections" # 아까 수정한 RPC 함수 이름
)

# 검색기(Retriever) 설정
# -> k=3: 3개만 찾아와라
# -> filter: F24WD 모델 것만 찾아라 (메타데이터 필터링 가능!)
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3} 
    # 필요하면 여기에 {'filter': {'model_id': '...'}} 로직 추가 가능
)

# 4. 기억(Memory) 장치 추가
# -> 대화 내용을 'chat_history'라는 키에 저장해서 계속 들고 다님
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# 5. 프롬프트 (페르소나) 설정
custom_template = """
당신은 LG전자 가전제품 전문 상담원 'ThinQ 봇'입니다.
아래의 [문맥(Context)]과 [대화 기록(Chat History)]을 바탕으로 사용자의 질문에 친절하게 답하세요.
만약 매뉴얼에 없는 내용이면 솔직하게 모른다고 말하세요.

1. 표 내용은 문장으로 자연스럽게 풀어서 설명하세요.
2. 항상 정중한 말투(하십시오체 또는 해요체)를 사용하세요.


[대화 기록]:
{chat_history}

[문맥(매뉴얼 검색 결과)]:
{context}

[사용자 질문]: {question}

[답변]:
"""

QA_PROMPT = PromptTemplate(
    template=custom_template,
    input_variables=["chat_history", "context", "question"]
)

# 6. 체인 생성 (두뇌 + 기억 + 검색기 연결)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True, # 답변 출처도 같이 반환
    combine_docs_chain_kwargs={"prompt": QA_PROMPT} # 커스텀 프롬프트 적용
)

def main():
    print("🤖 LG ThinQ 챗봇 (LangChain 버전) - 종료하려면 'exit' 입력")
    print("-" * 50)
    
    while True:
        query = input("\n나: ")
        if query.lower() == "exit":
            break
            
        # 랭체인에게 질문 던지기 (자동으로 검색하고, 기억해서 답변함)
        result = qa_chain.invoke({"question": query})
        
        print(f"\n🤖 봇: {result['answer']}")
        
        # (선택) 출처 확인
        # print("\n[참고한 매뉴얼]:")
        # for doc in result['source_documents']:
        #     print(f"- {doc.metadata.get('section_title', '제목없음')}")

if __name__ == "__main__":
    main()
