import os
import re
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import PlainTextResponse, JSONResponse
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
from llm import get_rag_chain
from llm import get_dictionary_chain
from llm import get_llm
from fastapi.middleware.cors import CORSMiddleware
import textwrap
import json

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

async def verify_frontend_token(request: Request):
    token = request.headers.get("X-FRONTEND-TOKEN")
    expected_token = os.getenv("X_FRONTEND_TOKEN")
    if token != expected_token:
        raise HTTPException(
            status_code=403,
            detail={
                "success": False,
                "detail": "Invalid FrontEnd token"
            }
        )


@app.post("/chat", response_class=PlainTextResponse)
async def chat_with_ai(request: Request, _: None = Depends(verify_frontend_token)):
    payload = await request.json()
    user_message = payload.get("message", "")
    session_id = payload.get("session_id", "default-session")

    def response_stream():
        chain = get_rag_chain()
        dictionary_chain = get_dictionary_chain()
        tax_chain = {"input": dictionary_chain} | chain
        stream = tax_chain.stream(
            {"question": user_message},
            config={"configurable": {"session_id": session_id}},
        )
        for chunk in stream:
            yield chunk

    return StreamingResponse(response_stream(), media_type="text/plain")

@app.post("/diagnosis", response_class=JSONResponse)
async def diagnosis_with_ai(request: Request, _: None = Depends(verify_frontend_token)):
    
    payload = await request.json()
    keywords = payload.get("keywords", "")
    description = payload.get("description", "")
    llm = get_llm()

    request_message = f"""
        이것이 핵심 키워드입니다. {keywords} 그리고 사용자의 환경 설명은 다음과 같습니다. {description}
        이 둘을 조합하여 진단결과를 2~3 문장으로 요약해주세요. 
        그리고 추정보다는 발생 가능하다고 문장을 만들어주세요.
    """
    diagnosis_result = f"""
        그리고 사용자의 환경 설명은 다음과 같습니다. {description}
        진단 결과를 위험 순으로 매우 높음, 높음, 보통, 낮음, 매우 낮음 중 하나로 진단 결과를 평가해주세요.
        참고로 매우 높음은 위험이 높다는 것을 의미합니다. 설명 대신 5가지 중 하나로만 대답해주세요.
        반드시 content 안에는 매우 높음, 높음, 보통, 낮음, 매우 낮음 중 하나로만 대답해주세요.
    """

    response = llm.invoke(request_message)
    diagnosis_response = llm.invoke(diagnosis_result)
    
    valid_levels = ["매우 높음", "높음", "보통", "낮음", "매우 낮음"]

    response_text = diagnosis_response.content
    diagnosis_level = None
    for level in valid_levels:
        if level in response_text:
            diagnosis_level = level
            break
    return { "content" : response.content, "diagnosis": diagnosis_level }


@app.post("/checklist", response_class=JSONResponse)
async def checklist_with_ai(request: Request, _: None = Depends(verify_frontend_token)):
    
    payload = await request.json()

    types = payload.get("type", "")
    jobType = payload.get("jobType", "")
    subJobType = payload.get("subJobType", "")
    reason = payload.get("reason", "")
    difficulty = payload.get("difficulty", "")
    riskLevel = payload.get("riskLevel", "")
    workTime = payload.get("workTime", "")

    llm = get_llm()

    prompt = f"""
    당신은 산업안전 분야의 전문 컨설턴트입니다.

    다음과 같은 작업 조건을 가진 사업장을 대상으로, 작업 전/작업 중/작업 후 단계별로 반드시 점검해야 할 안전 항목을 각각 5개씩 작성해주세요.

    작업 조건:
    - 공종(업종 분류): {types}
    - 작업명(예: 전선 정비, 컨테이너 이동): {jobType}
    - 단위 작업명(예: 고소작업, 지게차 운전): {subJobType}
    - 사고 원인 또는 유사 사고 사례: {reason}
    - 작업 난이도: {difficulty}
    - 위험도 수준: {riskLevel}
    - 작업 시간대: {workTime}

    작성 시 유의할 점:
    - 조건 조합을 바탕으로, 해당 작업에서 실제로 자주 발생하는 사고나 위험 시나리오를 예측하고, 이에 기반해 실질적인 예방 조치 중심으로 항목을 구성해주세요.
    - 형식적인 문구 대신, 실행 가능한 점검 문장으로 작성해주세요.
    - 중복되는 일반 문구는 제외하고, 입력 조건에 특화된 맞춤형 점검 항목을 만들어주세요.
    - importance 필드는 "필수", "권장", "참고" 3가지로 구분해주세요. 반드시 모두 필수일 필요는 없습니다.

    응답 형식(JSON):
    [
        {{
            "stage": "작업 전",
            "content": "작업 전, 가설 전기설비의 접지 상태를 점검하였습니까?",
            "importance": "필수"
        }},
    ...
    ]
    """
    response = llm.invoke(prompt)
    response_text = response.content.strip()
    print(f"Raw response:\n{repr(response_text)}")

    if not response_text:
        return {"error": "Empty response received from LLM"}
    
    match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
    if not match:
        return {"error": "JSON 블록이 응답에 포함되어 있지 않습니다."}

    json_block = match.group(1).strip()

    try:
        parsed_response = json.loads(json_block)
        
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON response: {str(e)}"}
    return {"content": parsed_response}