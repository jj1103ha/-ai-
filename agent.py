"""
소나무재선충병 정책결정 지원 AI 에이전트 (Groq 백엔드, 무료)
- ML 예측 모델을 도구로 호출해 '근거에 기반한' 답변/보고서/시나리오를 생성
- 포지셔닝: 공무원의 판단을 '대체'하지 않고 '지원'. 데이터에 없는 수치는 지어내지 않음.

실행:  python agent.py            (대화형)
       python agent.py "경북에서 가장 시급한 5곳은?"   (단발 질문)
"""
import os, json, sys, urllib.request
import decision_tools as T

MODEL = "llama-3.3-70b-versatile"
API = "https://api.groq.com/openai/v1/chat/completions"


def _load_key():
    k = os.environ.get("GROQ_API_KEY")
    if not k and os.path.exists(".env"):
        for line in open(".env", encoding="utf-8"):
            if line.strip().startswith("GROQ_API_KEY"):
                k = line.split("=", 1)[1].strip()
    if not k:
        sys.exit("GROQ_API_KEY 가 없습니다. .env 파일을 확인하세요.")
    return k


KEY = _load_key()

SYSTEM = """너는 산림청 공무원의 '소나무재선충병 방제 정책결정'을 돕는 지원 에이전트다.
원칙:
1) 모든 수치는 반드시 제공된 도구(predict_damage/get_priority_ranking/simulate_budget)를 호출해 얻어라. 절대 숫자를 추측하지 마라.
2) 도구 결과에 없는 값은 "데이터에 없음"이라고 솔직히 말하라.
3) 너는 결정을 대신하지 않는다. '근거와 선택지'를 제시하고, 최종 판단·책임은 공무원에게 있음을 전제로 한다.
4) 모델 한계(시군구×연도 표본 제한, 시도단위 일부 변수, 단가는 근사치)를 필요시 짚어라.
5) 답변은 한국어로, 간결하고 근거 중심으로."""

TOOLS = [
    {"type": "function", "function": {
        "name": "predict_damage",
        "description": "특정 시군구(법정동코드 앞 5자리)의 차년도 감염목 수를 예측",
        "parameters": {"type": "object", "properties": {
            "sgg_code": {"type": "string", "description": "시군구 코드 5자리 예:47830"}},
            "required": ["sgg_code"]}}},
    {"type": "function", "function": {
        "name": "get_priority_ranking",
        "description": "예측 피해 기준 방제 우선순위 시군구 목록. sido로 특정 시도만 필터 가능",
        "parameters": {"type": "object", "properties": {
            "top_n": {"type": "integer", "description": "상위 몇 개"},
            "sido": {"type": "string", "description": "예:경북, 경남 (생략시 전국)"}},
            "required": ["top_n"]}}},
    {"type": "function", "function": {
        "name": "simulate_budget",
        "description": "총예산을 예측 피해에 비례 배분하는 시나리오. min_share로 형평성(최소배분) 옵션",
        "parameters": {"type": "object", "properties": {
            "total_budget_won": {"type": "number", "description": "총 예산(원)"},
            "unit_cost_won": {"type": "number", "description": "감염목 1본 방제단가(원), 기본 15000"},
            "top_n": {"type": "integer", "description": "배분 대상 상위 시군구 수"},
            "min_share": {"type": "number", "description": "0~1, 각 지역 최소 배분비율"}},
            "required": ["total_budget_won"]}}},
]
FUNCS = {"predict_damage": T.predict_damage,
         "get_priority_ranking": T.get_priority_ranking,
         "simulate_budget": T.simulate_budget}


def _post(messages):
    body = json.dumps({"model": MODEL, "messages": messages, "tools": TOOLS,
                       "temperature": 0.2}).encode()
    req = urllib.request.Request(API, data=body,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
                 "User-Agent": "Mozilla/5.0"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def ask(question: str, verbose=True) -> str:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]
    for _ in range(6):  # 최대 6회 도구 호출 루프
        msg = _post(messages)["choices"][0]["message"]
        messages.append(msg)
        calls = msg.get("tool_calls")
        if not calls:
            return msg.get("content", "")
        for c in calls:
            name = c["function"]["name"]
            args = json.loads(c["function"]["arguments"] or "{}")
            if verbose:
                print(f"  · 도구호출: {name}({args})")
            try:
                result = FUNCS[name](**args)
            except Exception as e:
                result = {"error": str(e)}
            messages.append({"role": "tool", "tool_call_id": c["id"],
                             "content": json.dumps(result, ensure_ascii=False)})
    return "(도구 호출이 너무 많아 중단)"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(ask(" ".join(sys.argv[1:])))
    else:
        print("재선충 정책결정 지원 에이전트 (종료: quit)\n")
        while True:
            try:
                q = input("질문> ").strip()
            except EOFError:
                break
            if q.lower() in ("quit", "exit", "q", ""):
                break
            print("\n" + ask(q) + "\n")
