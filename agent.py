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
2) 시군구를 조회할 때는 '이름'을 그대로 넘겨라(예: predict_damage(region="울산 북구")). 법정동코드를 절대 네 기억으로 추측하지 마라 — 도구가 이름을 코드로 변환한다.
3) 도구가 'ambiguous'(여러 후보)를 반환하면, 후보 목록을 사용자에게 보여주고 시도명을 함께 물어 다시 조회하라. 'found:false'면 그 이유(미매칭/모호)를 그대로 전하고, 함부로 "데이터에 없음"으로 단정하지 마라.
3-1) predict_damage가 found:true면 예측 감염목 수(predicted_infected_next)와 함께 level/note를 활용해 피해 수준을 '풀어서' 설명하라. 특히 예측이 0이거나 작을 때 숫자만 던지지 말고 "피해가 거의 없는/경미한 지역입니다"처럼 의미를 전하라.
4) 너는 결정을 대신하지 않는다. '근거와 선택지'를 제시하고, 최종 판단·책임은 공무원에게 있음을 전제로 한다.
5) 모델 한계(시군구×연도 표본 제한, 시도단위 일부 변수, 단가는 근사치)를 필요시 짚어라.
6) 데이터 해상도를 구분하라: 예산(get_budget)·발생방제 추세(get_national_trend)는 '전국 단위', 산림면적·침엽수림·재선충 실적(get_region_context/predict_damage)은 '시군구 단위', 기후는 '시도/근접 관측소 근사'다. 시군구별 예산 실적은 데이터가 없으니, 예산 질문은 전국 실적(get_budget) 또는 예측피해 기반 배분(simulate_budget)으로 답하라.
7) 답변은 한국어로, 간결하고 근거 중심으로."""

TOOLS = [
    {"type": "function", "function": {
        "name": "predict_damage",
        "description": "특정 시군구의 차년도 감염목 수를 예측. 시군구 '이름'을 넘기면 내부에서 코드로 변환한다. 코드를 추측하지 말 것.",
        "parameters": {"type": "object", "properties": {
            "region": {"type": "string", "description": "시군구 이름 또는 '시도 시군구'. 예:'울산 북구','경남 밀양','당진','제주시'"}},
            "required": ["region"]}}},
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
            "top_n": {"type": "integer", "description": "배분 대상 상위 시군구 수(전국 기준)"},
            "min_share": {"type": "number", "description": "0~1, 각 지역 최소 배분비율"},
            "regions": {"type": "array", "items": {"type": "string"},
                        "description": "특정 지역만 비교 배분할 때 이름/코드 리스트. 예:['경남 밀양','울산 북구']"}},
            "required": ["total_budget_won"]}}},
    {"type": "function", "function": {
        "name": "get_budget",
        "description": "전국 산림병해충방제 예산 조회. kind='집행'(2019~2024 결산·집행률,백만원) 또는 '계획'(2021~2030 전략별 소요예산,억원). 전국 단위 데이터임.",
        "parameters": {"type": "object", "properties": {
            "year": {"type": "integer", "description": "조회 연도(생략시 전체)"},
            "kind": {"type": "string", "enum": ["집행", "계획"], "description": "집행 실적 또는 중장기 계획"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "get_national_trend",
        "description": "전국 산림병해충 발생·방제 면적 추이(2014~2024, ha). 전국 단위.",
        "parameters": {"type": "object", "properties": {
            "year": {"type": "integer", "description": "조회 연도(생략시 전체)"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "get_region_context",
        "description": "특정 시군구의 산림면적·침엽수림면적·임목축적·기후(근사)·최신 재선충 실적·차년도 예측을 종합 반환. 지역 이름을 넘길 것.",
        "parameters": {"type": "object", "properties": {
            "region": {"type": "string", "description": "시군구 이름. 예:'경남 밀양','울주군'"}},
            "required": ["region"]}}},
]
FUNCS = {"predict_damage": T.predict_damage,
         "get_priority_ranking": T.get_priority_ranking,
         "simulate_budget": T.simulate_budget,
         "get_budget": T.get_budget,
         "get_national_trend": T.get_national_trend,
         "get_region_context": T.get_region_context}


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
