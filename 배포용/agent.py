"""
소나무재선충병 정책결정 지원 AI 에이전트 (Groq 백엔드, 무료)
- ML 예측 모델을 도구로 호출해 '근거에 기반한' 답변/보고서/시나리오를 생성
- 포지셔닝: 공무원의 판단을 '대체'하지 않고 '지원'. 데이터에 없는 수치는 지어내지 않음.

실행:  python agent.py            (대화형)
       python agent.py "경북에서 가장 시급한 5곳은?"   (단발 질문)
"""
import os, json, sys, re, time, urllib.request, urllib.error
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
6) 데이터 해상도를 구분하라: 예산(get_budget)·발생방제 추세(get_national_trend)는 '전국 단위', 산림면적·침엽수림·재선충 실적(get_region_context/predict_damage)은 '시군구 단위', 기후는 '시도/근접 관측소 근사'다.
6-0) 특정 시군구(예: 진주)의 '과거/작년 집행 예산'을 물으면, 시군구별 예산 데이터는 없다고 분명히 말하라. get_budget의 전국 수치를 줄 때는 반드시 "전국 기준"이라고 명시하고, 그 지역의 예산이 아님을 밝혀라(전국값을 그 지역 값처럼 답하지 마라). 특정 시군구에 '얼마가 필요한가'를 물으면 predict_damage(region)의 '직접방제비_억원'을 그대로 인용하고, 이는 감염목 직접 제거비이며 전체 사업예산과 다름을 밝혀라(숫자를 임의로 만들지 마라).
6-1) "○○(시도)에 N억 배분/자치구 배분" 같은 질문은 개별 시군구를 일일이 조회하지 말고 simulate_budget(total_budget_won, sido="○○") '한 번'으로 처리하라. 도구 호출은 최소로 하라. 금액을 말할 때는 도구가 준 'budget_억원'·'total_budget_억원' 필드를 그대로 써라(원→억 변환을 직접 하지 마라, 0 개수 실수 방지). 예측 감염목이 0인 지역은 배분에서 빠지는 게 정상이며, 전액이 피해가 있는 지역에 집중될 수 있다.
6-1-2) 직전 배분/우선순위에 대해 "근거·이유·왜"를 물으면, 같은 배분 금액을 반복하지 말고 '차년도 예측 감염목 수(predicted_infected_next)에 비례해 배분했다'는 점을 밝히고 각 지역의 예측 감염목 수치를 제시하라. 필요하면 도구 결과의 predicted_infected_next를 근거로 다시 인용하라.
6-2) "전년 대비 예산을 얼마나 투입/증액/감액해야 하나" 같은 '필요예산 추정' 질문은 estimate_required_budget를 호출하라. 과거 수치만 나열하지 말고, 예측 피해 증감률(피해_증감률_pct)을 근거로 "전년 대비 약 ±X% 조정 검토"처럼 방향과 참고치를 제시하라. 직접제거비는 전체 예산의 일부일 뿐임을 밝혀라.
7) 도구 결과의 각 값을 그 라벨 그대로 정확히 사용하라. 특히 '발생면적'과 '방제면적', '예산현액'과 '결산'을 절대 뒤바꾸지 마라. 비교·증감을 말할 때는 같은 항목끼리(발생↔발생, 방제↔방제)만 비교하라.
8) 답변은 반드시 자연스러운 한국어로만 작성하라. 한자(漢字)나 다른 언어 문자를 절대 섞지 마라(예: '적습니다'를 '少습니다'로 쓰지 마라). 간결하고 근거 중심으로. 함수 호출 문법(<function=...>)을 답변 텍스트에 절대 출력하지 마라."""

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
            "sido": {"type": "string", "description": "특정 시도 내 시군구에만 배분할 때. 예:'대구'(대구 자치구 배분), '경북'"},
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
    {"type": "function", "function": {
        "name": "estimate_required_budget",
        "description": "'전년 대비 내년 예산을 얼마나 투입해야 하나' 같은 필요예산 추정. 예측 피해 증감률과 전년 집행예산 기반 참고치를 반환. '예산 얼마나 늘려야/줄여야' 류 질문에 사용.",
        "parameters": {"type": "object", "properties": {
            "sido": {"type": "string", "description": "특정 시도만 볼 때(생략시 전국). 단 예산은 전국 단위만 존재"}},
            "required": []}}},
]
FUNCS = {"predict_damage": T.predict_damage,
         "get_priority_ranking": T.get_priority_ranking,
         "simulate_budget": T.simulate_budget,
         "get_budget": T.get_budget,
         "get_national_trend": T.get_national_trend,
         "get_region_context": T.get_region_context,
         "estimate_required_budget": T.estimate_required_budget}


def _post(messages, _retries=2):
    body = json.dumps({"model": MODEL, "messages": messages, "tools": TOOLS,
                       "temperature": 0.2}).encode()
    req = urllib.request.Request(API, data=body,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
                 "User-Agent": "Mozilla/5.0"})
    for attempt in range(_retries + 1):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=60).read())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < _retries:  # 무료등급 호출한도 → 짧게 대기 후 재시도
                wait = int(e.headers.get("retry-after", 0)) or (3 * (attempt + 1))
                time.sleep(min(wait, 6))
                continue
            raise


# Llama가 구조화 tool_calls 대신 텍스트로 함수호출을 뱉는 경우 파싱:
#   <function=name>{json}</function>  또는  <function=name>{json}
_FN_TEXT = re.compile(r"<function=(\w+)>\s*(\{.*?\})\s*(?:</function>)?", re.DOTALL)


def _run(name, args, verbose):
    if verbose:
        print(f"  · 도구호출: {name}({args})")
    try:
        if name not in FUNCS:
            return {"error": f"알 수 없는 도구: {name}"}
        return FUNCS[name](**args)
    except Exception as e:
        return {"error": str(e)}


def _slim(msg):
    """모델 응답 메시지를 재전송용으로 정리(role/content/tool_calls만 유지)."""
    out = {"role": msg.get("role", "assistant")}
    out["content"] = msg.get("content") or ""
    if msg.get("tool_calls"):
        out["tool_calls"] = msg["tool_calls"]
    return out


def _converse(question, history, verbose):
    messages = [{"role": "system", "content": SYSTEM}]
    if history:
        for role, content in history[-8:]:
            if role in ("user", "assistant") and content and isinstance(content, str):
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    for _ in range(6):  # 최대 6회 도구 호출 루프
        msg = _post(messages)["choices"][0]["message"]
        messages.append(_slim(msg))
        calls = msg.get("tool_calls")
        if calls:
            for c in calls:
                name = c["function"]["name"]
                args = json.loads(c["function"]["arguments"] or "{}")
                result = _run(name, args, verbose)
                messages.append({"role": "tool", "tool_call_id": c["id"],
                                 "content": json.dumps(result, ensure_ascii=False)})
            continue
        # 안전장치: 구조화 호출이 없으면 content 안의 텍스트형 함수호출을 파싱
        content = msg.get("content", "") or ""
        found = _FN_TEXT.findall(content)
        if found:
            results = []
            for name, argstr in found:
                try:
                    args = json.loads(argstr)
                except Exception:
                    args = {}
                results.append(f"{name}({args}) → " +
                               json.dumps(_run(name, args, verbose), ensure_ascii=False))
            messages.append({"role": "user", "content":
                "다음은 도구 실행 결과다. 이것만 근거로 한국어로 자연스럽게 답하라. "
                "함수 호출 문법(<function=...>)은 절대 출력하지 마라.\n" + "\n".join(results)})
            continue
        return content
    return "(도구 호출이 너무 많아 중단)"


def ask(question: str, history=None, verbose=True) -> str:
    """history: [(role, content)] 직전 대화. '왜?' 같은 이어지는 질문 맥락 유지용.
    무료등급 토큰 한도를 고려해 최근 8개만 사용. 400이면 기록 없이 1회 재시도."""
    try:
        return _converse(question, history, verbose)
    except urllib.error.HTTPError as e:
        if e.code == 400 and history:  # 대화기록이 원인일 수 있으니 기록 없이 재시도
            return _converse(question, None, verbose)
        raise


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
