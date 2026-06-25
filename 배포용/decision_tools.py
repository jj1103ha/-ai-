"""
소나무재선충병 정책결정 지원 — 핵심 도구 모듈
ML 모델(차년도 감염목 예측)을 정책 도구로 노출한다.
이 함수들이 그대로 (1) Groq 에이전트의 tool, (2) 향후 MCP 서버의 tool 이 된다.

도구:
  predict_damage(region)          : 특정 시군구의 차년도 감염목 예측 (이름/코드)
  get_priority_ranking(top_n,sido): 예측 피해 기준 방제 우선순위
  simulate_budget(...)            : 예산·가중치 시나리오별 배분
  get_budget(year,kind)           : 전국 예산 집행/계획
  get_national_trend(year)        : 전국 발생·방제 면적 추이
  get_region_context(region)      : 시군구 산림·기후·재선충 종합
모든 수치는 학습데이터/모델/정제 데이터에서 산출하며, 없으면 빈 결과를 반환한다(환각 방지).
"""
import pandas as pd, numpy as np
from sklearn.linear_model import Ridge
from sgg_names import full_name, find_codes

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
def _p(*a): return os.path.join(_HERE, *a)
DATA = _p("재선충_ML학습데이터셋_2016_2023.csv")
FEATS = ['surveyed_lag1','infected_lag1','dead_lag1','infect_rate_lag1','treat_rate_lag1',
         'surveyed_lag2','infected_lag2','dead_lag2','infected_growth']

SIDO = {'11':'서울','26':'부산','27':'대구','28':'인천','29':'광주','30':'대전','31':'울산',
        '36':'세종','41':'경기','42':'강원','43':'충북','44':'충남','45':'전북','46':'전남',
        '47':'경북','48':'경남','50':'제주','51':'강원','52':'전북'}


class Engine:
    """데이터 로드 + 모델 학습 + 차년도(2024) 예측표 생성 (import 시 1회)."""
    def __init__(self):
        d = pd.read_csv(DATA, dtype={'sgg_code': str})
        d = d.dropna(subset=['infected_lag1']).copy()
        d[FEATS] = d[FEATS].fillna(0)
        self.df = d
        X = pd.concat([d[FEATS], pd.get_dummies(d['sido_nm'], prefix='sido')], axis=1)
        self.cols = list(X.columns)
        self.model = Ridge(alpha=10.0).fit(X[d['year'] <= 2021], d.loc[d['year'] <= 2021, 'infected'])
        self.pred = self._predict_next_year(d)

    def _row_to_X(self, rows):
        X = pd.concat([rows[FEATS], pd.get_dummies(rows['sido_nm'], prefix='sido')], axis=1)
        return X.reindex(columns=self.cols, fill_value=0)

    def _predict_next_year(self, d):
        # 각 시군구의 최신연도(2023) 실측을 lag 로 써서 차년도(2024) 감염목 예측
        last = d.sort_values('year').groupby('sgg_code').tail(1).copy()
        nxt = pd.DataFrame({
            'sgg_code': last['sgg_code'].values, 'sido_nm': last['sido_nm'].values,
            'surveyed_lag1': last['surveyed'].values, 'infected_lag1': last['infected'].values,
            'dead_lag1': last['dead'].values, 'infect_rate_lag1': last['infect_rate'].values,
            'treat_rate_lag1': last['treat_rate'].values,
            'surveyed_lag2': last['surveyed_lag1'].values, 'infected_lag2': last['infected_lag1'].values,
            'dead_lag2': last['dead_lag1'].values,
            'infected_growth': (last['infected'].values - last['infected_lag1'].values),
        })
        nxt['pred_infected'] = np.clip(self.model.predict(self._row_to_X(nxt)), 0, None).round().astype(int)
        nxt['recent_infected'] = last['infected'].values
        nxt['treat_rate'] = last['treat_rate'].values
        nxt['base_year'] = int(last['year'].max())
        return nxt.sort_values('pred_infected', ascending=False).reset_index(drop=True)


_E = None
def engine():
    global _E
    if _E is None:
        _E = Engine()
    return _E


# ---------- 내부: 지역명/코드 → 단일 코드 해석 ----------
def _resolve(region: str):
    """지역명 또는 코드 → (sgg_code, 에러dict). 성공 시 (code, None), 실패 시 (None, dict)."""
    cands = find_codes(region)
    if not cands:
        return None, {"found": False, "query": region,
                      "msg": f"'{region}'에 해당하는 시군구를 찾지 못했습니다. 시도명을 함께 적어주세요(예: '울산 북구')."}
    if len(cands) > 1:
        return None, {"found": False, "ambiguous": True, "query": region,
                      "candidates": [f"{c['sido']} {c['name']}" for c in cands],
                      "msg": f"'{region}'는 여러 곳입니다. 시도명을 함께 지정하세요: " +
                             ", ".join(f"{c['sido']} {c['name']}" for c in cands)}
    return cands[0]["sgg_code"], None


# ---------- 도구 1: 개별 시군구 예측 ----------
def predict_damage(region: str = None, sgg_code: str = None) -> dict:
    """차년도 감염목 예측. region에는 시군구 이름('울산 북구','밀양')이나 코드 모두 가능."""
    q = region if region is not None else sgg_code
    code, err = _resolve(q)
    if err:
        return err
    e = engine()
    r = e.pred[e.pred['sgg_code'] == str(code)]
    if r.empty:
        return {"found": False, "sgg_code": code, "msg": "데이터에 없는 시군구코드"}
    r = r.iloc[0]
    pred = int(r['pred_infected'])
    name = full_name(code, r['sido_nm'])
    level, note = _severity(pred, name)
    direct = pred * 15000  # 직접 방제(제거)비 = 예측 감염목 × 단가(1.5만원)
    return {"found": True, "sgg_code": code, "sido": r['sido_nm'],
            "name": name,
            "base_year": int(r['base_year']), "predict_year": int(r['base_year']) + 1,
            "recent_infected": int(r['recent_infected']),
            "predicted_infected_next": pred,
            "level": level, "note": note,
            "직접방제비_원": direct, "직접방제비_억원": round(direct / 1e8, 2),
            "직접방제비_주의": "감염목 제거비만 반영(예찰·예방·인건비 제외). 시군구별 실제 집행예산 데이터는 없음.",
            "treat_rate": round(float(r['treat_rate']), 3)}


def _severity(pred: int, name: str):
    """예측 감염목 수를 피해 수준 등급+설명문으로 변환(에이전트가 풀어서 답하도록)."""
    if pred <= 0:
        return "피해 거의 없음", f"{name}: 차년도 예측 감염목이 0본으로, 피해가 거의 없는 지역입니다."
    if pred <= 500:
        return "경미", f"{name}: 차년도 예측 감염목이 약 {pred:,}본으로, 피해가 경미한 지역입니다."
    if pred <= 3000:
        return "보통", f"{name}: 차년도 예측 감염목이 약 {pred:,}본으로, 피해가 보통 수준인 지역입니다."
    if pred <= 10000:
        return "심각", f"{name}: 차년도 예측 감염목이 약 {pred:,}본으로, 피해가 심각한 지역입니다."
    return "매우 심각", f"{name}: 차년도 예측 감염목이 약 {pred:,}본으로, 피해가 매우 심각한 지역입니다."


# ---------- 도구 2: 우선순위 ----------
def get_priority_ranking(top_n: int = 10, sido: str = None) -> dict:
    e = engine()
    t = e.pred.copy()
    if sido:
        t = t[t['sido_nm'] == sido]
    t = t.head(int(top_n))
    rows = [{"rank": i + 1, "name": full_name(r['sgg_code'], r['sido_nm']),
             "sgg_code": r['sgg_code'], "sido": r['sido_nm'],
             "predicted_infected_next": int(r['pred_infected']),
             "recent_infected": int(r['recent_infected']),
             "treat_rate": round(float(r['treat_rate']), 3)}
            for i, (_, r) in enumerate(t.iterrows())]
    return {"basis": "차년도 예측 감염목 수 기준", "filter_sido": sido or "전국", "items": rows}


# ---------- 도구 3: 예산 시나리오 ----------
def simulate_budget(total_budget_won: float, unit_cost_won: float = 15000,
                    top_n: int = 20, min_share: float = 0.0, regions=None, sido: str = None) -> dict:
    """예측 피해에 비례해 예산 배분. unit_cost_won=감염목 1본 방제 단가(기본 1.5만원, 근사치).
    sido: 특정 시도 내 시군구에만 배분(예: '대구' → 대구 자치구들).
    regions: 특정 지역만 비교 배분할 때 이름/코드 리스트(예: ['경남 밀양','울산 북구']).
    둘 다 생략 시 전국 상위 top_n."""
    e = engine()
    if regions:
        codes, unresolved = [], []
        for rg in regions:
            code, err = _resolve(rg)
            (unresolved if err else codes).append(err["msg"] if err else code)
        if unresolved:
            return {"msg": "일부 지역을 해석하지 못했습니다.", "errors": unresolved}
        t = e.pred[e.pred['sgg_code'].isin(codes)].copy()
        if t.empty:
            return {"msg": "지정한 지역이 예측표에 없습니다."}
    elif sido:
        t = e.pred[e.pred['sido_nm'] == sido].copy()
        if t.empty:
            return {"msg": f"'{sido}' 시도의 시군구가 예측표에 없습니다."}
    else:
        t = e.pred.head(int(top_n)).copy()
    w = t['pred_infected'].clip(lower=0).astype(float)
    if w.sum() == 0:
        return {"msg": "예측 피해가 0이라 배분 불가"}
    share = w / w.sum()
    if min_share > 0:  # 형평성: 최소 배분 보장 후 잔여를 피해비례
        n = len(share); floor = np.full(n, min_share)
        share = floor + (1 - floor.sum()) * share if floor.sum() < 1 else floor / floor.sum()
    alloc = (share * total_budget_won).round().astype(int)
    treatable = (alloc / unit_cost_won).round().astype(int)
    rows = [{"name": full_name(r['sgg_code'], r['sido_nm']),
             "sgg_code": r['sgg_code'], "sido": r['sido_nm'],
             "predicted_infected_next": int(r['pred_infected']),
             "budget_won": int(a), "budget_억원": round(int(a) / 1e8, 1),
             "treatable_trees": int(tr),
             "coverage": round(min(1.0, tr / r['pred_infected']), 3) if r['pred_infected'] > 0 else None}
            for (_, r), a, tr in zip(t.iterrows(), alloc, treatable)]
    return {"total_budget_won": int(total_budget_won), "total_budget_억원": round(int(total_budget_won) / 1e8, 1),
            "unit_cost_won": int(unit_cost_won), "min_share": min_share, "allocations": rows}


# ---------- 보조 데이터 로더 (data/ 정제본, 1회 캐시) ----------
_CACHE = {}
def _csv(fname):
    if fname not in _CACHE:
        _CACHE[fname] = pd.read_csv(_p("data", fname), encoding="utf-8-sig", dtype={"sgg_code": str})
    return _CACHE[fname]


# ---------- 도구 4: 예산 (집행 실적 / 중장기 계획) ----------
def get_budget(year: int = None, kind: str = "집행") -> dict:
    """전국 산림병해충방제 예산. kind='집행'(2019~2024 단위사업 결산·집행률, 백만원),
    kind='계획'(2021~2030 예찰·방제계획 전략별 소요예산, 억원)."""
    try:
        if kind == "계획":
            d = _csv("예산_계획_2021_2030.csv")
            cols = [c for c in d.columns if c.isdigit()]
            if year:
                y = str(year)
                if y not in cols:
                    return {"found": False, "msg": f"{year}년 계획 데이터 없음(범위 2021~2030)"}
                items = [{"전략": r["전략"], "소요예산_억원": r[y]} for _, r in d.iterrows()]
                return {"found": True, "kind": "계획", "year": year, "unit": "억원", "items": items}
            return {"found": True, "kind": "계획", "unit": "억원", "years": cols,
                    "items": d.to_dict("records")}
        else:
            d = _csv("예산_집행_2019_2024.csv")
            if year:
                r = d[d["연도"] == int(year)]
                if r.empty:
                    return {"found": False, "msg": f"{year}년 집행 데이터 없음(범위 2019~2024)"}
                r = r.iloc[0]
                return {"found": True, "kind": "집행", "year": int(year), "unit": "백만원",
                        "예산현액": int(r["예산현액_백만원"]), "결산": int(r["결산_백만원"]),
                        "집행률_pct": float(r["집행률_pct"])}
            return {"found": True, "kind": "집행", "unit": "백만원", "items": d.to_dict("records")}
    except Exception as e:
        return {"found": False, "msg": f"예산 데이터 로드 오류: {e}"}


# ---------- 도구 5: 전국 발생/방제 면적 추이 ----------
def get_national_trend(year: int = None) -> dict:
    """전국 산림병해충 발생·방제 면적 추이(2014~2024, ha, 전체 해충 '계' 기준)."""
    try:
        d = _csv("전국_발생방제_2014_2024.csv")
        if year:
            r = d[d["연도"] == int(year)]
            if r.empty:
                return {"found": False, "msg": f"{year}년 데이터 없음(범위 2014~2024)"}
            r = r.iloc[0]
            return {"found": True, "year": int(year), "unit": "ha",
                    "발생면적": int(r["발생면적_ha"]), "방제면적": int(r["방제면적_ha"])}
        return {"found": True, "unit": "ha", "basis": "전체 산림병해충 '계'(방제는 천ha→ha 환산)",
                "items": d.to_dict("records")}
    except Exception as e:
        return {"found": False, "msg": f"추세 데이터 로드 오류: {e}"}


# ---------- 도구 7: 차년도 필요 예산 추정 (전년 대비) ----------
def estimate_required_budget(unit_cost_won: float = 15000, sido: str = None) -> dict:
    """차년도 예측 피해를 근거로 '전년 대비 예산을 얼마나 투입해야 하는지'를 추정한다.
    핵심 논리: 예측 감염목이 전년 대비 ±X% 변하면 방제 예산도 그에 준해 조정 검토.
    단정이 아니라 근거(증감률)+참고치를 제시한다(최종 판단은 공무원)."""
    e = engine()
    t = e.pred if not sido else e.pred[e.pred['sido_nm'] == sido]
    if sido and t.empty:
        return {"found": False, "msg": f"'{sido}' 시도 데이터가 없습니다."}
    tot_pred = int(t['pred_infected'].sum())
    tot_recent = int(t['recent_infected'].sum())
    chg = (tot_pred - tot_recent) / tot_recent if tot_recent else None
    out = {"found": True, "scope": sido or "전국",
           "predict_year": int(e.pred['base_year'].iloc[0]) + 1,
           "예측_감염목_합계": tot_pred, "전년_감염목_합계": tot_recent,
           "피해_증감률_pct": round(chg * 100, 1) if chg is not None else None,
           "직접제거비_원": int(tot_pred * unit_cost_won),
           "직접제거비_주의": "감염목 제거비만 반영. 예찰·예방·인건비 등 전체 사업예산의 일부임."}
    try:
        b = _csv("예산_집행_2019_2024.csv").sort_values("연도")
        last = b.iloc[-1]
        prev_year, prev_mw = int(last["연도"]), int(last["결산_백만원"])
        out["전년도"] = prev_year
        out["전년_집행예산_백만원"] = prev_mw
        if sido:  # 예산은 전국 단위만 존재 → 시도 증감률을 전국예산에 곱하지 않음
            out["예산_단위_주의"] = "예산은 전국 단위만 존재(시도별 예산 데이터 없음). 전년 예산은 전국 기준."
            if chg is not None:
                out["해석"] = (f"{sido}의 내년 예측 피해가 전년 대비 {chg*100:+.1f}%다. "
                              f"시도별 예산 데이터가 없어 금액 환산은 어렵고, 증감률을 조정 방향의 근거로 활용하라.")
        elif chg is not None:
            out["피해비례_참고예산_백만원"] = round(prev_mw * (1 + chg))
            out["해석"] = (f"내년 예측 피해가 전년 대비 {chg*100:+.1f}%이므로, "
                          f"전년 집행예산({prev_mw:,}백만원)을 피해에 비례시키면 "
                          f"약 {round(prev_mw*(1+chg)):,}백만원 수준. 정책적 조정폭은 공무원이 판단.")
    except Exception as ex:
        out["예산비교_오류"] = str(ex)
    return out


# ---------- 도구 6: 시군구 종합 컨텍스트 (산림·기후·재선충) ----------
def get_region_context(region: str) -> dict:
    """특정 시군구의 산림면적·침엽수림 면적·임목축적·기후(시도/근접 관측소 근사) +
    최신 재선충 감염·방제 실적 + 차년도 예측을 한 번에 반환."""
    code, err = _resolve(region)
    if err:
        return err
    out = {"found": True, "sgg_code": code}
    try:
        c = _csv("시군구_산림_기후.csv")
        row = c[c["sgg_code"] == str(code)]
        if not row.empty:
            r = row.iloc[0]
            def g(k):
                v = r[k]
                return None if pd.isna(v) else (int(v) if float(v).is_integer() else float(v))
            out.update({
                "name": f"{r['sido']} {r['name']}", "sido": r["sido"],
                "산림면적_ha": g("forest_area_ha"), "침엽수림면적_ha": g("conifer_area_ha"),
                "임목축적_m3": g("growing_stock_m3"),
                "기후_관측소": r["climate_station"], "여름평균기온": g("summer_temp"),
                "폭염일수": g("heat_days"), "연강수량_mm": g("annual_precip"),
                "기후_주의": "기후는 시군구명 일치 또는 시도 대표 관측소 기준 근사치",
            })
    except Exception as e:
        out["산림기후_오류"] = str(e)
    # 최신 재선충 실적 + 예측
    pred = predict_damage(sgg_code=code)
    if pred.get("found"):
        out.update({"최근_감염목": pred["recent_infected"], "방제율": pred["treat_rate"],
                    "차년도_예측감염목": pred["predicted_infected_next"], "피해수준": pred["level"]})
    return out


if __name__ == "__main__":
    import json
    print("전국 우선순위 top5:")
    print(json.dumps(get_priority_ranking(5), ensure_ascii=False, indent=2))
    print("\n예산 1,000억 시나리오:")
    print(json.dumps(simulate_budget(100_000_000_000, top_n=5), ensure_ascii=False, indent=2))
