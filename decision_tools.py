"""
소나무재선충병 정책결정 지원 — 핵심 도구 모듈
ML 모델(차년도 감염목 예측)을 정책 도구로 노출한다.
이 함수들이 그대로 (1) Groq 에이전트의 tool, (2) 향후 MCP 서버의 tool 이 된다.

도구:
  predict_damage(sgg_code)        : 특정 시군구의 차년도 감염목 예측
  get_priority_ranking(top_n,sido): 예측 피해 기준 방제 우선순위
  simulate_budget(...)            : 예산·가중치 시나리오별 배분
모든 수치는 학습데이터/모델에서 산출하며, 데이터에 없으면 빈 결과를 반환한다(환각 방지).
"""
import pandas as pd, numpy as np
from sklearn.linear_model import Ridge
from sgg_names import full_name

DATA = "재선충_ML학습데이터셋_2016_2023.csv"
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


# ---------- 도구 1: 개별 시군구 예측 ----------
def predict_damage(sgg_code: str) -> dict:
    e = engine()
    r = e.pred[e.pred['sgg_code'] == str(sgg_code)]
    if r.empty:
        return {"found": False, "sgg_code": sgg_code, "msg": "데이터에 없는 시군구코드"}
    r = r.iloc[0]
    return {"found": True, "sgg_code": sgg_code, "sido": r['sido_nm'],
            "name": full_name(sgg_code, r['sido_nm']),
            "base_year": int(r['base_year']), "predict_year": int(r['base_year']) + 1,
            "recent_infected": int(r['recent_infected']),
            "predicted_infected_next": int(r['pred_infected']),
            "treat_rate": round(float(r['treat_rate']), 3)}


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
                    top_n: int = 20, min_share: float = 0.0) -> dict:
    """예측 피해에 비례해 예산 배분. unit_cost_won=감염목 1본 방제 단가(기본 1.5만원, 근사치)."""
    e = engine()
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
             "budget_won": int(a), "treatable_trees": int(tr),
             "coverage": round(min(1.0, tr / r['pred_infected']), 3) if r['pred_infected'] > 0 else None}
            for (_, r), a, tr in zip(t.iterrows(), alloc, treatable)]
    return {"total_budget_won": int(total_budget_won), "unit_cost_won": int(unit_cost_won),
            "min_share": min_share, "allocations": rows}


if __name__ == "__main__":
    import json
    print("전국 우선순위 top5:")
    print(json.dumps(get_priority_ranking(5), ensure_ascii=False, indent=2))
    print("\n예산 1,000억 시나리오:")
    print(json.dumps(simulate_budget(100_000_000_000, top_n=5), ensure_ascii=False, indent=2))
