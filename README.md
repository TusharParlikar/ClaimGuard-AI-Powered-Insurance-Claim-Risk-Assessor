# ClaimGuard — AI-Powered Insurance Claim Risk Assessor

**Stack:** FastAPI + scikit-learn (ML) + LangChain. Exactly these three.

**Concept it clears:** ML makes the decision, LangChain explains the decision in plain English. This "predict + explain" pattern is a real production pattern (fraud, credit risk, medical triage) — small enough to build in a day, but it's a genuine talking point for interviews at both product and service companies.

---

## What it does

A claims adjuster submits a claim → the system flags it as **Low / Medium / High** risk → and instead of a bare label, they get a short, human-readable reason why.

```
POST /assess-claim
{
  "claim_amount": 8500,
  "policy_age_months": 4,
  "claims_filed_last_year": 3,
  "claim_description": "Vehicle damage from parking lot collision, no witnesses"
}

→ {
  "risk_level": "High",
  "risk_score": 0.82,
  "explanation": "New policy (4 months) combined with 3 prior claims this year is a strong fraud indicator, and the claim lacks witness corroboration.",
  "suggested_action": "Route to manual review before payout"
}
```

---

## The 3 layers

**1. ML (scikit-learn)**
- `RandomForestClassifier` or `GradientBoostingClassifier` on structured features: claim amount, policy age, claim frequency, claim type
- Train on a small public dataset — search Kaggle for "insurance claims fraud" or "auto insurance claims" (a few thousand rows is plenty)
- Output: risk label + probability score, pickled to `model.pkl`

**2. LangChain**
- One `PromptTemplate` + chain (`LLMChain` or `Runnable`)
- Input: the structured features + the ML model's label/score
- Output: 1–2 sentence plain-English reasoning + a suggested next action
- No memory, no tools — a single deterministic call per request

**3. FastAPI**
- One route: `POST /assess-claim`
- Pydantic request model (the claim fields) and response model (risk_level, risk_score, explanation, suggested_action)
- Loads `model.pkl` once at startup, not per-request
- Calls ML → feeds result into the LangChain chain → returns combined JSON

---

## File structure (single project, minimal)

```
claimguard/
├── train_model.py      # trains + pickles the sklearn model
├── model.pkl
├── chain.py             # LangChain PromptTemplate + chain
├── main.py               # FastAPI app, single endpoint
└── requirements.txt   # fastapi, uvicorn, scikit-learn, langchain, pandas
```

---

## Build order (fits a single sitting)

1. Get/clean a small insurance claims dataset (structured features + fraud label)
2. Train + pickle the classifier — confirm it predicts sanely on a few rows
3. Write the LangChain chain — test it standalone with a hardcoded label/score first
4. Wire both into the FastAPI endpoint
5. Test end-to-end with `curl` or the FastAPI docs UI (`/docs`)

---

## Why this is resume-worthy

- Fraud/risk detection is a recognizable, high-value use case at both product companies (fintech, insurtech) and service companies (BFSI clients)
- Demonstrates you can combine classic ML with LLM reasoning — not just "called an API," but a coherent hybrid system
- Small enough to explain fully in a 2-minute interview answer, end to end
