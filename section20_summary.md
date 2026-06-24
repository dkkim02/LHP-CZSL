# §20 Redefined-Axis Program — 1-Page Summary (as of 2026-06-01)

**목표.** Troika baseline의 cross-framework gap (mit-states, ViT-L/14: **−0.017 HM / −0.020 AUC** vs ClusPro 0.407/0.238)을 *재정의된 4개 축* C′/D′/E/F로 닫을 수 있는지 사전 등록(pre-registered) 검증.

**Baseline & 판정 기준 (모든 축 공통).**
- Troika best_loss 3-seed mean: **HM 0.3899 ± 0.0038 / AUC 0.2177** (single-seed §11-11 = 0.3940). 재현 자체는 Troika paper(0.392) 위 → gap은 ClusPro 대비 cross-framework.
- 시드 노이즈 큼(±0.0038, R2D2/LHP의 5배) → effective band ±0.011 HM(3σ).
- **Cutoff**: test HM **> 0.4050** (Δ > +0.011) → 유의(Stage 2 진입). 0~+0.011 → tie, 봉인. < 0 → fail, 봉인.

---

## 결과 (4축 전부 SEALED)

| 축 | 메커니즘 | 개입 강도(bite) | 결과 (seed0/3-seed) | Δ HM | 판정 |
|---|---|---|---|---|---|
| **C′** | logic-rule: attr mutex + obj–attr incompat soft penalty (λ=0.1) | base의 ~0.2% | 3-seed HM **0.3890** ± .0035 / AUC .2160 | **−0.0009** | **fail** → 봉인 (5-28) |
| **D′** | LLM semantic hard-neg InfoNCE (λ=0.1, τ=0.1) | ~0.8% | seed0 HM **0.3951** / AUC .2189 | **+0.0011** | **tie** → 봉인 (5-31) |
| **E** | LLM taxonomy aux CE (attr→18 state-groups, obj→21 super-cats) | **~3.3%** (최강) | seed0 HM **0.382** / AUC .2136 | **−0.012** | **fail** → 봉인 (6-1) |
| **F** | LLM multimodal test-time rerank (top-5 → re-pick) | inference-only | clean probe: top1 0.340→**0.320** | **−0.02** (net) | **null** → 봉인, $65 waive (6-1) |

---

## 핵심 발견

1. **개입을 강하게 할수록 더 나빠진다.** bite 0.2% → 0.8% → 3.3%로 단조 증가시켰으나 신호는 0 또는 음수, **E(최강 bite)가 가장 나쁨(−0.012)**. "auxiliary supervision을 강하게 주입하면 representation이 바뀐다"는 가설이 직접 반증됨 — coarse prior 주입이 fine-grained attribute 변별을 오히려 희석(과적합 가속, val HM ep2 정점 후 단조 하락).

2. **Inference-side rerank도 효과 없음.** F는 학습 변경 0인 test-time LLM 재랭킹. self-Claude 무료 probe(50장)로 측정 → **net −0.02**. zero-shot LLM은 *한 이미지에 동시 참인 속성이 여럿*일 때(huge↔inflated balloon, sliced↔browned potato) dataset의 single-label과 다른 "똑같이 맞는" 속성을 골라, 진짜 오류 복구(+5)보다 멀쩡한 예측 파괴(−6)가 많음. 학습 모델은 dataset labeling 편향을 학습했고 LLM은 모름.
   - *방법론 주의*: 1차 probe는 +0.22로 보였으나 **이미지 파일 경로에 정답 label이 박혀 leak**된 artifact. 익명 파일명 + gt-blind 평가자로 재측정 후 −0.02 확정. (vision-LLM eval 시 path/filename leak 필수 점검.)

3. **결론.** axis(학습 보조손실)·hyperparam·inference-rerank 어느 경로로도 Troika에서 ClusPro gap을 닫을 수 없음. **gap은 architectural** — Troika의 표현/구조 자체 한계. legacy 11축 + ckpt-selection + §20 4축 = **누적 16개 검증축이 모두 null/sealed.**

## 다음 단계
- axis/hyperparam/inference 변형의 expected value ≈ 0 확정. → **새 architecture/framework 진입 결정 국면.** (다음 세션 후보 brainstorm.)
- 재사용 산출물: `Troika/code/probe_f.py`(rerank harness), `Troika/data/F_probe/test_top5_preds.json`(test 12,995장 top-5 캐시).
- 상세: `RESEARCH_LOG.md` §20-4(C′)/§20-5(D′)/§20-6(E)/§20-7(F).
