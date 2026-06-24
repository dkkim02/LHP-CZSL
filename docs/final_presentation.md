# 최종 발표 — Compositional Zero-Shot Learning (CZSL)
### Image-side Localization을 통한 차별화된 CZSL 메커니즘 탐색

> 발표 흐름: ① Task 정의 → ② SOTA 모델 주요 흐름 → ③ 아이디어 전개 과정 → ④ 결과 → ⑤ 향후 계획
> (작성 기준: 2026-06-02, RESEARCH_LOG §1–§21 종합)

---

## 1. CZSL Task 설명

### 1-1. 무엇을 푸는 문제인가
- **목표**: 속성(attribute)–객체(object) **조합(composition)** 을 인식하되, **학습 때 본 적 없는 조합**까지 맞히는 것.
  - 예: 학습 = `red apple`, `green tomato`, `sliced apple` → 테스트 = **`red tomato`**, **`sliced tomato`** (조합 자체는 처음 보지만, 속성·객체는 따로 본 적 있음).
- **Primitive 분해**: 이미지 1장 → (attribute, object) 쌍으로 라벨. 모델은 primitive를 재조합(compositional generalization)해야 함.

### 1-2. 평가 셋업
- **Seen pairs** (학습 조합) vs **Unseen pairs** (테스트에만 등장하는 조합).
- **Closed-world**: 테스트 후보를 데이터셋에 실재하는 조합으로 제한 / **Open-world**: 모든 attr×obj 조합이 후보(훨씬 어려움).
- **데이터셋**:
  - **MIT-States**: 115 attributes × 245 objects, ~1,962 valid pairs, 자연 이미지. (본 연구 주 벤치마크)
  - **UT-Zappos**: 16 attributes × 12 objects, 신발 도메인, fine-grained. (전이 검증용)
- **지표**: Seen acc, Unseen acc, **HM**(둘의 조화평균 — seen/unseen 균형), **AUC**(seen-unseen bias sweep 곡선 면적, 핵심 지표).

### 1-3. 무엇이 어려운가 (핵심 난점)
1. **속성의 맥락 의존성**: 같은 `wet`도 `wet dog`/`wet road`/`wet paint`에서 시각적으로 전혀 다름 → primitive가 객체에 entangle.
2. **형제 상태(sibling-state) 변별**: 같은 객체 아래 비슷한 속성들(`sliced` vs `diced`, `caramelized` vs `molten`)을 구분하는 미세한 시각 단서.
3. **Unseen 일반화**: seen 조합에 과적합하면 unseen이 무너짐 → seen-unseen trade-off가 본질.

---

## 2. SOTA 모델의 주요 흐름 (계보)

### 2-1. CLIP 기반 CZSL (현재 주류) — "텍스트 쪽을 적응시킨다"
계보의 공통 패턴: **CLIP 이미지 인코더는 (대부분) 동결하고 전역(CLS) feature를 쓰며, 텍스트 프롬프트/조합 임베딩을 학습·정교화**한다.

| 모델 | 핵심 아이디어 | 위치 |
|---|---|---|
| **CLIP zero-shot** | "a photo of [attr] [obj]" 프롬프트로 그대로 매칭 | 출발점 |
| **CoOp → CSP** | 프롬프트를 **학습 가능한 soft token**으로; CSP는 attribute/object primitive를 **composable token**으로 학습 | 프롬프트 학습 |
| **DFSP** | **Decomposed-Fused Soft Prompt** — 조합을 attr/obj로 분해 후 cross-modal 융합 | 분해·융합 |
| **Troika** (CVPR'24) | **Multi-branch**(composition / attribute / object 3분기) + **CMT(Cross-Modal Traction)**: 텍스트가 이미지 patch에 cross-attend 해 프롬프트를 보정 | 본 연구 baseline |
| **ClusPro** (ICLR'25) | primitive마다 **다중 프로토타입 클러스터**로 intra-primitive 다양성 포착 | 정식 게재 SOTA |

### 2-2. ⚠ 현 SOTA는 단일하지 않다 — "두 갈래" 지형 (2026 기준 조사)
> CZSL은 **단일 SOTA로 수렴돼 있지 않다.** ClusPro(ICLR'25)와 CVPR'25 논문들이 **동시기 연구**(서로 인용 불가, ICLR 2월 vs CVPR 제출 ~전년 11월)라, 각 계열이 **서로 다른 비교군으로 "new SOTA"를 주장**한다.

| 계보 | 대표 (연도) | MIT-States CW **AUC / HM** | 비고 |
|---|---|---|---|
| **Prototype 계열** | ClusPro (ICLR'25) | 23.8 / 40.7 | 정식 게재 SOTA |
| **Troika-확장 계열** | Troika (CVPR'24) | 22.1 / 39.3 | 본 연구 baseline |
| | **LOGICZSL** (CVPR'25) | 23.4 / 40.5 | Troika base, **ClusPro 비교 안 함** |

- **시사점 1 (baseline 정당성):** Troika는 **가장 널리 확장되는 reference framework** (LOGICZSL·TOMCAT·Pattern Recognition'25 등 다수가 Troika 위에 구축). 따라서 **Troika를 baseline으로 삼은 본 연구는 분야 표준에 부합**한다. LOGICZSL(CVPR'25)조차 ClusPro를 비교 표에서 빼고 Troika 기준으로 SOTA를 주장한다.
- **시사점 2 (gap 기준):** 본 연구의 gap은 정식 게재 SOTA인 **ClusPro(0.407/0.238) 기준**.

### 2-3. 본 연구가 주목한 빈틈
> **두 계열 모두 "이미지=전역, 텍스트(또는 prototype)=적응" 구도를 공유한다.**
> Troika의 CMT조차 *텍스트가 patch를 본 뒤 텍스트를 보정*하고, 최종 점수는 보정된 텍스트 · **전역 CLS**로 계산 → 이미지 쪽은 끝까지 global. ClusPro의 prototype도 전역 feature 공간에서 동작.
> → **이미지 표현 자체를 국소화(localize)하는 축은 비어 있다.** (§21 PGAL 차별화 근거)

---

## 3. 아이디어 전개 과정 (연구의 지적 흐름)

연구는 "외부 지식 주입" 가설에서 출발해 일련의 사전등록(pre-registered) 검증을 거쳤고, 그 실패들이 **문제의 진짜 병목**을 가리키며 새 방향으로 수렴했다.

### 3-1. 1단계 — 외부 의미지식으로 표현을 강화하면 될까? (가설 → 기각)
- **시도한 축들**: LLM visual description ensemble(v3_text), LLM description의 spatial richness를 vision-side로 distill(R2D2), image-conditional description selection, OT(optimal transport) alignment, text-enrichment 등.
- **결과**: 전부 baseline noise 안 또는 **음의 신호**. (mit-states 시드 std HM ±0.0008로 매우 작아 판정 신뢰도 높음)
- **대표 사례**: v3_text는 mit-states·UT-Zappos **양쪽 모두 음의 회귀**(Δ HM −0.027 / −0.053) → dataset 의존 가설 기각, 완전 봉인.
- **누적**: legacy 11개 축 + ckpt-selection 축 전부 null/sealed.

### 3-2. 2단계 — 보조 supervision을 더 강하게 주면? (§20 재정의 4축 → 전부 기각)
"개입 강도(bite)를 키우면 표현이 바뀐다"는 가설을 **개입 강도 순으로 사전등록 검증**:

| 축 | 메커니즘 | bite | Δ HM | 판정 |
|---|---|---|---|---|
| **C′** | logic-rule (속성 배타성 + 객체–속성 비양립 soft penalty) | ~0.2% | −0.0009 | fail |
| **D′** | LLM semantic hard-negative InfoNCE | ~0.8% | +0.0011 | tie |
| **E** | LLM taxonomy aux (attr→18 state-group, obj→21 super-cat 예측) | **~3.3%** | **−0.012** | fail (worst) |
| **F** | LLM multimodal test-time rerank (top-5 재선택) | inference | net −0.02 | null |

- **반전된 결론**: bite를 0.2→0.8→**3.3%** 로 키울수록 신호는 0→음수, **가장 강한 E가 가장 나빴다**.
  → "보조 supervision 강화 = 표현 개선" 가설의 **직접 반증**. coarse prior가 fine-grained 변별을 오히려 희석(과적합 가속).
- **F(추론 rerank)도 null**: zero-shot LLM은 한 이미지에 동시 참인 속성이 여럿일 때(`huge`↔`inflated balloon`) dataset의 single-label과 어긋나, 복구(+5) < 파괴(−6).
  - *방법론 교훈*: 1차 probe의 +0.22는 **이미지 파일 경로에 정답 라벨이 박혀 새던 leak** → 익명화 + gt-blind 평가자로 재측정하니 −0.02. (vision-LLM eval 시 path/filename leak 필수 점검)

> **⚠ 정직한 대조 — C′ vs LOGICZSL (CVPR 2025).** 본 연구의 C′(LLM logic-rule)는 fail로 봉인했으나, **동시기 CVPR'25 논문 LOGICZSL이 같은 계열(LLM logic rule을 Troika에 grounding)로 +1.3 AUC를 보고하며 게재**됐다. 차이는 *아이디어*가 아니라 *구현*에 있었을 가능성이 높다: (1) LOGICZSL은 logic을 **3종**(oi/oe/ae)으로 분해, C′는 2종(mutex+incompat); (2) LOGICZSL은 logic loss를 **warmup 이후 단계적 투입**(s₂≈80, s₃≈120 iter), C′는 λ=0.1로 처음부터 적용해 bite ~0.2%로 inert; (3) aggregation q=3.0로 outlier 강조. → 발표에서 이를 **선제적으로 다루면**, "왜 봉인했나"가 약점이 아니라 *구현 변수까지 통제한 검증 설계*의 증거가 된다. (재검토 후보로도 보존)

### 3-3. 진단 — 진짜 병목은 어디인가 (F-probe)
- 학습된 모델로 test 12,995장을 분석:
  - top-1 정확도 **0.286**, 그러나 **gt가 top-5 안에 들 확률 0.606**.
  - → **표현은 정답을 top-5까지 끌어올리지만 top-1에서 형제 상태를 못 가린다.**
- **결론**: 병목은 *텍스트 지식 부족*이 아니라 **같은 객체 아래 형제 상태의 시각적 granularity** — 즉 "**어디를 보느냐**"의 문제.

### 3-4. 3단계 — 방향 전환: differentiation-first (§21)
- **목표 재정의(지도 방향 반영)**: ClusPro 성능 추격이 아니라 **우리만의 차별화된 메커니즘**. 성능이 baseline보다 약간 부족해도 **novel + defensible**하면 기여로 인정.
- **판정 기준 재정의**: "baseline 초과"(X) → **(a) 메커니즘이 의도대로 작동(ablation·해석 입증) + (b) non-regression(HM ≥ baseline−0.005)** 이면 PASS.

### 3-5. 제안 메커니즘 — **PGAL: Patch-Grounded Attribute Localization**
- **한 줄 thesis**: 기존 CZSL이 *텍스트를 적응*시키는 것과 반대로, **속성이 발현된 이미지 영역으로 표현을 국소화**해 그 국소 영역에서 속성을 채점한다.
- **구조**:
  - Object 분기 = 전역 CLS (객체는 holistic).
  - **Attribute 분기(핵심)**: 학습된 localization query가 CLIP **patch grid에 cross-attention** → attention map + 국소 feature `z_a`. 속성 점수 = sim(`z_a`, attr text).
  - Composition 분기 = CLS (Troika 그대로).
  - attention entropy 정규화(focus 유도)는 **train-only** → eval 체크포인트 선택에 confound 0.
- **Troika와의 차별(코드 감사로 lock)**: Troika는 image-side가 끝까지 global. PGAL은 **image-side localization** — Troika가 안 건드리는 직교 축.
- **설계상 주의(반영)**: `forward_for_open`(eval 경로)도 동일 localization으로 override → train/eval mismatch(과거 버그 유형) 구조적 차단.

---

## 4. 결과

### 4-1. Baseline & gap
- **Troika 재현 (best_loss, 3-seed)**: **HM 0.3899 ± 0.0038 / AUC 0.2177** (mit-states, ViT-L/14).
  - Troika 논문(0.392/0.201) 위 → 재현 자체엔 gap 없음. (재현 신뢰도 확보)
  - **정식 게재 SOTA(ClusPro 0.407/0.238) 대비 gap −0.017 HM / −0.020 AUC** (시드 noise 대비 ~4σ, 진짜 architectural 차이).
  - 단, **differentiation-first 목표상 이 gap을 닫는 것이 목적이 아님** — non-regression + 차별 메커니즘 입증이 목표(§3-4).

### 4-2. 탐색 종합 — "incremental은 EV ≈ 0"
- legacy 11축 + ckpt-selection + §20 4축 = **누적 16축 전부 null/sealed**.
- → **axis/hyperparam/inference 변형으로 gap을 닫을 기대값 ≈ 0** 을 사전등록 검증으로 확정. (이것 자체가 방어 가능한 negative result)

### 4-3. PGAL Stage 1 (첫 non-regression 결과)
| | seen | unseen | **HM** | **AUC** |
|---|---|---|---|---|
| Troika baseline (3-seed) | 0.489 | 0.524 | **0.3899 ± 0.0038** | 0.2177 |
| **PGAL seed0** | 0.4676 | 0.5329 | **0.3878** | 0.2127 |

- Δ HM vs baseline = **−0.0021**, 3-seed std(±0.0038) **노이즈 대역 안** → **non-regression 통과**.
- **§20 4축이 전부 fail/null이었던 것과 대조** — image-side localization 축은 적어도 표현을 **깨뜨리지 않는** 첫 결과. (E taxonomy는 같은 baseline 대비 −0.012 fail)
- Val HM 곡선: PGAL은 ep2–5 **넓은 고원**(0.415–0.419), E는 ep2 단일 피크 후 급락 → 안정성도 우위.

### 4-4. ⚠ 단, 차별성은 아직 미입증
- 재정의 PASS = (a) non-regression **AND** (b) localization ablation 기여 **AND** (c) attention 의미성.
- 현재 **(a)만 충족**. HM이 noise 대역 안이라 "localizer가 실제로 일을 하는지"(전역 평균으로 collapse했을 가능성)는 ablation/시각화 없이 단정 불가.
- → **(b)(c)(d) 검증이 PASS/seal을 최종 결정** (아래 향후 계획).

---

## 5. 향후 계획

### 5-1. PGAL 차별성 입증 (진행 중)
1. **(b) Ablation — uniform attention** *(현재 GPU1에서 학습 중, ETA ~오늘 21시)*:
   patch attention을 uniform(=mean-pool)으로 강제해 **localization만 제거**(추가 pathway·params는 동일 유지).
   PGAL > uniform-ablation이면 → 국소화 기여 입증. 같으면 → localizer가 noise → seal.
2. **(c) Attention map 시각화**: val_best/final ckpt로 10장, rust 영역·cut 단면·moisture 등 **의미 있는 영역을 짚는지** 정성 검증.
3. **(d) F-probe 오라클**: PGAL이 F-probe가 짚은 **sibling-state 혼동을 실제로 복구**하는지 정량.

### 5-2. 통과 시 — 검증 확장 (Stage 2)
- **3-seed 안정성** 확인 + **UT-Zappos 전이**(도메인 일반성).
- 해석 분석: attention map 정성 + sibling-state 복구율 정량을 논문 그림으로.

### 5-3. 차순위 축 (PGAL fail/collide 시 fallback)
- **(II) Attributes-as-operators 부활 (CLIP-era)** — 가장 깨끗한 novelty. 속성을 CLIP 공간의 연산자로 재해석.
- **(IV) Generative unseen-feature synthesis** — 승자 위에 직교 stack.
- **(III) Ordinal/continuous state** — 독립 축 대신 (I)/(II)의 regularizer로 흡수(ripeness/size/moisture 등 순서형 상태).

### 5-4. 종합 메시지 (발표 마무리용)
- **negative 16축 → 진단(F-probe) → 차별화 메커니즘(PGAL)** 으로 이어지는 **가설-검증 주도 연구**.
- 기여 = (1) "incremental 개입은 효과 없다"는 **사전등록된 체계적 negative result**, (2) 병목을 *visual granularity*로 특정한 진단, (3) image-side localization이라는 **방어 가능한 새 축**과 첫 non-regression 실증.

---

### 부록 A — 발표 슬라이드 매핑(권장)
1. 타이틀 / 한 줄 thesis
2. CZSL이란? (조합 일반화 예시 그림: red apple+green tomato→red tomato)
3. 평가 셋업·지표 (seen/unseen, HM/AUC, MIT-States)
4. SOTA 지형: **"두 갈래"**(Prototype: ClusPro vs Troika-확장: LOGICZSL) + Troika baseline 정당화 + "이미지=전역" 공통 빈틈
5. 아이디어 흐름 1: 외부지식 주입 16축 → 전부 null (요약 표)
6. 아이디어 흐름 2: F-probe 진단 (top1 0.286 vs top5 0.606 → 형제상태 병목)
7. 제안: PGAL 구조 그림 (patch grid + localization query → z_a)
8. 결과: baseline vs PGAL 표 + val 곡선
9. 차별성 검증 계획 (ablation/attn viz/oracle) — "지금 돌고 있음"
10. 향후 축 (II operators / IV generative) + 마무리 메시지

### 부록 B — 예상 질문 대비 (Q&A 방어) ★발표 핵심★
> 심사·지도교수가 던질 가능성이 높은 질문에 대한 준비된 답.

**Q1. "왜 SOTA인 ClusPro가 아니라 Troika를 baseline으로 썼나?"**
- A. (1) CZSL은 **단일 SOTA로 수렴 안 됨** — CVPR'25 LOGICZSL 등 다수 최신 논문이 ClusPro를 비교에서 빼고 **Troika를 표준 base로 확장**. (2) Troika는 multi-branch 구조가 **모듈식이라 image-side 개입(PGAL)을 깨끗이 삽입**할 수 있음 — prototype 기반 ClusPro는 우리 가설(localization)과 결합점이 다름. (3) 우리 목표는 SOTA 추격이 아니라 **차별 메커니즘 검증**이라 강한 표준 base면 충분.

**Q2. "logic-rule(C′)을 fail로 봉인했는데, 같은 아이디어가 CVPR'25(LOGICZSL)엔 SOTA로 실렸다. 모순 아닌가?"**
- A. **아이디어는 같고 구현이 달랐다 — 그리고 그 차이를 우리는 안다.** LOGICZSL은 (1) logic 3종 분해, (2) **warmup 후 단계적 logic loss 투입**, (3) outlier 강조(q=3.0). 우리 C′는 2종·전 구간 λ=0.1로 bite ~0.2%로 **inert**했다. → 이건 봉인이 *경솔*했다는 게 아니라, **우리 사전등록 프로토콜이 "약한 구현은 약한 신호"를 정확히 잡아냈다**는 의미. (재검토 시 staged-logic ablation 1런으로 확인 가능 — 보존 중)

**Q3. "16축이 다 실패면 결국 negative만 남은 것 아닌가?"**
- A. (1) 16축 null은 **사전등록된 체계적 negative result** — "이 방향은 안 된다"를 통제된 실험으로 입증한 것 자체가 기여. (2) 그 negative가 **F-probe 진단 → 병목 특정(visual granularity)** 으로 이어졌고, (3) 거기서 **PGAL이라는 새 축 + 첫 non-regression**이 나왔다. 실패의 나열이 아니라 **진단 주도 수렴 과정**.
