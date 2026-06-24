# 최종 발표 — 슬라이드별 제작 가이드
### "Image-side Localization을 통한 차별화된 CZSL 메커니즘 탐색"

> 본문 상세는 `final_presentation.md` 참조. 본 문서는 **PPT 제작용 슬라이드별 스펙**(제목 / 본문 불릿 / 시각자료 / 말할 포인트).
> 수치 표기: 본 연구 실험은 소수(0.3878), 논문 SOTA도 동일 스케일 소수로 통일(ClusPro 0.407 = 40.7%).
> 총 14장(본편) + 백업 3장. 발표 ~15분 기준.
>
> **[가독성 패스 — 2026-06-02]** CZSL을 모르는 청중(타 분야 교수·학생)도 따라올 수 있도록 용어 풀이·프레이밍을 보강함. 각 슬라이드의 `→ 외부인 주의` 항목 참조. PDF 슬라이드 번호 매핑: md S1–S7 = PDF 1–7 / S8-1·8-2 = PDF 8·9 / S9 = PDF10 / S10 = PDF11 / S11 = PDF12 / S12 = PDF13 / S13 = PDF14 / S14 = PDF15.

---

## [슬라이드 1] 타이틀
- **제목**: Patch-Grounded Attribute Localization — 이미지 국소화 기반 CZSL 메커니즘 탐색
- **부제 한 줄**: "기존 CZSL이 텍스트를 적응시킬 때, 우리는 이미지를 국소화한다"
- 발표자 / 소속 / 날짜
- *말할 포인트*: 오늘 발표는 ① CZSL이 뭔지 → ② 현 SOTA 흐름 → ③ 내 아이디어 전개 → ④ 결과 → ⑤ 향후 순서.
- *말할 포인트(프레이밍 — 맨 처음에 깔 것)*: **이 발표는 "SOTA를 이겼다"가 아니라, "왜 안 되는지 체계적으로 진단하고 → 아무도 안 건드린 축(이미지 국소화)을 제안한다"는 진단-주도 연구다.** 이 한 줄을 먼저 선언해야 뒤의 "16축 전부 실패"가 변명이 아니라 통제 실험(=기여)으로 읽힌다.
- *→ 외부인 주의*: 부제의 "텍스트를 적응시킨다 / 이미지를 국소화한다"는 아직 정의 전. 한 박자 뒤(슬라이드 5·7)에서 풀린다고 미리 안내.

## [슬라이드 2] CZSL이란? (문제 정의)
- **제목**: Compositional Zero-Shot Learning
- 본문:
  - 속성(attribute) × 객체(object) **조합**을 인식
  - **학습 때 못 본 조합**까지 맞히는 것이 목표
  - primitive(속성·객체)는 봤지만 그 *조합*은 처음
- **시각자료(핵심)**: 그림 — 학습: 🍎`red apple`, 🍅`green tomato` → 테스트: `red tomato`(처음 보는 조합) ✅
- *말할 포인트*: "빨강도 알고 토마토도 아는데, '빨간 토마토'를 본 적 없을 때 맞힐 수 있나?"가 핵심.

## [슬라이드 3] 평가 셋업 & 지표
- **제목**: 어떻게 평가하나
- 본문(2단):
  - **데이터**: MIT-States(115 attr × 245 obj, 자연이미지) / UT-Zappos(신발, fine-grained)
  - **Split**: Seen pairs(학습) vs Unseen pairs(테스트 전용) / Closed-world(실재하는 조합만 후보) vs Open-world(모든 attr×obj가 후보 — 훨씬 어려움)
  - **지표**(모두 **높을수록 좋음**): Seen acc · Unseen acc · **HM**(둘의 조화평균=균형) · **AUC**(핵심)
- **시각자료**: seen-unseen accuracy 곡선 + AUC 면적 도식
- *말할 포인트*: seen↔unseen은 trade-off라 HM·AUC로 균형을 본다.
- *→ 외부인 주의*: **AUC** = "seen을 얼마나 선호할지(bias)를 0→최대로 쓸어가며(bias-sweep) 그린 seen–unseen 곡선의 아래 면적". 한 점이 아니라 **전 구간 성능**을 한 숫자로 본다고 풀어주기.

## [슬라이드 4] 무엇이 어려운가
- **제목**: CZSL의 3대 난점
- 본문:
  1. **맥락 의존성**: 같은 `wet`도 `wet dog`/`wet road`/`wet paint`가 시각적으로 전혀 다름
  2. **형제 상태(sibling-state) 변별**: `sliced` vs `diced`, `caramelized` vs `molten` 같은 미세 차이
  3. **Unseen 일반화**: seen 과적합 시 unseen 붕괴
- **시각자료**: 같은 속성이 객체마다 다르게 보이는 예시 이미지 3장
- *말할 포인트*: 특히 ②형제 상태가 뒤에서 우리 병목 진단의 핵심이 됨(복선).

## [슬라이드 5] SOTA 흐름 — CLIP 기반 계보
- **제목**: 현재 주류 — "텍스트 쪽을 적응시킨다"
- **시각자료(표)**:
  | 모델 | 핵심 | 
  |---|---|
  | CLIP zero-shot | 프롬프트 그대로 매칭 |
  | CoOp→CSP | soft/composable token 학습 |
  | DFSP | 분해-융합 soft prompt |
  | **Troika** (CVPR'24) | multi-branch + CMT=Cross-Modal Transformer (텍스트가 이미지 patch에 attend) |
  | **ClusPro** (ICLR'25) | primitive 다중 프로토타입 |
- *말할 포인트*: 공통점 — **이미지는 전역(CLS), 텍스트/프로토타입만 적응**. 이게 다음 장 빈틈의 근거.
- *→ 외부인 주의*: **CLS** = ViT가 이미지 한 장을 통째로 요약한 토큰(=위치 정보 없는 전역 표현). 약어(CMT·CLS)는 표 아래 한 줄 각주로 박아두기.

## [슬라이드 6] SOTA는 단일하지 않다 (+ baseline 정당화)
- **제목**: 두 계열로 갈린 SOTA — 그리고 왜 Troika인가
- **시각자료(표, 소수 통일 — AUC·HM 모두 높을수록 좋음)**:
  | 계보 | 대표 | MIT-States CW (AUC/HM) |
  |---|---|---|
  | Prototype | ClusPro (ICLR'25) | 0.238 / 0.407 |
  | Troika-확장 | Troika (CVPR'24) | 0.221 / 0.393 |
  | | LOGICZSL (CVPR'25) | 0.234 / 0.405 |
- 본문:
  - 동시기 연구(ICLR 2월 vs CVPR 제출 전년 11월) → **서로 비교 안 하고 각자 "SOTA" 주장** (=단일 1등이 없음)
  - **LOGICZSL(CVPR'25)조차 ClusPro 빼고 Troika 기준** → Troika는 가장 널리 확장되는 표준 base
  - → **Troika를 baseline으로 삼은 건 분야 표준에 부합** (성능 1등 ClusPro가 아니라 *확장 표준* Troika를 고른 이유)
- *말할 포인트*: "왜 (더 높은) ClusPro 안 썼냐"는 질문의 선제 답변 — 우리 목표는 성능 1등이 아니라 깨끗하게 개입할 수 있는 표준 base.
- *→ 외부인 주의*: 숫자가 0.40대라 "낮다"고 오해하기 쉬움 — CZSL은 unseen 조합 과제라 이 대역이 정상 SOTA임을 한마디 덧붙이기.

## [슬라이드 7] 우리가 주목한 빈틈
- **제목**: 비어 있는 축 — Image-side Localization
- 본문(큰 글씨):
  - 두 계열 모두 **이미지=전역, 적응은 텍스트/프로토타입 쪽**
  - Troika CMT도 결국 보정된 텍스트 · **전역 CLS**로 채점
  - → **"이미지 표현 자체를 국소화"하는 축은 아무도 안 건드림**
- **시각자료**: 기존(텍스트 적응 화살표) vs 우리(이미지 국소화 화살표) 대비 도식
- *말할 포인트*: 이게 PGAL의 한 줄 차별점.

## [슬라이드 8 (선택 도입)] 16축 — 5개 범주 한눈에 (8-1/8-2 안 쓸 때 대체용)
> 시간 부족 시: 이 범주표 1장으로 대체. 상세가 필요하면 8-1/8-2 사용.
- **제목**: 1단계 — 가능한 모든 각도로 gap을 공략 → 전부 기각
- **시각자료(범주 요약 — 큰 그림)**: 5개 개입 범주 × 16변형이 모두 null/sealed임을 한눈에
  | # | 개입 범주 | 축 (개수) | 결과 |
  |---|---|---|---|
  | A | 텍스트/의미지식 주입 | v3_text · image_cond · feasibility · C′ · D′ · E · F (7) | 전부 tie/null/fail |
  | B | Vision-side 표현 | R2D2 · imgsel · vision_distill (3) | 전부 tie/null |
  | C | Loss 기하 | cosface · hardpair (2) | 전부 null |
  | D | Prototype assignment | OT-LHP · OT-Troika (2) | 양쪽 dead |
  | E | 체크포인트 운영 | ckpt-ensemble · ckpt-selection (2) | tie/null |
- 본문(핵심 한 줄): **누적 16축 전부 null/sealed** + 개입 강도(bite)를 0.2%→3.3%로 키워도 신호 0~음수(최강 E가 −0.012로 최악)
- *말할 포인트*: "보조지식·구조·loss·운영 어느 각도로도 안 닫힌다 → gap은 architectural." 실패 나열이 아니라 **사전등록된 체계적 negative result.**
- ※ 16축 개별 설명은 [슬라이드 8b]로 분리(밀도). 발표 시 8은 범주표만, 8b는 백업/부록 또는 질문 대비.

## [슬라이드 8-1] 아이디어 전개 ① — 16축 전부 효과 0~음수 (1/2): 표현 강화 계열
- **제목**: 1단계 — "지식·표현을 강화하면?" → 10개 변형, 전부 기각
- **상단 roadmap(작게)**: 5개 개입 범주 = A 의미지식 / B vision / C loss / D prototype / E 운영 → 이번 장은 **A·B**
- **읽는 법(표 위에 큰 글씨로 — 외부인 필수)**: 아래는 *baseline에 한 가지씩 더해본 시도*들. **`Δ HM` = baseline 대비 성능 변화량** → **양수=좋아짐 / 음수=나빠짐 / 0 근처=차이 없음**. 즉 이 표는 전부 "더했더니 그대로거나 오히려 나빠졌다".
- *각주(중요)*: Δ HM은 **해당 실험의 framework baseline 기준** (LHP-CZSL 0.3887 / Troika best_loss 0.3899·single-seed 0.3940). 시드 std ±0.0008(LHP)~±0.0038(Troika) → **±0.0038보다 작은 변화는 사실상 노이즈**. **모두 실행 완료한 실험** — "Δ"가 곧 효과크기.
- *→ 외부인 주의*: `v3_text·C′·D′·E·F` 같은 코드네임은 우리 내부 라벨일 뿐. 청중에겐 **메커니즘 한 줄 설명**(가운데 칸)만 읽게 하고, 코드네임은 "그냥 실험 번호"로 넘기기. `tie`=무효, `fail`=하락, `봉인/sealed`=폐기 확정.

**A. 텍스트 / LLM 의미지식 주입 (7)**
1. **v3_text** — LLM 시각 description ensemble을 텍스트 prompt에 결합 → mit-states·UT-Zap 양쪽 음의 회귀(Δ −0.027/−0.053), sealed
2. **image_cond** — sub-meaning을 image-conditional softmax로 aggregation(mean-pool 대체) → HM 0.3597, **Δ −0.029**, fail
3. **feasibility** — LLM(Gemini) (attr×obj) feasibility 0–10 점수를 comp logit에 prior bias 주입(open-world masking) → 신호 없음, 봉인
4. **C′** — logic-rule: attr 상호배타 + 객체–속성 비양립 soft penalty(λ=0.1, bite ~0.2%) → Δ HM −0.0009, fail
5. **D′** — LLM semantic hard-negative InfoNCE(λ=0.1/τ=0.1, bite ~0.8%) → Δ HM +0.0011, tie
6. **E** — LLM taxonomy aux CE(attr→18 state-group, obj→21 super-cat, bite ~3.3% 최강) → Δ HM −0.012, fail(worst)
7. **F** — LLM multimodal test-time rerank(top-5 재선택, inference-only) → net −0.02, null

**B. Vision-side 표현 (3)**
8. **R2D2** — LLM description의 spatial richness를 vision-side로 distill(CLIP patch grid 활용) → HM 0.3926, **Δ +0.0039**, tie(best_hm로 AUC +0.0094였으나 HM cutoff 미달)
9. **imgsel (v4_imgsel)** — image-conditional 시각 feature 선택(v3_text "mean-pool collapse" 약점 검증) → HM 0.3354, **Δ −0.059**(v3_text 대비 +0.004 noise)
10. **vision_distill** — R2D2 distill-only 변형(routing 제거) → HM 0.3835, **Δ −0.015**

- *말할 포인트(8-1)*: "외부 지식을 텍스트로든 vision으로든 부어도 안 된다." (다음 장: 구조·loss·운영도 동일)

## [슬라이드 8-2] 아이디어 전개 ① — 16축 전부 효과 0~음수 (2/2): 구조·운영 계열 + 결론
- **제목**: 1단계(계속) — 구조·loss·assignment·운영도 전부 기각

**C. Loss 기하 (2)**
11. **cosface (Axis G)** — comp/attr/obj head에 CosFace additive cosine margin(seen boundary tightening) → HM 0.3763, **Δ −0.0124**, fail (val +0.020이나 test 일반화 실패)
12. **hardpair** — same-primitive hard pair sampling + hard-pair loss(weight 0.1/temp 0.1) → HM 0.383, **Δ −0.0049**, tie(음의 가장자리)

**D. Prototype assignment (2)**
13. **OT-LHP** — Sinkhorn 기반 prototype assignment(ClusPro §4.4의 +3.3 AUC 메커니즘) on LHP framework → Δ −0.0012, dead
14. **OT-Troika** — 동일 OT를 Troika에 이식 → Δ −0.0073, dead (양쪽 framework 모두 죽음 = OT는 train/val noise fit)

**E. 체크포인트 운영 (2)**
15. **ckpt-ensemble** — multi-epoch 체크포인트 앙상블(H2) → HM 0.3924, **Δ +0.0037**(6/9-ckpt 확장은 오히려 하락)
16. **ckpt-selection** — val_metric best_loss vs best_hm → best_hm로 AUC ~10σ 회수했으나 HM cutoff 0.0011 미달 + 메커니즘 아님 → sealed

- **결론 박스(강조)**:
  - 5개 각도 × 16변형 = **전부 실행 → 효과 0~음수(전부 봉인)**. 최고치도 ckpt-ensemble Δ +0.0037 / R2D2 Δ +0.0039로 cutoff(+0.005) 미달
  - 개입 강도(bite) 0.2%→3.3%로 키워도 신호 0~음수 — **최강 E가 −0.012로 최악**
  - → **gap은 architectural** (보조손실·구조·운영으로 못 닫음) = 사전등록된 **체계적 negative result**
- *말할 포인트*: 실패 나열이 아니라 "다 해봤고 안 됐다"를 통제 실험으로 입증 → PGAL pivot의 정당성.
- *→ 외부인 주의*: 슬라이드(PDF 9) 결론에 쓰인 **"bite↑에도 음수"의 `bite`는 "개입 강도"**(모델을 얼마나 세게 건드렸나, 0.2%→3.3%)라는 뜻 — 반드시 "개입 강도(bite)"로 풀어쓰기. 또 `OT`=Optimal Transport(프로토타입 배정 기법), `dead`=양쪽 프레임워크에서 모두 효과 없음.

## [슬라이드 9] 아이디어 전개 ② — 진짜 병목 진단 (F-probe)
- **제목**: 2단계 — 병목은 "지식"이 아니라 "어디를 보느냐"
- **시각자료(핵심 숫자, 크게)**:
  - top-1 정확도 **0.286** ↔ gt가 top-5 안: **0.606**
- 본문:
  - 표현은 정답을 top-5까지 올림 → 그러나 **top-1에서 형제 상태를 못 가림**
  - 실패 = sibling-state 혼동 (`caramelized`↔`molten` 등)
  - → 병목 = **같은 객체 아래 형제 상태의 시각적 granularity**
- *말할 포인트*: 16축의 실패가 여기로 수렴 — "텍스트 지식이 아니라 시각 위치가 문제."

## [슬라이드 10] 제안 — PGAL 메커니즘
- **제목**: PGAL: Patch-Grounded Attribute Localization
- **시각자료(구조도, 핵심 1장)**:
  - CLIP patch grid → **localization query가 cross-attention** → attention map + 국소 feature `z_a`
  - Attribute 분기: sim(`z_a`, attr text) / Object·Comp 분기: 전역 CLS (Troika 유지)
- 본문:
  - 속성이 **발현된 영역으로 이미지 표현을 국소화**해 그 영역에서 채점
  - attention map = **해석 가능** (어디 보고 판단했는지)
  - entropy 정규화는 **train 단계에서만** 사용 → 평가 점수를 부풀리지 않음(공정)
  - 목표 재정의: SOTA 추격 X → **novel + defensible + non-regression**이면 기여
- *말할 포인트*: 텍스트 보정이 아니라 **이미지 표현의 위치 이동.**
- *→ 외부인 주의*: PDF 11의 "train-only → confound 0"은 **"평가 땐 안 쓰므로 점수 부풀림 없음"**으로 평이하게. `z_a`=속성을 보기 위한 *지역(local)* 이미지 표현(전역 CLS와 대비).

## [슬라이드 11] 결과 — Baseline & PGAL
- **제목**: 결과 — 첫 non-regression
- **시각자료(표)**:
  | | seen | unseen | HM | AUC |
  |---|---|---|---|---|
  | Troika baseline (3-seed) | 0.489 | 0.524 | **0.3899 ±0.0038** | 0.2177 |
  | PGAL seed0 | 0.468 | 0.533 | **0.3878** | 0.2127 |
- 본문:
  - **non-regression = "새 메커니즘을 넣어도 기존 성능을 (노이즈 이상으로) 깎지 않음"** — 앞의 16축은 대부분 이걸 못 하고 성능을 떨어뜨렸다.
  - Δ HM **−0.0021 = 시드만 바꿔도 생기는 노이즈(±0.0038) 안 → non-regression 통과** (=성능은 사실상 동률)
  - 앞서(슬라이드 8·9) 본 시도들이 전부 fail/null이었던 것과 대조 (E taxonomy −0.012 fail)
  - **솔직히: 아직 숫자를 *올린* 메커니즘은 없음** — 핵심은 "차별 축을 심으면서 표현을 안 깨뜨렸다"
- *말할 포인트*: 정직하게 — 성능 초과가 목표가 아니라 차별 메커니즘 검증이 목표(슬라이드 10 재정의).
- *→ 외부인 주의*: "성능 그대로인데 왜 성공?"이 자연스러운 반문. **한 화면에 "16축=성능 깎임 vs PGAL=안 깎임"을 나란히** 보여줘야 동률이 곧 진전임이 납득됨. `§20` 같은 내부 로그 번호는 "앞서 본 시도들"로 바꿔 말하기.

## [슬라이드 12] 차별성 검증 (메커니즘이 실제로 작동하나) — ✅ ablation 결과 반영 (2026-06-02 21:57)
- **제목**: PGAL이 *뭐라도* 하는가 — 검증 3종
- 본문:
  - *(라벨 안내 — 외부인 혼란 방지)*: **(a) = 슬라이드 12의 non-regression 결과 그 자체**(이미 제시). 이 장은 그 위에서 "그래서 메커니즘이 *실제로* 작동하나"를 묻는 (b)·(c)·(d) 검증. PDF에서 (a)가 빠져 보이지 않게, "검증 (a)→(d) 중 (b)(c)(d)" 식으로 연속 표기하거나 ①②③으로 바꿀 것.
  1. **(b) Ablation — ✅ 완료**: attention을 uniform(mean-pool)으로 강제해 *국소화만 제거*. **결과(MIT-States CW · test · seed0, 동일 조건)**:
     | | seen | unseen | HM | AUC |
     |---|---|---|---|---|
     | PGAL (localization) | 0.468 | 0.533 | **0.3878** | **0.2127** |
     | uniform (ablation) | 0.4676 | 0.5314 | **0.3831** | **0.2113** |
     → **PGAL > uniform, ΔHM +0.0047 / ΔAUC +0.0014** = 국소화가 점수에 기여(방향성 확인). **단 ΔHM은 1-seed 노이즈(±0.0038) 바로 바깥** → 약한 양(+)의 신호, 강한 주장은 3-seed로 굳힐 것.
  2. **(c) Attention map 시각화**: rust 영역·cut 단면 등 의미 영역 지목 확인
  3. **(d) F-probe 오라클**: sibling-state 혼동 실제 복구율
- **방향 확정 (ablation 결과 반영)**: PGAL > uniform 분기 → **"맵은 거칠어도(=(c) 시각화가 배경 지향이었던 것과 일관) 국소화가 점수엔 소폭 기여"**로 방어. (c) attention 시각화는 여전히 음성(맵 배경 지향, §21-1-4)이므로 **세게 주장하지 말고 "정량 ablation은 + 방향, 정성 맵은 아직 거칠다"로 정직하게**.
- **시각자료**: heatmap은 **현재 미사용**(참고용만 `slide_assets/slide12_attn_ep9/`). 정성 맵이 약하므로 ablation 정량 표를 메인으로.
- *말할 포인트*: 숫자가 노이즈 안이므로 "메커니즘이 작동한다"가 핵심 방어선 — **ablation이 그 판정자였고, 약하지만 + 방향으로 나왔다.**

## [슬라이드 13] 향후 계획
- **제목**: 향후 계획
- 본문:
  - PGAL 통과 시 → **Stage 2** (3-seed 안정성 + UT-Zappos 전이)
  - **C′ 재검토(진행)**: LOGICZSL식 staged-logic으로 봉인 축 재확인 (구현 차이 검증)
  - 차순위 축: **(II) Attributes-as-operators 부활** → (IV) generative synthesis → (III) ordinal은 regularizer로 흡수
- *말할 포인트*: 차별 축이 살면 해석·전이로 논문화, 죽으면 (II)로.

## [슬라이드 14] 결론 / 기여
- **제목**: 결론
- 본문(기여 3줄):
  1. **사전등록된 체계적 negative**: incremental 16축으로 "gap은 안 닫힌다(추가 효과 ≈ 0)"를 통제 실험으로 입증
  2. **병목 진단**: visual granularity(형제 상태)로 특정 (F-probe)
  3. **차별 축 제안**: image-side localization(PGAL) + 첫 non-regression + 해석 가능성
- **마무리 한 줄**: "실패의 나열이 아니라 **진단 주도 수렴** — negative → 진단 → 차별 메커니즘"
- *→ 외부인 주의*: PDF 15의 **`EV≈0`은 "Explained Variance≈0", 즉 "추가로 설명되는 효과가 거의 0"**이라는 뜻 — 약어 대신 풀어쓰기. `F-probe`=병목을 짚어낸 진단용 oracle 실험.

---

## [백업 B1] Q&A — Troika baseline
- "왜 SOTA ClusPro 아니라 Troika?" → 단일 SOTA 미수렴(LOGICZSL도 Troika 기준) + Troika의 모듈성으로 image-side 개입 깨끗 + 목표는 SOTA 추격 아님.

## [백업 B2] Q&A — C′ vs LOGICZSL
- "봉인한 logic-rule이 CVPR'25엔 SOTA?" → 아이디어 동일, **구현 차이**(3종 logic + staged 투입 + q=3.0). 우리 C′는 2종·전구간 λ=0.1로 inert. 사전등록 프로토콜이 약한 구현을 정확히 잡은 것. **staged 재검토 진행 중**(데이터로 확인).

## [백업 B3] Q&A — "negative만 남은 것 아닌가"
- 16축 null = 통제된 negative result(기여). 그 negative가 진단→PGAL→첫 non-regression으로 수렴. 진단 주도 과정.

## [백업 B4] 숫자 디테일 (질문 대비)
- 시드 std HM ±0.0038(3σ ±0.011), Troika 재현 0.3899(논문 0.392 위, 재현 gap 0).
- 유일한 양의 신호였던 R2D2+best_hm: AUC 0.2231(+0.006 ~10σ)이나 ckpt-선택 효과 + HM cutoff 미달로 봉인.
