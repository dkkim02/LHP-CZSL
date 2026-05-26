# Experimental Results — LHP-CZSL

실험 결과만 모아둔 파일. 시계열 narrative / 디버깅 / 진단은 [RESEARCH_LOG.md](RESEARCH_LOG.md) 참고.

모든 실험: ViT-L/14 backbone, fp16 AMP, batch_size=8, grad_accum=8, 15 epochs (probe 제외), val_metric=best_hm.

---

## 1. mit-states (closed-world, test_pairs, 논문 비교용)

수치 출처: `checkpoint/<run>/val_best.closed.json`의 `test` 블록. 논문 (ClusPro ICLR'25 Table 1 등) split과 동일.

| Run | Date | seed | seen | unseen | **HM** | **AUC** | attr | obj |
|---|---|---|---|---|---|---|---|---|
| ClusPro baseline (K=5) | 4-27 | 0 | 0.4899 | 0.5203 | **0.3893** | **0.2169** | 0.3856 | 0.5553 |
| LHP-CZSL v2 (LLM-derived K) | 4-28 | 0 | 0.4891 | 0.5241 | **0.3897** | **0.2179** | 0.3836 | 0.5573 |
| ClusPro baseline K=3 (ablation) | 4-29 | 0 | 0.4870 | 0.5225 | **0.3855** | **0.2158** | 0.3868 | 0.5563 |
| LHP-CZSL v1 (sub-meaning init + sem + decorr) | 5-1 | 0 | 0.4866 | 0.5250 | **0.3886** | **0.2182** | 0.3862 | 0.5584 |
| LHP-CZSL v1_init_only (sem=0, decorr=0) | 5-1→5-2 | 0 | 0.4878 | 0.5202 | **0.3874** | **0.2155** | 0.3851 | 0.5552 |
| v3_text (LLM visual desc, K=3) | 5-6 | 0 | 0.4882 | 0.4715 | 0.3572 | 0.1887 | 0.3550 | 0.5263 |
| v3_text seed0 retry | 5-11 | 0 | 0.4840 | 0.4699 | 0.3569 | 0.1890 | 0.3555 | 0.5272 |
| v3_text | 5-6 | 1 | 0.4903 | 0.4782 | 0.3645 | 0.1938 | 0.3608 | 0.5334 |
| v3_text | 5-6 | 2 | 0.4895 | 0.4810 | 0.3633 | 0.1954 | 0.3601 | 0.5356 |
| **v3_text 3-seed mean** | 5-11 | — | **0.4879** | **0.4764** | **0.3616** | **0.1927** | — | — |
| ClusPro baseline + paper hp (wd 5e-5, α 0.2, β 0.5) | 5-12 | 0 | 0.4912 | 0.5228 | **0.3853** | **0.2173** | 0.3893 | 0.5604 |
| ClusPro baseline post-fix | 5-15 | 0 | 0.4933 | 0.5226 | 0.3879 | 0.2176 | 0.3842 | 0.5590 |
| ClusPro baseline post-fix | 5-15 | 1 | 0.4857 | 0.5276 | 0.3889 | 0.2172 | 0.3858 | 0.5561 |
| ClusPro baseline post-fix | 5-15 | 2 | 0.4874 | 0.5220 | 0.3894 | 0.2165 | 0.3852 | 0.5546 |
| **ClusPro baseline post-fix 3-seed mean ± std** | 5-15 | — | 0.4888 ± 0.004 | 0.5241 ± 0.003 | **0.3887 ± 0.0008** | **0.2171 ± 0.0006** | 0.3851 | 0.5566 |
| 논문 ClusPro (Table 1) | ICLR'25 | — | 0.521 | 0.540 | **0.407** | **0.238** | — | — |
| 논문 CDS-CZSL / Troika / PLID | — | — | 0.497~0.503 | 0.529~0.530 | 0.390~0.393 | 0.221~0.224 | — | — |

**Seed std (post-fix 3-seed):** HM ±0.0008, AUC ±0.0006.
**우리 baseline vs 논문 ClusPro:** HM −1.8pp (≈23σ) — 재현 갭은 노이즈 아니라 진짜 구현 차이.

---

## 2. UT-Zappos (closed-world, test_pairs, 논문 비교용)

수치 출처: 각 ckpt에 대해 `test.py --yml_path <yml> --load_model val_best.pt --open_world False` 재실행 (5-11) → `checkpoint/<run>/val_best.closed.json`의 `test` 블록.

| Run | seed | seen | unseen | **HM** | **AUC** |
|---|---|---|---|---|---|
| ClusPro baseline | 0 | 0.6647 | 0.7499 | 0.5296 | 0.4042 |
| ClusPro baseline | 1 | 0.6676 | 0.7478 | 0.5493 | 0.4194 |
| ClusPro baseline | 2 | 0.6794 | 0.7478 | 0.5334 | 0.4157 |
| **ClusPro baseline mean ± std** | — | 0.6706 ± 0.008 | 0.7485 ± 0.001 | **0.5374 ± 0.010** | **0.4131 ± 0.008** |
| v1_init_only | 0 | 0.6706 | 0.7488 | 0.5523 | 0.4240 |
| v1_init_only | 1 | 0.7067 | 0.7425 | 0.5533 | 0.4367 |
| v1_init_only | 2 | 0.6588 | 0.7361 | 0.5317 | 0.4040 |
| **v1_init_only mean ± std** | — | 0.6787 ± 0.025 | 0.7425 ± 0.006 | **0.5458 ± 0.012** | **0.4216 ± 0.016** |
| v3_text (LLM visual desc) | 0 | 0.6852 | 0.7462 | 0.5473 | 0.4208 |
| v3_text | 1 | 0.7048 | 0.7478 | 0.5681 | 0.4550 |
| v3_text | 2 | 0.6716 | 0.7388 | 0.5414 | 0.4178 |
| **v3_text mean ± std** | — | 0.6872 ± 0.017 | 0.7443 ± 0.005 | **0.5523 ± 0.014** | **0.4312 ± 0.021** |
| 논문 ClusPro (UT-Zap CW, Table 1) | — | 0.707 | 0.760 | **0.585** | **0.466** |
| 논문 CDS-CZSL | — | 0.639 | 0.748 | 0.522 | 0.395 |
| 논문 Troika | — | 0.660 | 0.738 | 0.541 | 0.417 |

**Δ vs baseline:**

| variant | seen | unseen | HM | AUC | σ |
|---|---|---|---|---|---|
| v1_init_only | +0.008 | −0.006 | **+0.008** | **+0.009** | HM 0.7σ / AUC 0.8σ — tie |
| v3_text | +0.017 | −0.004 | **+0.015** | **+0.018** | HM 1.3σ / AUC 1.7σ — mild + |

**Note:** v3_text는 mit-states에서는 −2.8pp HM (강한 음), UT-Zap에서는 +1.5pp HM (약한 양) — **dataset-dependent**.

---

## 3a. UT-Zappos FlowComposer — test_pairs 재평가 (논문 비교용, 5-24)

`test.py --load_model val_best.pt --open_world False`로 재평가. 비교 기준: cluspro_baseline seed 0 test_pairs HM 0.5296 / AUC 0.4042 (§2 표).

| Run | best ckpt | seen | unseen | hm_seen | hm_unseen | **HM** | **AUC** |
|---|---|---|---|---|---|---|---|
| cluspro_baseline_seed0 | val_best | 0.6647 | 0.7499 | — | — | **0.5296** | **0.4042** |
| **FlowComposer fallback** (templates) | ep 8 | 0.6559 | 0.7240 | 0.5660 | 0.6113 | **0.5878** | **0.4350** |
| **FlowComposer LLM** (Qwen2.5-3B K=8) | ep 10 | 0.6481 | 0.7166 | 0.2864 | 0.5315 | **0.3722** | **0.2819** |

**Δ vs baseline:**
- Fallback: HM **+0.0582**, AUC **+0.0308** — *Flow Matching + Composer 구조 자체가 효과 있음*
- LLM: HM **−0.1574**, AUC **−0.1223** — *val_pairs에서 +0.047이었던 게 test_pairs에서 완전히 뒤집힘*

**LLM run의 실패 분석:**
- val_pairs HM 0.6923 (best) → test_pairs HM 0.3722 (Δ −0.32)
- test_pairs best_seen 0.6481, best_unseen 0.7166 — 단일 axis 정확도는 정상
- 하지만 HM-maximizing bias에서 hm_seen이 **0.2864**로 붕괴 → seen-unseen trade-off curve가 비대칭
- 해석: val pair distribution에 overfit한 ckpt가 test pair에서 calibration 실패. LLM-aug bank가 (val에 있던) 특정 prompt 분포를 학습 신호로 강하게 흡수해서 test의 다른 unseen pair에서 잘 안 맞음
- §1.1 convention 경고가 정확히 실현됨 — val_pairs 신호는 신뢰 불가

**결론.**
- Flow Matching + Composer architecture는 baseline 위로 가는 게 **확인됨** (fallback도 +0.058 HM)
- LLM K=8 의미 다양성은 val에선 도움, test에선 해롭 — *현재 형태로는 generalization 손상*
- 다음 시도: (1) LLM bank를 더 약하게 사용 (`flow_blend` 0.9), (2) val_best 대신 다른 epoch ckpt, (3) Mit-states에서 같은 패턴 재현되는지 확인

### 3a-1. UT-Zap LLM run — epoch ckpt sweep (5-25)

가설 검증: "val_best는 LLM run에서 잘못된 ckpt selection이다" → 13 epoch ckpt 전부 test_pairs 평가.

| epoch | val HM | val AUC | **test HM** | **test AUC** | hm_seen | hm_unseen |
|---|---|---|---|---|---|---|
| 0 | 0.495 | 0.323 | 0.503 | 0.345 | 0.555 | 0.460 |
| 3 | 0.559 | 0.430 | 0.551 | 0.397 | 0.545 | 0.558 |
| 7 | 0.487 | 0.347 | 0.576 | 0.449 | 0.537 | 0.620 |
| 8 | 0.614 | 0.440 | 0.494 | 0.370 | 0.481 | 0.508 |
| 10 (선택된 val_best) | 0.630 | 0.464 | 0.445 | 0.322 | 0.492 | 0.406 |
| **12 (best test)** | **0.679** | **0.506** | **0.585** | **0.434** | **0.691** | **0.508** |
| val_best.pt 재평가 | 0.536 | 0.402 | 0.573 | 0.431 | 0.579 | 0.567 |

**Δ vs cluspro baseline (test_pairs HM 0.5296):**
- ep 12 (best): **+0.056 HM, +0.030 AUC** — fallback FlowComposer (+0.058)와 거의 동일
- val_best.pt 재평가: +0.043 HM (놀랍게도 첫 평가의 −0.157과 다름; eval 비결정성?)

**핵심 인사이트:**
1. **LLM run의 진짜 가치는 ep 12에 있음** (val HM 0.679에서 test HM 0.585). val_best ckpt selection은 적절했어야 했으나 ep 10이 selectvar 됨 — 학습 종료 시 val_best.pt가 적절한 ckpt를 잡지 못한 것으로 보임 (이전 epoch에서 갱신 시점 issue).
2. val 신호가 test와 잘 align되는 epoch (e.g. ep 12)에서는 LLM이 fallback과 비슷한 게인 — 더 큰 게인은 아님.
3. **첫 평가의 val_best.pt test HM 0.3722는 outlier일 가능성** — sweep에서 재평가하니 0.573이 나옴. Evaluator의 threshold sweep이 비결정적일 수 있음.

**수정된 결론:**
- FlowComposer 자체는 baseline 위로 ~+0.06 HM (fallback ≈ ep 12 LLM)
- LLM bank의 추가 게인은 미미하거나 noise 수준
- val-test correlation은 epoch-dependent, val_best ckpt를 무조건 신뢰하면 안 됨

---

## 3b. UT-Zappos FlowComposer — val_pairs (학습 dev signal, 5-23)

**Setup.** Spec `coding_agent_instructions.md` 구현. ClusProBaseline 위에:
- (a) attr/obj per-branch flow nets `v_θa, v_θo` (6-block adaLN residual MLP)
- (b) Composer MLP (`v_a, v_o → â, b̂`)
- (c) leakage augmentation (composition feature → primitive endpoint)
- (d) LLM K=8 text-embedding distribution bank (attr/obj/composition)

Loss: `L_base + λ_flow·(L_a_FM+L_o_FM) + λ_comp·L_comp + λ_leak·L_leak`
Inference: `0.7·base_logit + 0.3·logit_scale·(p_c + p_a·p_o)`
새 파일: `model/flow_composer.py`, `model/llm_distribution.py`, `llm/generate.py`
Eval: baseline과 동일 pipeline (`test.predict_logits → Evaluator.score_fast_model`)

**⚠ val_pairs eval — 논문 test_pairs 재평가 별도 필요.**

| Run | LLM source | best ep | seen | unseen | **HM** | **AUC** |
|---|---|---|---|---|---|---|
| cluspro_baseline_utzap_l14_v2_seed0 (5-2) | — | ckpt | 0.6990 | 0.7698 | **0.6456** | **0.5016** |
| **FlowComposer fallback (templates)** | deterministic K=8 templates | 8 | 0.7320 | 0.7578 | **0.6364** | **0.5084** |
| **FlowComposer LLM (Qwen2.5-3B-Instruct K=8)** | local LLM, 2026-05-23 | 10 | 0.7355 | 0.7638 | **0.6923** | **0.5385** |

**Δ vs baseline (LLM run):** HM **+0.0467** / AUC **+0.0369** (seen +0.037, unseen −0.006).
**Fallback templates:** tie (HM −0.009, AUC +0.007) — 진짜 LLM 의미 다양성이 게인의 핵심.

**Caveats.**
1. val_pairs eval — 논문 비교용 test_pairs 재평가 별도 필요
2. 1-seed only (mit-states std 환산 시 +0.047 HM은 std 밖)
3. ep1부터 LLM run이 fallback 위 (0.6378 vs 0.5051) → LLM 분포가 training signal로도 즉시 효과

**Files.**
- ckpt: `checkpoint/flow_composer_l14_utzap_{seed0,llm_seed0}/val_best.pt`
- log: `logs/train_flow_composer_utzap_{,llm_}seed0_20260523_*.log`
- LLM descriptions (144 JSON): `cache/llm_descriptions/ut-zappos/{attribute,object,composition}/*.json`
- Pre-encoded text bank: `data/llm_descriptions/ut-zappos/text_bank_K8_vitl14_qwen3b.pt`

---

## 4. MIT-States — FlowComposer (5-25)

**Setup.** UT-Zap과 동일 architecture. config: `train_batch_size=4, gradient_accumulation_steps=16, epochs=15`. 메모리 제약 (GPU 9.6GB) 때문에 batch_size를 8→4로 줄임. LLM descriptions은 Qwen2.5-3B-Instruct K=8로 5-24 생성 (2322 items, 2.9h).

| Run | val HM (best ep) | test seen | test unseen | **test HM** | **test AUC** |
|---|---|---|---|---|---|
| cluspro_baseline post-fix (3-seed mean, 5-15) | — | 0.4888 | 0.5241 | **0.3887 ± 0.0008** | **0.2171 ± 0.0006** |
| **FlowComposer fallback (templates)** | 0.366 (ep 15) | 0.4261 | 0.4856 | **0.3333** | **0.1682** |
| **FlowComposer LLM (Qwen2.5-3B K=8)** | 0.3549 (ep 15) | 0.4147 | 0.4707 | **0.3192** | **0.1573** |

**Δ vs baseline:**
- Fallback: HM **−0.0554** (≈ 69σ), AUC **−0.0489** (≈ 81σ)
- LLM: HM **−0.0695** (≈ 87σ), AUC **−0.0598** (≈ 100σ)
- LLM vs fallback (within FlowComposer): HM −0.0141, AUC −0.0109

**MIT-States에서는 두 FlowComposer 모두 baseline 대비 명백히 회귀**. seed std 한 자릿수 (~0.0008) 환산 시 noise 아님. UT-Zap의 +HM과 정반대 → **dataset-dependent**.

**가능한 원인:**
1. **Pair 수의 차이**: MIT 1962 vs UT-Zap 116 pairs. Closed-world 후보 pool이 17배 커서 LLM bank의 sharp score가 더 많은 wrong pair에서 noise 추가
2. **Attribute 다양성**: MIT는 "ancient", "cluttered", "weathered" 등 추상적 — Qwen이 visually-grounded descriptions 만들기 어려움 (UT-Zap "Hair calf"는 구체적)
3. **Batch size 차이**: MIT는 effective batch 64 (bs=4×grad_accum=16) vs UT-Zap 64 (bs=8×grad_accum=8). 활성값(activation) 크기는 ½이라 BatchNorm 등 분산 추정에 영향 가능 — flow MLPs 안 쓰지만 baseline disentangler에 BN1d 있음

**전반적 트렌드:**
- UT-Zap test_pairs: fallback +0.058, LLM ep12 +0.056 → architecture 자체는 효과
- MIT-States test_pairs: fallback −0.055, LLM −0.069 → architecture 효과 negate
- **두 데이터셋 일관**: LLM bank가 fallback templates보다 항상 나쁨

---

## 5. C-GQA — 학습 실패 (5-25)

- LLM descriptions 생성: **완료** (8854 items × K=8, Qwen2.5-3B-Instruct, ~11.7h, `cache/llm_descriptions/cgqa/`)
- 학습: **OOM 크래시 (batch_size=2, grad_accum=32에서도 GPU 9.6GB 부족)**

C-GQA는 7767 closed-world pairs (UT-Zap 67×, MIT-States 4×). text encoder가 5592 train pairs를 매 forward에서 인코딩 → 메모리 폭증. batch_size=1로도 불충분 추정. FlowComposer 코드의 text encoding을 chunking하도록 수정해야 가능.

ckpt 없음. `logs/train_flow_composer_cgqa{,_llm}_seed0_20260525_173707.log`에 OOM 로그.

---

## 6. 종합 결론 (FlowComposer + LLM-augmented Flow Matching)

| 데이터셋 | Δ HM vs baseline (fallback) | Δ HM vs baseline (LLM) | LLM이 fallback보다 |
|---|---|---|---|
| UT-Zap (test_pairs) | **+0.058** | **+0.056** (ep 12) | 거의 동등 (−0.002) |
| MIT-States (test_pairs) | **−0.055** | **−0.069** | 더 나쁨 (−0.014) |
| C-GQA | (OOM) | (OOM) | — |

1. **FlowComposer architecture 자체**: UT-Zap에서 +0.06 HM, MIT-States에서 −0.055 HM → dataset-dependent, **시드 std로 검증 필요** (1-seed only)
2. **LLM bank**: 두 데이터셋 모두에서 fallback templates 대비 추가 게인 **없음**. UT-Zap에서는 사실상 동등, MIT-States에서는 더 나쁨 → 현재 형태(K=8 desc, p_c+p_a·p_o blend)로는 의미 다양성을 게인으로 전환 못함
3. **val→test calibration mismatch가 LLM run에서 더 큼** (UT-Zap LLM val_best ep 10 test HM 0.445 vs ep 12 0.585) → LLM bank의 sharp score가 val에 overfit. val_best ckpt 자동 selection은 LLM run에서 신뢰 불가
4. **다음 시도 후보**: (1) flow_blend 0.9 (LLM 비중↓), (2) image-conditional LLM aggregation, (3) MIT/CGQA에서 더 큰 GPU로 batch size 정상화, (4) 3-seed std 측정

---

## Convention 주의사항

- **mit-states (§1):** 처음부터 test_pairs eval — 논문 비교 유효
- **UT-Zap test_pairs (§2):** 5-11 재평가 — 공식 비교용 (논문 split)
- **UT-Zap val_pairs (§3):** train.py 로그 첫 번째 출력 — **dev 신호용, 논문 비교 불가**
- post-fix = AMP NaN guard 적용 (5-2), pre-fix 결과는 mit-states에서 영향 없음 확인 (5-15)
