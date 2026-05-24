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

## 3. UT-Zappos — FlowComposer (LLM-augmented Flow Matching, 5-23)

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

## Convention 주의사항

- **mit-states (§1):** 처음부터 test_pairs eval — 논문 비교 유효
- **UT-Zap test_pairs (§2):** 5-11 재평가 — 공식 비교용 (논문 split)
- **UT-Zap val_pairs (§3):** train.py 로그 첫 번째 출력 — **dev 신호용, 논문 비교 불가**
- post-fix = AMP NaN guard 적용 (5-2), pre-fix 결과는 mit-states에서 영향 없음 확인 (5-15)
