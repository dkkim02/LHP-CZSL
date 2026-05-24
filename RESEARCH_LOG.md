# LHP-CZSL Research Log

작성일 2026-05-02 16:15 KST. 4-27 ClusPro baseline (K=5) 시작 시점부터 오늘 3-seed 본런 시작 직전까지의 정리. **5-3 09:00 업데이트**: UT-Zap v1_init_only 3-seed 결과 (§1) + baseline tie 결론 + 다음 단계 후보 (§4). **5-3 21:35 업데이트**: v3_text 3-seed 본런이 16:08 KST에 끝났으나 train loss와 test 숫자가 v1_init_only와 비트 일치 → **invalid run으로 폐기**, §4.5에 디버깅 단서 정리. **5-5 20:35 업데이트**: mit-states LLM feasibility 점수 28,175 조합 생성 완료 (§6). **5-6 10:54 업데이트**: v3_text refinement Phase A — LLM visual descriptions를 mit-states에 적용한 3-seed 본런 launch (§10). **5-10 22:19 업데이트**: 5-8 재현 런이 seed0 ep8 test 도중 SIGKILL로 사망 (디스크 96% + GPU 외부 contention 추정), 디스크 정리 후 seed0만 재시도 launch (§10-6). **5-11 16:30 업데이트**: seed0 retry 완주 + 3-seed test_pairs 재평가 → **Phase A negative 확정** (HM Δ=−0.028 vs baseline). 이전 "+0.012 HM 신호"는 val_pairs(우리) vs test_pairs(논문/baseline) 비교 오류였음. §1 mit-states 표 + §10-5 분기 갱신, §11 신설(convention 정리 + ClusPro 재현 갭). **5-16 R2D2 진입 결정 업데이트**: §11-18 결과 보고 사용자가 axis pivot. v3_text 류 add-on은 봉인했지만 description 정보는 살린다 — **vision-side 구조를 새로 설계해서 distillation으로 흡수, inference에는 description을 안 쓴다**. CLIP ViT patch grid(기존 CZSL이 안 쓰는 lever) + LLM description spatial richness 결합. 4-stage plan + pre-registered judgment 사전 등록. 상세 §13. **5-14 21:27 업데이트**: UT-Zap chain (Troika baseline + Troika v3_text-claude) 종료. baseline HM 0.5139 / AUC 0.3782, v3_text-claude HM 0.4612 / AUC 0.3252 → **Δ HM −0.053, AUC −0.053** (사전 등록 "<−0.005" bucket 적중). **UT-Zap에서도 v3_text negative → dataset-dependency 가설 기각, v3_text axis 완전 봉인**. 면담 page 9 +1.5pp 신호의 출처는 LHP-CZSL framework noise + short categorical sub-name 형식이었음. 상세 §11-18. **5-15 03:55 업데이트**: 4-27/28 baseline은 pre-AMP-fix(5-2 이전) ckpt라 노이즈 corrupt 위험 있음. 비교 기준선 재확보를 위해 **mit-states cluspro_baseline post-fix 3-seed 재학습 launch** (config: `cluspro_baseline_mit_l14_v2_seed{0,1,2}.yml`, GPU 1, sequential). 첫 batch 1.08 it/s, 1 run ≈ 11.6 GB VRAM. paperhp probe(5-11) 결과는 baseline 대비 test 거의 동률(HM −0.004, AUC +0.0004) → **paperhp 설정 폐기**. **5-16 업데이트**: 5-15 baseline 3-seed 완주 (22:34 KST 종료). post-fix 3-seed mean HM 0.3887 / AUC 0.2171, **4-27 pre-fix single-seed(0.3893/0.2169)와 사실상 동일** (Δ HM −0.0006, Δ AUC +0.0002). 시드 std HM ±0.00076 / AUC ±0.00056로 매우 작음 → **mit-states/ViT-L/14에서는 AMP NaN 오염이 metric에 영향 없었음** 확인 ([[project_lhp_czsl_amp_nan]] 가정 일부 기각). 기존 mit-states 표의 1-seed 결과 모두 신뢰 유효. v3_text −2.8pp HM은 시드 std 대비 ~35σ → 분명한 음의 신호. 상세 §12. **5-17 09:46 업데이트**: R2D2 distill_weight sweep(dw=0.1/0.3/0.5) 완주. dw=0.1이 best (test HM 0.3832, Δ vs baseline −0.0055 ≈ 7σ). 셋 다 사전 등록 봉인 조건(<−0.005) 적중하지만, val_metric=best_loss로 ep1-2 ckpt 선택 → val HM peak(ep2-3, 0.41+) 회수 못함. **봉인 확정 전 best_hm val_metric ablation 1런(dw=0.1, seed0) 필요**. Over-distillation 가설은 부분 확인 (dw 1.0→0.1 test HM +0.0095, val peak +0.0078). 상세 §13-5f. **5-18 17:10 업데이트**: §13-5g best_hm ablation 완주 (5-17 23:29 ~ 5-18 04:36 KST). val_best=ep3 ckpt → test HM 0.3926 / AUC 0.2231. val_metric 단일 변경만으로 dw=0.1 best_loss 대비 HM +0.0094, AUC +0.0085 — val_metric 미스매치(§13-5d-2)가 격차 일부의 원인 확인. 그러나 baseline 대비 Δ HM +0.0039 (~5σ) / Δ AUC +0.0060 (~10σ)로 **사전 등록 cutoff(+0.005) 진입 실패, 0.0011 short**. tie 영역(0.3837 ≤ HM < 0.3937) 진입 → **§13-5g 룰에 의해 R2D2 framework sealing 결정**, Stage 2 routing(§13-5c) launch 보류. SOTA 대비: Troika/CDS-CZSL/PLID range(HM 0.390~0.393) 도달, ClusPro 논문 SOTA(HM 0.407)에는 −1.4pp 미달 — 우리 ClusPro 재현 갭(§11-2, −1.8pp) 안에 머무름. 상세 §13-5h. **5-18 17:30 axis pivot**: R2D2 봉인 후 다음 axis는 [[project_lhp_czsl_text_enrich]]의 Phase B — **image-conditional description selection on LHP-CZSL v3_text framework**. v3_text mean-pool 가설을 직접 검증: description content는 그대로 두고 sub-meaning aggregation만 image-cond softmax로 교체. CDS-CZSL과 차별: (1) primitive-level multi-prototype (variable K_p, CDS는 single state per primitive), (2) attr+obj 양쪽 image-cond (CDS는 attr/state-only). 4-stage plan + pre-registered judgment §14 신설. **5-18 18:05 업데이트**: Stage 0 probe (1ep sanity) 통과 — train loss 2.53→1.68 monotone, NaN 0회, VRAM 13GB. PyTorch 1.11 scatter_reduce_ 미지원으로 helper signature 변경 (padded indices + masked F.softmax, semantically 동일). **Stage 1 launch (mit-states seed 0, 15 epoch, PID 4025895)** — ETA 5-19 00:00 KST. 상세 §14-9.

모든 실험: ViT-L/14 backbone, fp16 AMP, batch_size=8, grad_accum=8, 15 epochs (probe 제외), seed=0 (3-seed run 제외), val_metric=best_hm. 데이터셋 경로 등 환경 설정은 [LHP-CZSL setup memory](/home/student/.claude/projects/-home-student-dongki/memory/project_lhp_czsl_setup.md) 참고.

---

## 1. 결과 요약 테이블

### mit-states (ViT-L/14, 15 ep) — closed-world **test_pairs** (apples-to-apples with paper)

수치 출처: `checkpoint/<run>/val_best.closed.json`의 `test` 블록 (test_pairs eval, val-best ckpt 기준). 논문(ClusPro ICLR 2025 Table 1 등)이 보고하는 split과 동일.

| Run | Date | seed | seen | unseen | **HM** | **AUC** | attr | obj |
|---|---|---|---|---|---|---|---|---|
| ClusPro baseline (K=5) | 4-27 | 0 | 0.4899 | 0.5203 | **0.3893** | **0.2169** | 0.3856 | 0.5553 |
| LHP-CZSL v2 (LLM-derived K) | 4-28 | 0 | 0.4891 | 0.5241 | **0.3897** | **0.2179** | 0.3836 | 0.5573 |
| ClusPro baseline K=3 (ablation) | 4-29 | 0 | 0.4870 | 0.5225 | **0.3855** | **0.2158** | 0.3868 | 0.5563 |
| LHP-CZSL v1 (sub-meaning init + sem + decorr) | 5-1 | 0 | 0.4866 | 0.5250 | **0.3886** | **0.2182** | 0.3862 | 0.5584 |
| LHP-CZSL v1_init_only (sem=0, decorr=0) | 5-1→5-2 | 0 | 0.4878 | 0.5202 | **0.3874** | **0.2155** | 0.3851 | 0.5552 |
| v3_text (LLM visual desc, K=3) | 5-6 | 0 | 0.4882 | 0.4715 | 0.3572 | 0.1887 | 0.3550 | 0.5263 |
| v3_text seed0 retry (5-11) | 5-11 | 0 | 0.4840 | 0.4699 | 0.3569 | 0.1890 | 0.3555 | 0.5272 |
| v3_text | 5-6 | 1 | 0.4903 | 0.4782 | 0.3645 | 0.1938 | 0.3608 | 0.5334 |
| v3_text | 5-6 | 2 | 0.4895 | 0.4810 | 0.3633 | 0.1954 | 0.3601 | 0.5356 |
| **v3_text 3-seed mean** (retry seed0 + 5-6 seed1/2) | 5-11 | — | **0.4879** | **0.4764** | **0.3616** | **0.1927** | — | — |
| ClusPro baseline + paper hp (wd 5e-5, α 0.2, β 0.5) | 5-11→5-12 | 0 | 0.4912 | 0.5228 | **0.3853** | **0.2173** | 0.3893 | 0.5604 |
| **ClusPro baseline post-fix** | 5-15 | 0 | 0.4933 | 0.5226 | 0.3879 | 0.2176 | 0.3842 | 0.5590 |
| **ClusPro baseline post-fix** | 5-15 | 1 | 0.4857 | 0.5276 | 0.3889 | 0.2172 | 0.3858 | 0.5561 |
| **ClusPro baseline post-fix** | 5-15 | 2 | 0.4874 | 0.5220 | 0.3894 | 0.2165 | 0.3852 | 0.5546 |
| **ClusPro baseline post-fix 3-seed mean ± std** | 5-15 | — | 0.4888 ± 0.004 | 0.5241 ± 0.003 | **0.3887 ± 0.0008** | **0.2171 ± 0.0006** | 0.3851 | 0.5566 |
| **논문 ClusPro (Table 1)** | ICLR'25 | — | 0.521 | 0.540 | **0.407** | **0.238** | — | — |
| 논문 CDS-CZSL / Troika / PLID | — | — | 0.497~0.503 | 0.529~0.530 | 0.390~0.393 | 0.221~0.224 | — | — |

**상황 요약 (5-16 갱신):**
- 5-15 post-fix 3-seed baseline의 mit-states 시드 std는 HM ±0.0008 / AUC ±0.0006로 **종전 가정(HM ±0.005)보다 한 자릿수 작다**. 1-seed 측정의 한계 우려도 상당 부분 해소.
- post-fix mean(HM 0.3887, AUC 0.2171)이 4-27 pre-fix single-seed(HM 0.3893, AUC 0.2169)와 **시드 std 안에서 동률** → mit-states에서 AMP NaN 오염이 metric에 영향 안 줬음 ([[project_lhp_czsl_amp_nan]] "pre-fix mit-states 숫자도 신뢰 가능"으로 갱신). 따라서 기존 표의 v1/v2/v1_init_only/baseline tie 결론은 **사후적으로 신뢰 회복**.
- **v3_text는 baseline 대비 HM Δ=−0.027, AUC Δ=−0.024** — 시드 std로 환산 시 **HM 약 35σ, AUC 약 40σ** 떨어짐. 노이즈가 아니라 진짜 회귀. unseen이 특히 ~0.048 떨어지고 seen은 보존 → text-side description ensemble이 unseen 일반화를 손상.
- 우리 baseline 재현치 HM 0.3887는 **논문 ClusPro 0.407 대비 −1.8pp** — std 0.0008 기준 약 23σ → 재현 갭은 시드 노이즈가 아니라 진짜 구현 차이. §11-2 진단 유효.

**convention 경고 (5-11 발견)**: 5-2~5-10 사이 표에서 mit-states 표는 `test` 블록, UT-Zap §1 본표는 `val` 블록 — split 불일치 상태였음. **mit-states는 처음부터 test_pairs였고 그대로 유효**. UT-Zap 본표는 §1.1 부록의 test_pairs가 진짜 비교용. 5-11 재평가에서 v3_text를 잘못 val_pairs로 비교해 +0.012 양의 신호로 오판했던 게 이 convention 혼선의 직접 사례.

### UT-Zappos (ViT-L/14, seed 0) — closed-world test

| Run | epochs | NaN 상태 | seen | unseen | **HM** | **AUC** | 비고 |
|---|---|---|---|---|---|---|---|
| baseline lr=1e-4 | 15 | 오염 | 0.5885 | 0.7012 | **0.4902** | **0.3388** | epoch 1 step 1356에서 NaN 시작 |
| baseline lr=5e-5 | 15 | 오염 | 0.6285 | 0.6642 | **0.5009** | **0.3473** | step 1468에서 NaN 시작 |
| probe hsic_weight=0 lr=1e-4 | 1 | 오염 | 0.6373 | 0.6684 | **0.4861** | **0.3389** | step 1980에서 NaN 시작 |
| probe bf16 lr=1e-4 (no scaler) | <1 | 영구 오염 | — | — | — | — | step 813 NaN, val_best 미저장 → 크래시 |
| **probe guard lr=1e-4 (수정)** | **1** | **클린** | **0.6403** | **0.6785** | **0.4919** | **0.3493** | NaN 표시 0회, skip 18회(0.6%) |

### UT-Zappos — FlowComposer (LLM-augmented Flow Matching, 5-23)

**Setup.** Spec `coding_agent_instructions.md` 구현. ClusProBaseline 위에 (a) attr/obj per-branch flow nets `v_θa, v_θo` (6-block adaLN residual MLP), (b) Composer MLP (`v_a, v_o → â, b̂`), (c) leakage augmentation (composition feature → primitive endpoint), (d) LLM K=8 text-embedding distribution bank (attr/obj/composition). 학습 loss: `L_base + λ_flow·(L_a_FM+L_o_FM) + λ_comp·L_comp + λ_leak·L_leak`, 추론 score: `0.7·base_logit + 0.3·logit_scale·(p_c + p_a·p_o)`. 새 파일: `model/flow_composer.py`, `model/llm_distribution.py`, `llm/generate.py`. eval pipeline은 baseline과 동일 (`test.predict_logits → Evaluator.score_fast_model`, val_pairs convention per §1.1 deprecated).

| Run | LLM source | best ep | seen | unseen | **HM** | **AUC** |
|---|---|---|---|---|---|---|
| cluspro_baseline_utzap_l14_v2_seed0 (5-2) | — | ckpt | 0.6990 | 0.7698 | **0.6456** | **0.5016** |
| **FlowComposer fallback (templates)** | deterministic K=8 templates | 8 | 0.7320 | 0.7578 | **0.6364** | **0.5084** |
| **FlowComposer LLM (Qwen2.5-3B-Instruct K=8)** | local LLM, 2026-05-23 | 10 | 0.7355 | 0.7638 | **0.6923** | **0.5385** |

**Δ vs baseline (LLM run, best epoch=10):** HM **+0.0467**, AUC **+0.0369**, seen +0.037, unseen −0.006. Fallback templates는 baseline tie (HM −0.009, AUC +0.007) — *진짜 의미 다양성이 있는 LLM 설명이 게인의 핵심*. K=8 단순 phrasing-only template로는 baseline 위로 못 올라감.

**Caveats.** (1) val_pairs eval — 논문 비교용 test_pairs 재평가는 별도 필요. (2) 1-seed only, std 모름 (mit-states 5-15 결과로 보면 UT-Zap std는 ~0.013 수준 → +0.047 HM은 명백히 std 밖). (3) Qwen-3B descriptions은 visually rich했음 (예: "Hair calf skin with tiny, intricate pigmentation details") — 더 큰 LLM(Llama-3-8B) 또는 image-conditional refinement는 추가 게인 가능성. (4) 처음 epoch에서 LLM run이 fallback보다 한 단계 위 (ep1 HM 0.6378 vs 0.5051) — LLM 분포가 training signal로서도 빠르게 정착.

**Files.** ckpt `checkpoint/flow_composer_l14_utzap_{seed0,llm_seed0}/val_best.pt`, log `logs/train_flow_composer_utzap_{,llm_}seed0_20260523_*.log`, descriptions `cache/llm_descriptions/ut-zappos/{attribute,object,composition}/*.json`, text-embedding bank `data/llm_descriptions/ut-zappos/text_bank_K8_vitl14_qwen3b.pt`.

---

### UT-Zappos (ViT-L/14, lr=1e-4, fix 적용, 15 ep) — 3-seed 본런 — **5-11 이후 deprecated**

⚠ **이 본표는 val_pairs eval (test.py:706 첫 번째 출력) 기준**으로 논문 비교 불가. 5-11 convention 정리 이후 §1.1 부록의 test_pairs 표가 진짜 비교용. 본표는 historical record로만 유지하고, 새 비교는 §1.1 부록 사용.

#### baseline (5-2 16:10 ~ 23:06)

| seed | seen | unseen | **HM** | **AUC** | attr | obj |
|---|---|---|---|---|---|---|
| 0 | 0.6990 | 0.7698 | **0.6456** | **0.5016** | 0.3989 | 0.8563 |
| 1 | 0.6807 | 0.7377 | **0.6251** | **0.4728** | 0.3892 | 0.8619 |
| 2 | 0.7149 | 0.7424 | **0.6485** | **0.4978** | 0.4079 | 0.8600 |
| **mean ± std** | **0.6982 ± 0.017** | **0.7500 ± 0.017** | **0.6397 ± 0.013** | **0.4907 ± 0.016** | — | — |

NaN 오염 pre-fix baseline (seed 0, HM 0.4902 / AUC 0.3388) 대비 **HM +0.150 / AUC +0.152** 개선. Fix가 의도한 대로 동작했음을 확인 — 이 3-seed 평균이 앞으로 LHP 변형 비교의 기준선.

#### LHP-CZSL v1_init_only (5-2 23:47 ~ 5-3 06:36)

baseline과 동일 조건 (lr=1e-4, 15 ep, fp16+GradScaler+guard), 추가로 sub-meaning name 기반 prototype init만 켬. `model/lhp_czsl.py:_update_prototypes`에 features 유한성 가드 미러링.

| seed | seen | unseen | **HM** | **AUC** | attr | obj |
|---|---|---|---|---|---|---|
| 0 | 0.7058 | 0.7612 | **0.6434** | **0.5007** | 0.3976 | 0.8544 |
| 1 | 0.6853 | 0.7326 | **0.6310** | **0.4737** | 0.3933 | 0.8600 |
| 2 | 0.7035 | 0.7390 | **0.6407** | **0.4887** | 0.4054 | 0.8578 |
| **mean ± std** | **0.6982 ± 0.011** | **0.7443 ± 0.015** | **0.6384 ± 0.007** | **0.4877 ± 0.014** | — | — |

vs baseline: **seen Δ=0.000 / unseen Δ=−0.006 / HM Δ=−0.001 / AUC Δ=−0.003**. 모든 metric이 시드 std 안. **UT-Zap에서도 sub-meaning init만으로는 baseline과 tie**. mit-states/ViT-L/14에서 본 패턴이 UT-Zap에서 그대로 재현됨.

#### §1.1 부록 — UT-Zap test_pairs (5-11 재평가, **공식 비교용**)

수치 출처: 9개 ckpt 각각에 대해 `test.py --yml_path <yml> --load_model val_best.pt --open_world False` 재실행 (5-11 16:00~16:25 GPU 1) → `checkpoint/<run>/val_best.closed.json`의 `test` 블록.

| run | seed | seen | unseen | **HM** | **AUC** |
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
| **논문 ClusPro (UT-Zap CW, Table 1)** | — | 0.707 | 0.760 | **0.585** | **0.466** |
| 논문 CDS-CZSL | — | 0.639 | 0.748 | 0.522 | 0.395 |
| 논문 Troika | — | 0.660 | 0.738 | 0.541 | 0.417 |

**Δ vs baseline:**

| variant | seen | unseen | HM | AUC | 시드 std 기준 σ |
|---|---|---|---|---|---|
| v1_init_only | +0.008 | −0.006 | **+0.008** | **+0.009** | HM 0.7σ / AUC 0.8σ — tie 안 |
| v3_text | +0.017 | −0.004 | **+0.015** | **+0.018** | HM 1.3σ / AUC 1.7σ — mild positive |

**결론 (5-11 갱신):**
- v1_init_only: tie 유지 (mit-states + UT-Zap 둘 다).
- **v3_text는 UT-Zap에서 mild positive (HM +1.5pp, AUC +1.8pp)** — 1.3σ 수준이라 단호하게 "이긴다"고 말하기 모호하지만 양의 방향성.
- 그러나 mit-states에서는 같은 v3_text가 −2.8pp HM으로 명백히 음. **데이터셋 의존적**.
- 논문 ClusPro UT-Zap 보고치(HM 0.585, AUC 0.466) 대비 우리 baseline은 −4.8 HM / −5.3 AUC. 재현 갭이 mit-states(−1.8/−2.1)보다 훨씬 큼. UT-Zap에서는 절대치 자체가 paper와 동떨어져 있어 "v3_text의 +1.5pp"가 paper 절대좌표에서 무엇을 의미하는지 모름.

**5-11 이전 §1.1 부록 (5-3 작성, 폐기)**: baseline HM 0.5481 / v1_init 0.5553으로 보고했었는데, 5-11 재평가에서 baseline HM 0.5374 / v1_init 0.5458로 나옴 — 둘 다 약 −0.010 시프트. 같은 ckpt + 같은 split이라 원인 불명. 가능성: ① 그때 final_model.pt를 썼고 지금은 val_best.pt 씀, ② test.py가 그동안 변경됨, ③ 그때 다른 yml/feasibility flag. 5-11 숫자가 현재 코드/현재 ckpt에서 재현 가능한 값이므로 이쪽을 정본으로 채택.

---

## 2. 시계열 narrative

### Phase 1 — mit-states K-as-hyperparameter (4-27 ~ 4-29)

가설: "LLM이 attribute/object마다 의미적으로 적절한 K(prototype 수)를 알려주면 모델이 더 잘 동작한다."

- **4-27** ClusPro baseline (K=5) → HM 0.3893 / AUC 0.2169
- **4-28** LHP-CZSL v2 (LLM이 attr/obj별 K를 추정, `cluster_min=3` floor 때문에 대부분 K=3) → HM 0.3897 / AUC 0.2179
- **4-29** ClusPro K=3 ablation (LLM 안 쓰고 그냥 K를 5→3으로 내리기) → HM 0.3855 / AUC 0.2158

결론: K를 1D 하이퍼파라미터로 보면 K=3, K=5, LLM-K 모두 사실상 동률. **LLM이 K 값을 정해주는 것 자체는 contribution이 아님**. K는 단순 정수가 아니라 "branch / no-branch"의 binary 신호로 봐야 의미가 있을 가능성을 메모.

### Phase 2 — sub-meaning name 기반 prototype init (5-1)

가설 전환: K 자체가 아니라 **sub-meaning name으로 prototype을 초기화**하는 것이 진짜 contribution일 것.

- **5-1** LHP-CZSL v1 (sub-meaning init + `sem_weight=0.05` + `decorr_weight=0.1`) → HM 0.3886 / AUC 0.2182
- **5-1→5-2** v1_init_only (init 유지, sem=0 / decorr=0) → HM 0.3874 / AUC 0.2155

결론: aux loss를 빼도 init만 남겨도 baseline tie. **mit-states/ViT-L/14에서는 sub-meaning name 방향도 dead end로 확인**. mit-states는 너무 거칠어서 sub-meaning 효과가 안 드러나는 것일 수 있음 → UT-Zappos / C-GQA로 이동 결정 (3 seed로).

### Phase 3 — UT-Zappos baseline 시도 + NaN 발견 (5-2 새벽~오후)

UT-Zap에서 ClusPro baseline + LHP 변형을 비교하기 위해 ClusPro baseline부터 돌림.

- **5-2 02:06** baseline lr=1e-4 → HM 0.4902 / AUC 0.3388. epoch 1 step 1356(47%)에서 train loss=NaN.
- **5-2 10:32** baseline lr=5e-5 → HM 0.5009 / AUC 0.3473. epoch 1 step 1468(51%)에서 NaN.

LR을 낮춰도 NaN이 ~100 step만 늦어짐. 시드 동일 → **데이터 순서 동일 → 같은 batch에서 터지는 패턴**. LR tune이 아니라 numeric 문제.

### Phase 4 — 오늘 오후 NaN 디버깅 (5-2 15:00 ~ 16:10)

#### 4.1 HSIC 비섹트 probe (hsic_weight=0, 1 epoch)
- 결과: NaN 여전히 발생, 다만 step 1980(69%)으로 +500 step 지연. **HSIC 단독 범인 가설 기각**.
- 부수적 발견: HSIC 빼도 metric 거의 동일 (HM 0.4861 / AUC 0.3389) → UT-Zap에서 HSIC 기여 ~0. `model/lhp_czsl.py:7` 주석 ("HSIC를 cosine decorrelation으로 교체, 소배치 안정성")과 일치.

#### 4.2 bf16 probe
- 가설: AMP fp16의 누적 numeric drift가 진짜 원인 → bf16 (fp32 동급 dynamic range)으로 해결 가능.
- 결과: **더 나쁨**. step 813에서 NaN, 이후 영구 NaN(2,790 step 동안 회복 안 됨), val_best 미저장으로 final test 크래시.
- 진단: GradScaler를 같이 뺐기 때문. fp16+GradScaler가 사실은 NaN gradient를 자동 skip해서 params를 NaN 오염으로부터 보호하고 있었음. bf16 단독은 그 안전망이 없어서 한 번 NaN 들어가면 영구 corruption.

#### 4.3 guard probe — 최종 fix
변경:
- `train.py`: fp16+GradScaler 복원 + 명시적 `if not torch.isfinite(loss): zero_grad + scheduler step + continue` 추가
- `model/cluspro_baseline.py:_update_prototypes` 진입 시 `torch.isfinite(batch_attr_f).all() and torch.isfinite(batch_obj_f).all()` 체크, 아니면 early return (proto_momentum=0.99 EMA queue 영구 오염 방지)

결과 (1 epoch, lr=1e-4):
- **NaN 표시 0회**, loss-skip 18회(2875 step 중 0.6%)
- HM 0.4919 / AUC 0.3493 → 1 epoch만 돌렸는데도 이전 15-epoch lr=5e-5 baseline (AUC 0.3473) 초과
- 이전 NaN 오염 환경에서는 학습의 절반 이상이 silently skip되고 있었다는 가설 확인

#### 4.4 핵심 교훈 (mit-states 결과 재해석 위험)
mit-states/ViT-L/14 결과들도 같은 NaN 오염을 겪었을 가능성이 있음. mit-states에서 baseline tie 결론은 **NaN 오염 환경에서의 tie**일 수 있고, 진짜 능력치 비교가 아니었을 수 있음. 향후 핵심 결과는 fix 후 재실행이 안전.

---

## 3. 코드/설정 변경 inventory

### 코드 변경 (5-2 오후)

- `train.py:46-86` — fp16 GradScaler 유지, autocast 유지. 추가:
  ```python
  if not torch.isfinite(loss):
      optimizer.zero_grad()
      scheduler = step_scheduler(scheduler, config, bid, len(train_dataloader))
      progress_bar.set_postfix({"train loss": ..., "skipped": "nan"})
      progress_bar.update()
      continue
  ```
- `model/cluspro_baseline.py:236-242` — `_update_prototypes` 진입 시 features 유한성 체크:
  ```python
  if not (torch.isfinite(batch_attr_f).all() and torch.isfinite(batch_obj_f).all()):
      return
  ```

### 새 config

- `config/cluspro_baseline_utzap_l14_hsic0_probe.yml` — bisect용 (HSIC=0, 1 ep)
- `config/cluspro_baseline_utzap_l14_bf16_probe.yml` — bf16 probe용 (사용 후 폐기 가능)
- `config/cluspro_baseline_utzap_l14_guard_probe.yml` — fix 검증용 (1 ep)
- `config/cluspro_baseline_utzap_l14_v2_seed{0,1,2}.yml` — 본런 3-seed (15 ep, lr=1e-4)

### 새 스크립트

- `scripts/run_baseline_utzap_3seeds.sh` — seed 0/1/2 순차 실행, 각 seed별 로그 분리

---

## 4. 현재 상태 + 다음 단계

### 완료
- **5-2 16:10 ~ 23:06**: 3-seed UT-Zap baseline (HM 0.6397 / AUC 0.4907). 비교 기준선 확정.
- **5-2 23:47 ~ 5-3 06:36**: 3-seed UT-Zap v1_init_only (HM 0.6384 / AUC 0.4877). **baseline과 tie** (val/test 두 split 모두).
- 결론: **sub-meaning name 기반 prototype init은 mit-states / UT-Zap 두 데이터셋 모두에서 baseline 대비 의미 있는 격차를 만들지 못함**. 단순 init 방향은 dead end로 확인.

### 폐기 (5-3 09:16 ~ 16:08 KST, invalid)
- **3-seed UT-Zap v3_text 본런 — invalid, 결과 신뢰 불가.** §4.5 참고.
  - 끝까지 NaN 0회로 정상 완료, config 덤프에 `text_ensemble=True` 정상 진입, 체크포인트 size도 v1 대비 +764B 차이.
  - 그러나 seed 0/1/2 모두에 대해 (a) 매 epoch tqdm train loss가 v1_init_only와 소수점 두 자리까지 동일, (b) `test.py` 5-line summary 5줄이 모두 v1_init_only와 비트 일치. 두 학습이 사실상 같은 trajectory를 탔다는 뜻.
  - 결론: 표에 올리지 않음. 표를 올리면 advisor 미팅에서 "v3=v1 tie"로 잘못 해석될 위험. 디버깅 후 재실행.

### (5-3 09:16 KST에 시작했던 v3_text 본런 원래 설명, 참고 보존)
- 코드 변경 (`model/lhp_czsl.py`):
  - `text_ensemble: True` 플래그 추가 — 켜지면 `soft_att_obj` shape이 `(N_attr+N_obj, dim)` → `(sum_Ka+sum_Ko, dim)`로 바뀌고 sub-meaning당 학습 임베딩 1개씩 (init = sub-meaning name 토큰 평균).
  - `_construct_token_tensors`: comp 분기는 sub 임베딩을 primitive별로 mean-pool 후 prompt slot에 주입; attr-only/obj-only 분기는 sub-level prompt 각각 인코딩.
  - `train_forward` / `val_forward`: text encoder 통과 후 attr/obj 헤드 feature를 `_pool_sub_features` (scatter_add 기반 mean) 로 per-primitive 환원.
  - `_format_sub_name`: K>1 sub 이름은 `_`/`.`을 공백으로 (e.g. `smooth_leather` → `smooth leather`); K=1이고 sub==primitive인 경우는 baseline tokenization 보존.
- UT-Zap 기준 sum_Ka=19, sum_Ko=16 (vs N_attr=16, N_obj=12) — text encoder 인코딩량 +20% 정도.
- 새 config: `config/lhp_czsl_v3_text_utzap_l14_seed{0,1,2}.yml` (baseline과 동일 조건: lr=1e-4, 15 ep, fp16+GradScaler+guard). 새 스크립트: `scripts/run_v3_text_utzap_3seeds.sh`.
- 마스터 로그 `logs/run_v3_text_utzap_3seeds_20260503_091602.log`, seed별 `logs/train_v3_text_utzap_seed{0,1,2}_20260503_091602.log`.
- 첫 3분 헬스체크: 22s만에 step 57/2875, train loss 1.5→1.48 수렴, NaN 0회. 정상.
- 예상 ~7h, 완료 ~16:30 KST.

### 끝나면 할 일 (v3_text 폐기로 보류)
1. v3_text vs baseline / v1_init_only 비교 (val + test 두 convention 모두):
   - **HM/AUC 차이가 시드 std 밖**(즉 `>0.013` HM, `>0.016` AUC) **+ 양의 방향**이면 sub-meaning text-side 효과 확인 → C-GQA로 확장.
   - tie면 sub-meaning 방향 전체가 dead end로 굳힘 → angle C (selective branching) 또는 다른 방향.
2. RESEARCH_LOG.md §1에 v3_text 표 추가.

### §4.5 v3_text invalid run 진단 (5-3 21:35)

증거:
- `logs/train_v3_text_utzap_seed{0,1,2}_*.log`의 첫 10개 tqdm `train loss=` 값이 `logs/train_v1_init_only_utzap_seed{0,1,2}_*.log`와 정확히 동일 (1.24, 1.35, 1.40, 1.45, 1.50, ...). baseline과는 다름 (1.26, 1.37, 1.42, ...).
- `test.py` 최종 5-line summary (best_seen / best_unseen / best_hm / AUC / attr_acc / obj_acc)가 v3_text seed 0/1/2 ↔ v1_init_only seed 0/1/2 비트 일치.
- config Namespace 덤프: `text_ensemble=True, sub_meanings_path='data/sub_meanings_utzap.json', decorr_weight=0.0, sem_weight=0.0` 정상.
- `data/sub_meanings_utzap.json` 검사: attr 19 sub (Faux.Fur K=2, Leather K=2, Sheepskin K=2 + 13×K=1) / obj 16 sub (Sandals/Clogs/Heels/Sneakers K=2 + 8×K=1). sum_Ka=19 ≠ N_attr=16, sum_Ko=16 ≠ N_obj=12 → text_ensemble path가 활성화될 조건은 만족.
- `model/lhp_czsl.py:99,310,366,381,571,605` text_ensemble 분기 코드는 sub-emb pool/scatter_add로 v1과 다른 graph를 만들어야 함.
- `checkpoint/lhp_czsl_v3_text_l14_utzap_seed0/`: epoch_*.pt 파일 size 1746013317 (v1: 1746012553), 차이 +764B. 7행 × 768dim × float32 = 21,504B 예상치와 안 맞음 → soft_att_obj가 실제로 (sum_Ka+sum_Ko, dim) = (35, 768)로 alloc 됐는지 의심스러움.

가설 (디버그 우선순위):
1. **a)** soft_att_obj가 (28, dim) 그대로 alloc 됐고, `_construct_token_tensors`의 `embs[:self.sum_Ka]` 슬라이싱이 길이 부족으로 fallback 동작.
2. **b)** `_construct_soft_prompt`에서 text_ensemble=True 분기는 탔으나 `tokens_to_init`이 K=1 case에서 primitive 이름 그대로라 `attr_pooled`/`obj_pooled`가 baseline과 동일한 init.
3. **c)** test.py가 학습된 v3 ckpt가 아닌 다른 ckpt(v1)를 로드. (가능성 낮음 — save_path는 분리됨.)

디버깅 결과 (5-3 21:50):
1. v3 epoch_0.pt의 `soft_att_obj.shape == (28, 768)` 확인. v1과 동일 shape — **가설 a/b 둘 다 부분 적중**: 학습 자체가 v1의 28-row prompt graph로 동작.
2. 직접 LHPCZSL을 수동 init하면 `(35, 768)` 정상. 둘의 차이는 **primitive 이름**이었음: train.py가 `'Faux.Fur'` → `'faux fur'`로 normalize한 후 모델에 넘기는데(`train.py:172-173`), `data/sub_meanings_utzap.json`의 key는 dotted/cased raw (`'Faux.Fur'`). `_load_sub_meanings`의 `if attr_name in attrs_data:` 매칭이 모두 실패 → 모든 primitive가 default `K=1, sub=[name]`으로 떨어져 v1과 사실상 동일한 구조.
3. mit-states json은 lowercase + no-dot이라 이 normalize의 영향이 없음 → 이전 mit-states 결과는 이 버그의 영향 없음.

수정 (5-3 21:50, `model/lhp_czsl.py:_load_sub_meanings`):
- json key를 train.py와 동일하게 `replace('.', ' ').lower()`로 normalize한 dict로 lookup.
- 함수 끝에 `[_load_sub_meanings] sub_meanings_path=... | attrs: total=N, K>1=k, sum_K=s | objs: ...` sanity print 추가.
- `text_ensemble=True`인데 K>1 매칭이 0이면 즉시 `RuntimeError` (silent regression 방지).
- 검증: 수정 후 동일 yml로 instantiate → `soft_att_obj.shape == (35, 768)`, `sum_Ka=19, sum_Ko=16` 정상.

상태:
- v3_text 3-seed 본런 **재실행 필요**. 폐기된 `checkpoint/lhp_czsl_v3_text_l14_utzap_seed{0,1,2}/`는 invalid이므로 정리 또는 _invalid suffix로 rename 후 새 학습. 동일 스크립트(`scripts/run_v3_text_utzap_3seeds.sh`)로 ~7h 예상.

### 다음 단계 후보 (B 결과 본 후)
- **C. selective branching**: K=1 vs K>1 binary signal로 재해석. K>1 attribute에만 branching.
- **D. mit-states fix 후 재실행**: §4.4 caveat 검증. 비용 큼.
- **A. C-GQA**: B가 효과 있으면 그쪽으로 확장; 효과 없으면 비용 대비 가치 낮음.

### 정리해야 할 것 (시간 날 때)
- probe 체크포인트 디렉터리 삭제: `checkpoint/cluspro_baseline_l14_utzap_{hsic0,bf16,guard}_probe/`
- mit-states 결과들을 fix 후 재실행할지 결정 (단, 시간 비용 큼; UT-Zap에서 의미 있는 격차가 나오면 mit-states는 secondary)

### 정리해 둘 미해결 의문
- fp16 GradScaler가 NaN gradient를 어떻게 skip하길래 prototype EMA가 결국 회복했는지 정확한 메커니즘은 unclear. (이전 fp16+scaler 환경에서도 step 1356~2790 동안 NaN 표시 후 회복 패턴이 있었음.) 진짜로 회복된 것인지, 아니면 EMA queue가 부분적으로만 오염됐다가 fresh batch들로 mix-out 된 것인지 미상. 새 가드 들어왔으니 실용적으로는 무관.
- 본질적 NaN 원인은 끝까지 안 잡았음 — HSIC도 contrastive도 단독 범인 아니었고, AMP fp16 + 어떤 batch의 특정 input combination이 numeric overflow 일으키는 패턴. 가드로 우회한 상태. 향후 같은 문제가 다른 데이터셋에서 더 심하게 나오면 그때 재조사.

---

## 5. 5-4 진단 라운드 — k_validation + prototype collapse (advisor 미팅 5/6 자료)

5/4 v3_text seed2 재실행이 16:13 KST에 끝나면서 baseline_v2 / v1_init_only / v3_text 모두 3-seed 정상 비교 가능. 동시에 thesis 자체에 대한 두 가지 진단을 수행.

### 5.1 v3_text seed2 재실행 결과

3-seed 최종 (test_pairs):
| seed | seen | unseen | HM | AUC |
|---|---|---|---|---|
| 0 | 0.6823 | 0.7536 | 0.5509 | 0.4326 |
| 1 | 0.6823 | 0.7488 | 0.5402 | 0.4194 |
| 2 (재실행) | 0.7128 | 0.7411 | 0.5299 | 0.4102 |
| **mean** | 0.6925 | 0.7478 | **0.5403** | **0.4207** |

baseline_v2 mean (HM 0.5481 / AUC 0.4279) 대비 **HM −0.008, AUC −0.007** — 시드 std 안. **v3_text도 baseline tie 확정**.

### 5.2 K-validation: LLM K vs CLIP visual K (UT-Zap, n=28 primitives)

스크립트: [`tools/k_validation.py`](tools/k_validation.py). primitive별로 학습 이미지를 CLIP ViT-L/14로 인코딩 → silhouette 기반 k_visual ∈ [1,5] 결정 → `sub_meanings_*.json`의 K_LLM과 비교. mit-states (v1 + v2)는 18:08~ 진행 중, 결과 나오면 추가.

| | n | Spearman ρ | exact agreement | mean k_vis | mean k_llm | k_vis 분포 | k_llm 분포 |
|---|---|---|---|---|---|---|---|
| UT-Zap **attrs** | 16 | **−0.450** (p=0.08) | 3/16 (19%) | 3.00 | 1.19 | K=2:8, K=3:2, K=4:4, K=5:2 | K=1:13, K=2:3 |
| UT-Zap **objs**  | 12 | −0.316 (p=0.32) | 4/12 (33%) | 2.50 | 1.33 | K=2:10, K=5:2 | K=1:8, K=2:4 |

**해석**:
1. **음의 상관** — LLM이 K 크다고 한 primitive에서 visual K는 오히려 작음. ρ=0이면 "다른 차원" 변명 가능, ρ<0이면 LLM K가 visual diversity와 **체계적으로 어긋남**.
2. **Visual K는 모두 ≥2** — 전 primitive가 시각적으로 ≥2 cluster. LLM은 attrs 81%, objs 66%를 K=1 처리.
3. **Mean 크게 어긋남** (visual ~3 vs LLM ~1.2) — LLM이 시각적 다양성을 **체계적으로 과소평가**.

UT-Zap 한 데이터셋만으로 단정은 무리지만, "LLM zero-shot K가 visual diversity proxy로 작동한다"는 가설은 UT-Zap에선 **명확히 기각**. mit-states 결과도 비슷하면 K determination 메커니즘 자체 재설계 근거 확보. (caveat: silhouette는 instance-level variation도 포함하므로 LLM이 측정하는 "semantic sub-type"과 정확히 같진 않음. 그러나 anti-correlation은 그 caveat를 넘는 신호.)

### 5.3 Prototype collapse 진단 — 모든 학습 ckpt에서 effective K 측정

스크립트: [`tools/prototype_diagnostic.py`](tools/prototype_diagnostic.py). 각 `val_best.pt`에서 모든 `attr_queue{i}` / `obj_queue{i}` (각 [K=5, D=768])를 꺼내 per-primitive로:
- mean off-diagonal cosine — prototype 간 유사도 (1 = collapse, 0 = orthogonal)
- effective K = (Σs)² / Σs² (singular values) ∈ [1, K] — 실제로 사용되는 prototype 차원 수

전 16개 체크포인트 결과:

| checkpoint | attr off-cos | attr eff_K | obj off-cos | obj eff_K |
|---|---:|---:|---:|---:|
| **cluspro_baseline_l14_mit** | **0.994** | **1.08/5** | **0.986** | **1.12/5** |
| cluspro_baseline_l14_mit_k3 | 0.989 | 1.08/3 | 0.981 | 1.10/3 |
| cluspro_baseline_l14_utzap (lr=1e-4) | 0.800 | 2.21/5 | 0.939 | 1.36/5 |
| cluspro_baseline_l14_utzap_lr5e5 | 0.914 | 1.67/5 | 0.975 | 1.21/5 |
| **cluspro_baseline_l14_utzap_v2_seed0** | **0.967** | **1.37/5** | **0.992** | **1.11/5** |
| cluspro_baseline_l14_utzap_v2_seed1 | 0.999 | 1.06/5 | 1.000 | 1.02/5 |
| cluspro_baseline_l14_utzap_v2_seed2 | 0.993 | 1.15/5 | 0.999 | 1.04/5 |
| **lhp_czsl_v1_init_only_l14_mit** | **0.024** | **3.88/5** | **0.052** | **3.65/5** |
| lhp_czsl_v1_init_only_l14_utzap_seed0 | 0.967 | 1.37/5 | 0.992 | 1.11/5 |
| lhp_czsl_v1_init_only_l14_utzap_seed1 | 0.988 | 1.20/5 | 0.998 | 1.06/5 |
| lhp_czsl_v1_init_only_l14_utzap_seed2 | 0.981 | 1.27/5 | 0.996 | 1.08/5 |
| lhp_czsl_v1_l14_mit | 0.023 | 3.88/5 | 0.051 | 3.65/5 |
| lhp_czsl_v2_l14_mit | 0.317 | 2.07/5 | 0.318 | 2.05/5 |
| **lhp_czsl_v3_text_l14_utzap_seed0** | **0.016** | **3.89/5** | **0.029** | **3.74/5** |
| lhp_czsl_v3_text_l14_utzap_seed1 | 0.025 | 3.88/5 | 0.033 | 3.74/5 |
| lhp_czsl_v3_text_l14_utzap_seed2 | 0.015 | 3.88/5 | 0.046 | 3.74/5 |

**핵심 관찰**:

1. **ClusPro baseline은 K=5를 사실상 안 씀**. mit에서 eff_K ≈ 1.08, off-cos ≈ 0.99 — 5개 prototype이 거의 동일 vector. utzap_v2도 평균 eff_K ~1.2.
2. **LHP-CZSL는 데이터셋·variant에 따라 분리도 차이 큼**:
   - mit + sub-meaning init (v1, v1_init_only) → eff_K ≈ 3.88 (분리 잘 됨)
   - mit + 강제 K≥3 (v2) → eff_K ≈ 2.07 (중간)
   - **utzap + v1_init_only → 거의 collapse (eff_K ~1.2)** — utzap에선 init 효과 없음
   - **utzap + v3_text (text ensemble) → eff_K ≈ 3.88** (text-side ensemble이 분리 강제)
3. 모든 LHP variant도 buffer는 K=5 할당 (variable K는 loss/매스킹으로만 표현; tensor shape 동일).

### 5.4 Thesis 자체의 재정의 필요 — 5-3 (advisor 미팅 핵심 포인트)

진단 결과를 종합하면 thesis의 전제 자체가 흔들림:

**기존 thesis** (5/3 시점): "Variable K가 fixed K=5보다 효율적 — 단순 primitive에 K=5는 낭비"

**현 진단으로 드러난 사실**:
1. ClusPro baseline의 fixed K=5는 **이미 collapse해서 effective K=1**로 동작 중 (특히 mit, utzap_v2_seed1/2)
2. LHP variant의 eff_K가 baseline보다 훨씬 큰 경우(mit v1/v1_init/v3_text utzap eff_K~3.88)에도 **test 성능 tie**
3. LLM K는 visual diversity의 proxy 역할 못 함 (UT-Zap ρ=−0.45)

→ **재정의된 질문**: "Variable K vs fixed K=5"가 아니라 **"prototype 수가 1이든 4든 효과 없는데, 그럼 prototype 메커니즘 자체가 CZSL 성능에 의미 있는 lever인가?"**

이건 thesis를 **확대**하는 게 아니라 **축소**시키는 발견. advisor에게 솔직히 던질 질문:
- (Q1) ClusPro의 "5개 cluster prototype"이 collapse하는 게 정상 동작인지, 학습 dynamics 결함인지? (decorrelation loss 부재가 원인일 가능성)
- (Q2) Prototype memory가 효과 없는 게 검증되면, LHP-CZSL 전체 framing(prototype 중심)을 재고할지? text-side / loss-side로 axis 이동?

### 5.5 Falsifiable next experiment — 1~2일

**"올바른 K (visual K)로 학습 → 이기면 thesis salvageable, tie면 thesis 폐기"**

1. `tools/k_validation.py` 결과로 visual-K 기반 sub_meanings JSON 생성 (LLM 우회)
2. 동일 학습 (`v1_init_only` 구조) on UT-Zap, 3-seed
3. 결과:
   - baseline tie → variable K thesis 폐기. prototype 메커니즘 부분 자체를 재고. text-side / open-world / 다른 axis로 pivot
   - baseline 이김 → "LLM K가 문제였다" 증명. Path A(multimodal LLM) 또는 Path C(LLM+visual hybrid) 진행

이 실험 한 번이면 thesis salvage 가능성에 대한 binary 결론 확보. **advisor 미팅에서 이 실험을 제안하면 "다음 1주 로드맵"이 명확**.

### 5.6 미팅 자료 정리 (5/6 수요일용)

들고 갈 것:
- 이 §5 섹션 전체 (한 번에 나열) + §1.1 부록 테이블
- `logs/k_validation/k_validation_*.json` summary (UT-Zap + mit_v1 + mit_v2)
- `logs/k_validation/prototype_diagnostic.json`

핵심 talking points (3분 내):
1. **부정적 결과 정직 보고**: variable K로 baseline 못 이김 (3 variants × 2 dataset × 3 seeds 모두)
2. **두 가지 진단으로 원인 좁힘**:
   - (a) LLM K 자체가 visual diversity와 anti-correlated (UT-Zap ρ=−0.45)
   - (b) ClusPro baseline은 K=5를 사실상 안 씀 (eff_K ≈ 1.08)
3. **Thesis 재정의 제안**: "어떤 K가 좋은가"가 아니라 "prototype 수 자체가 lever인가"
4. **Falsifiable next step 제안**: visual K로 학습해서 thesis 운명 결정 (1~2일)

---

## 6. mit-states LLM feasibility 생성 완료 — 5-5 20:35

open-world masking 자산 확보 차원에서 mit-states 전체 (attr × obj) 28,175 조합에 대해 LLM(Gemini Flash) feasibility 0–10 점수 생성. `tools/llm_feasibility.py --pace_sec 1.0 --save_every 50`, output `data/feasibility_mit.json` (5.38 MB). 5-5 00:03 시작 → 20:35 종료, 약 20.5h, 실패(None) 1개.

| score | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| count | 133 | 4136 | 1727 | 1704 | 883 | 321 | 2401 | 4812 | 3867 | 7099 | 1091 | 6.09 |

- **score ≤ 2**: 5,996 / 28,175 (~21%) — open-world에서 마스킹할 후보 풀.
- **score = 9**가 7,099개로 가장 큼 → LLM이 "자연스러운 조합"에 9를 자주 부여. calibration은 그럭저럭.
- UT-Zap feasibility는 별도 `data/feasibility_utzap.json`에 이미 존재 (5-3 생성, RESEARCH_LOG에 기록 누락).
- **5-5 21:00 추가**: `data/feasibility_ut-zap50k.pt` (5-5 12:15)은 LLM JSON에서 변환된 파일로 확인 (byte-identical to LLM 변환 결과). 즉 cosine-feasibility 변형이 아니라 LLM-feasibility. RESEARCH_LOG에는 변환 이력이 누락됐었음.

용도:
- **A.** 학습 변경 없는 inference-time multiplicative mask로 open-world AUC 비교 (baseline / v1_init_only / v3_text on UT-Zap, single seed로 cheap pilot 가능).
- **B.** §5.4의 "다른 axis로 pivot" 후보 중 하나 — sub-meaning 방향 dead end 굳어진 만큼, open-world feasibility-aware decoding이 thesis 새 방향이 될지 검토.

### 6.1 평가 코드 mask 통합 (5-5 21:00)
- `parameters.py`: `--feasibility_path` 추가 (없으면 기존 `data/feasibility_{dataset}.pt` lookup 그대로).
- `test.py:639-643, 681-685`: 두 분기(open-world threshold sweep / fixed-threshold) 모두 override 사용; 로드 직전 `loading feasibility scores from {path}` 로깅.
- `tools/feasibility_json_to_pt.py`: 이미 존재. `--open_world` 플래그로 `list(product(attrs, objs))` 순서 정렬, 0-10 → 0-1 정규화, 누락 pair는 0.5 (neutral).

변환 결과:
- `data/feasibility_mit-states_llm.pt` — 28,175 pairs, feas mean **0.609**, min 0.0, max 1.0. 누락 1개(`barren fig`) → 0.5.
- `data/feasibility_ut-zap50k_llm.pt` — 192 pairs, feas mean **0.830**, min 0.1, max 1.0. 누락 1개(`Suede Slippers`) → 0.5. 기존 `feasibility_ut-zap50k.pt`와 byte-identical.

mit-states 분포 sanity check (LLM이 실제로 의미 있는 정렬을 만드는지):
- 가장 infeasible: `young lake`, `dry water`, `bent sky`, `peeled smoke`, `cooked diamond` — 모두 nonsensical.
- 가장 feasible: `wet glass`, `open fire`, `closed book`, `winding stream`, `new tie` — 모두 자연.

남은 작업:
- mit-states fix 후 재학습 결정 시 (§4.4 caveat) open-world 평가는 `data/feasibility_mit-states_llm.pt`로.
- 실패 1개씩(`barren fig`, `Suede Slippers`) 채우려면 `GEMINI_API_KEY=... python tools/llm_feasibility.py ... --output data/feasibility_{name}.json` 한 번 더 → 다시 변환 (incremental, ~몇 초).

### 6.2 UT-Zap open-world pilot (5-5 20:48, seed 0, with LLM mask)

3 모델 × seed 0 × `--open_world --feasibility_path data/feasibility_ut-zap50k_llm.pt`. test.py의 50-step threshold sweep으로 val AUC 최대화 → 그 threshold로 test 평가. 전부 **threshold=0.504** 선택 (LLM score ≥ ~5 유지). 약 4분 소요.

| 모델 | split | seen | unseen | HM | AUC |
|---|---|---|---|---|---|
| baseline_v2_seed0 | val | 0.7377 | 0.6021 | **0.5412** | **0.3959** |
| baseline_v2_seed0 | test | 0.6637 | 0.6192 | **0.3942** | **0.2553** |
| v1_init_only_seed0 | val | 0.7389 | 0.6350 | **0.5464** | **0.4149** |
| v1_init_only_seed0 | test | 0.6706 | 0.5939 | **0.3985** | **0.2637** |
| v3_text_seed0 | val | 0.7286 | 0.6183 | **0.5286** | **0.3962** |
| v3_text_seed0 | test | 0.6833 | 0.6034 | **0.4021** | **0.2647** |

closed-world test_pairs 대비 open-world drop:
- baseline_v2: HM −0.144 / AUC −0.158
- v1_init_only: HM −0.139 / AUC −0.146
- v3_text: HM −0.149 / AUC −0.168

세 모델 다 closed → open 격차가 비슷. open-world 안에서의 상대 순위는 test HM 기준 **v3 > v1 > baseline** (격차 ≤ 0.008) — 3-seed std (HM ±0.013, AUC ±0.016) 안이므로 single-seed로는 tie.

**관찰**:
1. CLI 오버라이드 (`--open_world`, `--feasibility_path`, `--load_model`) 정상 동작 — load 로그에 `loading feasibility scores from data/feasibility_ut-zap50k_llm.pt` 출력 확인.
2. 결과 저장 경로: `checkpoint/{run}/val_best.open.calibrated.json` (closed-world `val_best.closed.json`과 분리됨).
3. seed 0 pilot은 closed-world tie 결론을 open-world에서도 재현. 단 mask 자체의 기여는 unmasked baseline 없이는 정량 불가.

다음 후보:
- **mask off baseline** (`--threshold 0` 또는 all-ones feasibility) 1회 — mask 기여분 측정.
- **3-seed로 확장** — 동일 스크립트로 seed 1, 2 (각 ~4분).
- baseline checkpoint의 `epoch_*.pt`에서 다른 epoch 평가하면 closed-trained model의 open-world generalization 변화 추적 가능. (현재 val_best는 closed-world best_hm 기준이라 open-world에는 sub-optimal일 수 있음.)

로그: `logs/openworld_pilot/{model}_{ts}.log`, `logs/openworld_pilot/master_{ts}.log`.

### 6.3 Mask off vs Mask on + 3-seed 확장 (5-5 21:14)

mask off seed 0 (`--threshold 0`, mask 비활성과 동등) + mask on seed 1, 2 = 9 runs sequential, 약 10분 소요. 모든 결과 `checkpoint/{run}/val_best.open.{masked,unmasked}.json`로 분리 저장.

**Open-world 3-seed (with mask) — test_pairs**

| 모델 | HM (mean ± std) | AUC (mean ± std) | vs baseline |
|---|---|---|---|
| baseline_v2 | 0.4010 ± 0.013 | 0.2653 ± 0.011 | — |
| v1_init_only | 0.4114 ± 0.012 | 0.2794 ± 0.015 | HM +0.010, AUC +0.014 |
| v3_text | 0.4094 ± 0.015 | 0.2748 ± 0.019 | HM +0.008, AUC +0.010 |

closed-world (val_pairs HM peak)과 같은 ranking 재현: **v1 ≥ v3 ≥ baseline**, 격차는 시드 std 안. open-world에서도 sub-meaning 방향 dead-end 결론 일관.

**Mask off 3-seed (test_pairs)** — 5-5 21:30 추가:

| 모델 | HM (mean ± std) | AUC (mean ± std) |
|---|---|---|
| baseline_v2 | 0.4137 ± 0.015 | 0.2804 ± 0.013 |
| v1_init_only | 0.4221 ± 0.013 | 0.2925 ± 0.016 |
| v3_text | 0.4239 ± 0.015 | 0.2912 ± 0.019 |

**Mask Δ — 3-seed (mask on − mask off, test_pairs):**

| 모델 | ΔHM (mean ± std) | **ΔAUC (mean ± std)** |
|---|---|---|
| baseline_v2 | −0.0127 ± 0.0022 | **−0.0151 ± 0.0021** (≈ 7σ) |
| v1_init_only | −0.0107 ± 0.0093 | **−0.0130 ± 0.0114** (≈ 1σ; noisy) |
| v3_text | −0.0145 ± 0.0029 | **−0.0164 ± 0.0036** (≈ 4.5σ) |

**핵심 발견** — image-agnostic LLM feasibility mask는 **val→test transfer 실패**:
- val에서 미세 + (threshold가 val AUC 최대화로 선택됐으니 당연)
- test에선 모든 3 모델에서 일관되게 −0.014 ~ −0.021 AUC 손실
- threshold 0.504 (LLM score ≥ 5)가 val 기준 best지만 test에선 정상 페어 over-mask

이는 **§8-3 axis (B) "Image-conditional feasibility (VLM 기반)"의 직접적 motivation**:
- FLM류 image-agnostic LLM prior는 val→test에서 calibration 깨짐.
- 다음 step: VLM (Gemini Pro Vision / GPT-4V)으로 image × pair conditional feasibility 시도.
- "axis B 우선순위 격상" 결론을 advisor 미팅에서 깔끔히 전달 가능.

**Phase 1 (§9) 의사결정**: α' ≈ −0.02 (음수) → "Phase 1 결과가 negative" 분기 활성. axis B (image-conditional) 또는 axis G (reality-check 논문) 진행.


---

## 10. v3_text refinement Phase A — LLM visual descriptions on mit-states — 5-6 10:54 launch

§8 의사결정의 자연 후속. 사용자가 **mit-states 먼저** 결정 (UT-Zap은 후순위). 핵심 가설: v3_text의 +1.5pp HM (UT-Zap)이 sub-meaning이 너무 짧고 image-agnostic mean pool인 탓 → primitive마다 K_p≥3, 각 sub를 LLM 시각 description으로 교체하면 description content가 lever인지 가려진다.

### 10-1. 생성 (10:53 완료, ~10분)

- 도구: `tools/llm_descriptions.py` (Gemini 2.5 Flash Lite, `--K 3 --tokens 40 --pace_sec 0.5`)
- 출력: `data/descriptions_mit.json` (115 attr + 245 obj × 3 desc = **1080 sub-prompt**)
- 품질 sanity: CLIP token 길이 max 28 / mean 17.4 / p95 22 → 77 limit 여유. LLM 실패 0건.
- 샘플 (`ancient`):
  - "weathered, cracked leather-bound book with faded gold lettering on its cover"
  - "rough stone sculpture with moss growing in its crevices, showing signs of age"
  - "chipped ceramic vase with intricate, hand-painted designs and a dusty patina"

### 10-2. 학습 설정

- yml: `config/lhp_czsl_v3_text_mit_l14_seed{0,1,2}.yml` 신규 — `sub_meanings_path: data/descriptions_mit.json`, `text_ensemble: True`, `decorr_weight=0`, `sem_weight=0`. 나머지는 기존 v1_init_only mit l14와 동일 (lr=1e-4, bs=8, ga=8, 15ep, ViT-L/14).
- runner: `scripts/run_v3_text_mit_3seeds.sh` (UT-Zap runner mirror)
- v3_text는 sem_weight=decorr_weight=0이라 description JSON의 `d_sem={}` 비어 있어도 무관 — **description 컨텐츠 단일 변수만 변동**.

### 10-3. 런 진행

- 10:54 launch, GPU 1, seed0부터 순차. `_load_sub_meanings`: `attrs K>1=115/115 sum_K=345 | objs K>1=245/245 sum_K=735` (목표 1080 일치).
- 속도: ~2.8 it/s × 3793 step/epoch → **epoch ~22분, per-seed ~5.5h, 3-seed 합 ~16-17h**. (당초 30-40h 추정보다 빠름 — text encoder pass는 dominant cost가 아니었음.)
- GPU 1 utilization 94%, 13.5GB / 24GB.

### 10-4. 비교 baseline

- 비교 대상: §1 mit-states 표의 v1_init_only HM 0.3886 / AUC 0.2182 (1-seed, sem_weight=0, decorr=0). 같은 yml 변형이 v3_text로 description만 바꾼 셈이므로 controlled.
- 끝나면 mask on/off + test_pairs/val_pairs 모두 산출 → §1 표에 row 추가, 3-seed mean±std로.

### 10-5. 결과 분기 (binary) — 5-11 판정: ❌ NEGATIVE

설정한 분기 규칙:
- **HM ≥ +0.005, std 안에서 분리**: description content가 lever → Phase B (image-conditional selection)으로 차별화 작업.
- **HM ≤ +0.002 또는 std 안 동률**: Phase A ❌. axis G(reality-check paper) pivot 또는 Phase B만 단독 시도.

5-11 재평가 결과 (test_pairs 기준, §1 mit-states 표):

| | seen | unseen | HM | AUC |
|---|---|---|---|---|
| baseline (1-seed) | 0.4899 | 0.5203 | 0.3893 | 0.2169 |
| v3_text 3-seed mean | 0.4879 | 0.4764 | **0.3616** | **0.1927** |
| Δ (v3_text − baseline) | −0.002 | **−0.044** | **−0.028** | **−0.024** |

**판정: Phase A ❌ negative — "분리되어 음수"** (tie도 아님). Description ensemble은 mit-states에서 unseen 일반화를 손상시킴. 가설(description content가 lever)은 기각.

원래 분기 규칙은 "positive"와 "tie/null"만 가정했으나 실제로는 **clearly negative**인 세 번째 분기에 들어옴. axis G/Phase B 양쪽 다 Phase A에서 출발한 정당성이 약해짐 → §11에서 다음 단계 재설계.

#### 10-5.1. 측정 오류 사례 (5-11 발견, 후속 일반 교훈)

5-11 오전 사용자에게 처음 보고한 "v3_text mean HM 0.4014 vs baseline 0.3893 → +0.012 양의 신호"는 잘못된 비교였음:
- "v3_text mean 0.4014"는 train log의 epoch-by-epoch "best_*" 출력 = **val_pairs val-best running max**.
- "baseline 0.3893"은 `val_best.closed.json`의 test 블록 = **test_pairs of best-val ckpt**.
- 두 숫자는 다른 split이고 다른 의미라 비교 불가.

같은 ckpt에서 test_pairs로 재산출하니 v3_text mean = **0.3616** (val_pairs val-best보다 −0.040 낮음). 모든 모델에서 val_pairs > test_pairs인 게 일반적이므로 단순 split mismatch가 +0.040 inflation을 만들었음. 향후 §1 표 row 추가는 반드시 `closed.json`의 `test` 블록만 사용해야 함 (§11 convention 정리).

### 10-6. seed0 재현 시도 (5-8 11:07) → 중단, 5-10 22:19 재시도

- **5-8 11:07 launch**: 같은 wrapper로 v3_text mit 3-seed 재실행 시작. 동기는 §10-3에서 GPU contention 가능성 있던 5-6 런 결과 검증.
- **5-8 15:15 사망**: seed0 epoch_8.pt 저장 직후 test/val 루프 batch 69/326에서 로그가 통째로 끊김. **Python traceback 0줄** → SIGKILL. dmesg 권한 없어 OOM 확정 못 함. 정황상 강한 후보:
  - `/dev/nvme0n1p2` 96% (83GB free) 도달, `LHP-CZSL/checkpoint/` 단독 386GB. `save_every_n=1` × 13개 dir × 15 epoch × 1.75GB로 누적.
  - 같은 머신 다른 사용자 tmux(`kiseoup`)에서 무거운 4-GPU emulation 학습이 5/8~5/9 진행 중 — 시스템 메모리 압박 동시 발생 가능.
  - wrapper에 `set -e` → seed0 실패로 seed1/seed2는 시작도 안 함. 그러나 5-6 본런의 seed1/seed2 final/val_best는 그대로 보존.
- **5-10 22:19 정리 + 재시도**:
  - 정리: 13개 큰 ckpt dir에서 epoch_*.pt 일괄 삭제, val_best/final/config만 유지. 디스크 96%→78%, 386GB→69GB. `seed0/final_model.pt`(5-6 완주분)는 `final_model_5_6_backup.pt`로 백업.
  - yml 수정: `save_every_n: 1 → 100` (epoch_*.pt 미저장, 디스크 압박 방지).
  - 재시도 범위: seed0만. seed1/seed2의 5-6 결과 유지.
  - launch: nohup, GPU 1, log `logs/train_v3_text_mit_seed0_retry_20260510_221916.log`.

### 10-7. seed0 retry 완주 (5-11 05:04) + 3-seed test_pairs 재평가

- **재시도 결과**: 2026-05-11 05:04:07 KST rc=0 정상 종료. 15 ep 학습, val-best는 epoch 12에서 val HM 0.4063 / AUC 0.2302 (val_pairs 기준; **test_pairs는 별도**).
- **3-seed test_pairs 재평가 (5-11 14:00~16:00)**: 3개 ckpt(seed0 retry / seed1 5-6 / seed2 5-6) 모두에서 `test.py --load_model val_best.pt` 재실행 → `checkpoint/<run>/val_best.closed.json` 생성, `test` 블록 추출. 결과는 §1 mit-states 표 참고.
- **재현성 확인**: seed0 5-6 원본 (HM 0.3572) vs seed0 retry (HM 0.3569) — Δ=−0.0003, 동일 분포. 디스크/GPU contention 외부 요인 없이 안정 재현됨. SIGKILL은 환경 이슈였지 결과 자체에는 영향 없었다고 결론.

---

## 11. Convention 정리 + ClusPro 재현 갭 + 다음 단계 — 5-11 17:00

### 11-1. test_pairs vs val_pairs convention (반드시 준수)

CZSL 논문(ClusPro/CDS-CZSL/Troika/PLID 등)이 보고하는 closed-world 숫자는 **거의 항상 test_pairs**. 우리도 같은 split을 써야 비교 가능. 5-11 발견 전까지 표마다 split이 섞여 있었음.

**규칙 (5-11 이후 적용):**
1. §1/§1.x 표의 모든 `seen/unseen/HM/AUC` 숫자는 **`checkpoint/<run>/val_best.closed.json`의 `test` 블록**만 사용.
2. train log의 epoch-by-epoch `best_*` 출력은 **val_pairs val-best running max**. 표에 그대로 넣으면 안 됨 (논문 비교 불가, val_pairs는 보통 test_pairs보다 0.02~0.05 높음).
3. `val_best.closed.json`이 없는 ckpt는 `test.py --load_model val_best.pt`로 재산출. 비용 ~5 min/ckpt × mit-states 기준.
4. UT-Zap §1.1 부록 표(test_pairs)는 이미 이 convention. UT-Zap §1 본표는 val_pairs라 §1.1 부록만 신뢰. 본표는 5-11 이후 deprecated 표기 필요.

### 11-2. ClusPro 재현 갭 (mit-states −1.8 HM, UT-Zap −4.8 HM) — 5-11 진단

| dataset | metric | 우리 baseline | 논문 ClusPro Table 1 | Δ |
|---|---|---|---|---|
| mit-states (1-seed) | HM | 0.3893 | 0.407 | **−1.8** |
| mit-states (1-seed) | AUC | 0.2169 | 0.238 | **−2.1** |
| UT-Zap (3-seed mean) | HM | 0.5374 | 0.585 | **−4.8** |
| UT-Zap (3-seed mean) | AUC | 0.4131 | 0.466 | **−5.3** |

**원인 진단 (5-11 (1)+(2) 작업):**

**(1) 모델 코드 — 우리 reimpl은 공식의 절반 (442 vs 904줄)**

루트 `cluspro_baseline.py` (34KB)는 공식 `ClusPro/cluspro.py`와 byte-identical(0 diff). 그러나 **실제 학습에 import되는 건 `model/cluspro_baseline.py` (18KB)** — 우리가 따로 작성한 stripped 버전. `model_factory.py:8`이 이쪽을 사용.

공식에 있고 우리 reimpl엔 없는 메커니즘:

| 메커니즘 | 공식 위치 | 우리 reimpl | 영향 |
|---|---|---|---|
| `local_assign` (Optimal Transport coupling) | line 706, 731 | 없음 (cosine+softmax+Gumbel로 대체) | prototype 할당 분포가 다름. paper §4.4 "Classical OT vs Ours"가 우리는 둘 다 아닌 cheap 버전 |
| `distributed_sinkhorn` / `entropic_COT_*` | line 163-315 (~150줄) | 없음 | OT 보조 |
| `CrossAttentionLayer` + `MulitHeadAttention` (sic) | line 111-162 | 없음 | text-feature와 prototype 융합 path |
| `Disentangler2` | line 99-109 | 없음 | 2단 disentangle |
| `_dequeue_and_enqueue` (sample queue) | line 630-646 | 없음 | contrastive negative bank |
| `pos_neg`, `_sample_negative` | line 611-663 | 없음 | hard negative sampling |
| `forward_for_open` / `encode_text_for_open` | line 563-610 | 없음 (open-world 따로 짜야 함) | OW inference 경로 |

**3-branch text encoding 부분만 우리가 옮겼고 핵심 clustering/OT/fusion 메커니즘은 전부 빠짐**. UT-Zap에서 갭이 mit-states의 2.7배인 이유 후보: UT-Zap의 fine-grained shoe variation일수록 prototype OT/fusion 효과가 큼.

**(2) 하이퍼파라미터 — 3개 값이 paper와 다름**

paper §4.2 + Eq.12 vs `config/cluspro_baseline_mit_l14.yml`:

| param | paper §4.2 | 우리 yml | Δ |
|---|---|---|---|
| backbone | ViT-L/14 | ViT-L/14 | ✅ |
| K (cluster_num) | 5 | 5 | ✅ |
| μ (proto_momentum) | 0.99 | 0.99 | ✅ |
| τ (Eq.10 temperature) | 0.1 | (code hardcoded?) | 미확인 |
| κ (Eq.7) | 1 | (code hardcoded?) | 미확인 |
| epochs | 15 | 15 | ✅ |
| optimizer | Adam | Adam | ✅ |
| lr | 1e-4 | 1e-4 | ✅ |
| **weight_decay** | **5e-5** | **1e-5** | ❌ **5× 작음** |
| **α (PCL/contrastive weight)** | **0.2** | **0.1** | ❌ **2× 작음** |
| **β (PDL/HSIC weight)** | **0.5** | **0.1** | ❌ **5× 작음** |
| scheduler | (논문 미명시) | StepLR step=5 γ=0.5 | ? |
| attr_dropout | (논문 미명시) | 0.3 | ? |
| effective bs | (논문 미명시) | 8 × ga 8 = 64 | ? |

**(3) Paper vs released code 자체의 불일치 발견**

공식 `cluspro.py:528-552` (`loss_calu`):
```python
if self.training:
    loss = ... pair_loss_weight + attr_loss_weight + obj_loss_weight + 0.1*loss_contras  # +0.1*ppc
```
- `0.1*loss_contras` → α=**0.1** (paper §4.2의 0.2가 아님)
- `loss_hsic` 계산은 하지만 **loss에 안 더함** (주석 "# +0.1*ppc"가 commented out 표시)
- 즉 **공식 코드 = α=0.1, β=0**. 우리 yml(α=0.1, β=0.1)이 paper(α=0.2, β=0.5)보다 오히려 공식 코드와 가까움.
- paper 보고치 0.407은 paper §4.2 hyperparam(α=0.2, β=0.5)으로 돌린 게 맞다면, 공개 코드로는 그 숫자가 재현 안 됨. 공개 코드는 더 오래된 / 부분 release 가능성 강함.

**(4) eval calibration**

`test.py:101-128` `generate_predictions`: 50-trial bias linspace로 best AUC 채택. paper §4.1 "official evaluation protocol" 인용 reference [28]/[30]/[15]/[14]는 CGE/CSP/CDS-CZSL/Troika — 같은 protocol일 가능성 높음. 미세한 차이일지 미확인.

**갭의 가장 큰 기여**는 (1)의 메커니즘 누락(OT, cross-attn, queue contrast)일 가능성이 가장 높음 — 단순 weight 차이(2)로는 −4.8 HM 같은 큰 격차를 만들기 어려움.

### 11-3. v3_text 음의 격차 해석

unseen만 0.044 떨어지고 seen은 보존된 패턴이 시사하는 것:
- LLM description은 **각 primitive의 prototypical 시각 표현**을 담음. 예: `ancient` → "weathered leather book"/"stone sculpture"/"chipped vase". 이 셋은 학습 데이터(seen composition)에 자주 등장하는 컨텍스트.
- text ensemble로 prototype init이 이 prototypical 패턴 쪽으로 강하게 anchor → **unseen composition**에서 attr-obj 결합이 prototypical과 다를 때 표현이 mismatch.
- 즉 description ensemble은 "seen 분포의 prior"를 강화하는 방향이고, CZSL이 평가하는 unseen generalization과 반대로 작용.

이는 §10 도입부 가설("sub-meaning이 너무 짧고 image-agnostic mean pool인 탓")의 정반대 결론: 문제는 길이/내용이 아니라 **image-agnostic mean pool 자체**가 unseen에 해롭다는 점.

### 11-4. UT-Zap v3_text 재평가 결과 (5-11 16:00~16:25)

9개 ckpt(baseline/v1_init_only/v3_text × 3 seed) `val_best.pt`에 `test.py --open_world False` 재실행. 자세한 표는 §1.1 부록 갱신본 참고. 요약:

| | mit-states test_pairs Δ vs baseline | UT-Zap test_pairs Δ vs baseline |
|---|---|---|
| v1_init_only | tie (HM Δ=−0.002, 1-seed 비교) | **tie** (HM Δ=+0.008, 0.7σ) |
| v3_text | **negative** (HM Δ=−0.028) | **mild positive** (HM Δ=+0.015, 1.3σ) |

**핵심: v3_text 효과가 데이터셋 의존적.** 한 axis가 한 데이터셋에서 음, 다른 데이터셋에서 (약한) 양인 패턴은 일반화 가능한 메커니즘이 아닐 가능성이 높음. 가설:
- UT-Zap (12 obj × 16 attr = 192 max comp, 정밀 신발 미세 분류): description의 시각 디테일이 fine-grained 구별에 도움.
- mit-states (245 obj × 115 attr, ~28k comp, 거친 카테고리): description ensemble이 seen-prior로 작용해 unseen에 해.
- 또는 단순히 UT-Zap에서 우리 baseline 재현이 paper 대비 −4.8 HM로 크게 낮아 "baseline이 약한 상태"라 어떤 noise라도 +로 보일 수 있음 (reproduction gap이 큰 데이터셋이라 unstable).

### 11-5. 다음 단계 후보 (5-11 18:00 갱신, §11-2 진단 반영)

§11-2 진단에서 갭의 가장 큰 원인이 **모델 메커니즘 누락(OT/cross-attn/queue contrast)** 임이 드러남 — 단순 yml 튜닝으로는 안 닫힘. 메일 없이도 갈 수 있는 길:

1. **(a) yml 3개 값 정렬해서 cheap probe** (1-2일, 가장 저렴): paper §4.2와 어긋난 weight_decay=5e-5, contrastive_weight=0.2, hsic_weight=0.5로 yml 수정 → mit-states 1-seed 재실행 (15ep, ~6h). 갭의 작은 일부(예상 0.5-1.0 HM 회복)만 닫혀도 의미 있음. 단 paper와 공식 코드가 서로 모순(§11-2 (3))이라 정확히 어느 값을 쓸지는 사전 결정 필요. **추천: paper §4.2 값(α=0.2, β=0.5, wd=5e-5) 먼저 시도**.
2. **(b) 누락 메커니즘 1개씩 이식** (각 2-3일): 우선순위 — OT(`local_assign` + `distributed_sinkhorn`) → CrossAttentionLayer → queue 기반 contrastive. 각 단계마다 mit 1-seed 재실행하여 ΔHM 측정. OT가 paper §4.4 "Classical OT vs Ours"에서 가장 큰 단일 component 기여(+3.3/+3.1 AUC on UT-Zappos)였으므로 OT만 이식해도 갭 절반 닫힐 가능성.
3. **(c) baseline을 CDS-CZSL/Troika로 옮김** (3-5일): 두 repo 다 공개 + reproducible. mit-states/UT-Zap paper 숫자 재현 확인 후 v3_text 이식. 우리 framework 의존성 끊고 새 framework 위에서 v3_text 검증. 장점: 완전한 protocol/eval 일치 보장. 단점: 코드 이식 비용.
4. **(d) reality-check axis pivot** — (a)/(b)/(c) 다 막히거나, 재현 갭의 메커니즘 누락을 narrative로 정직하게 보고하는 방향. "CZSL 재현 가능성이 method 차이보다 큼"이라는 일반 메시지. mit-states neg + UT-Zap mild pos + ClusPro 코드/paper 불일치(§11-2 (3))까지 다 묶을 수 있음.
5. **(e) UT-Zap v3_text 진단 ablation** (1-2일): description ensemble의 +0.015가 진짜인지 random control / attr-only / obj-only로 분해. (a)와 병행 가능.

**추천 실행 순서:**
- **즉시**: (a) yml 3개 정렬해서 mit 1-seed probe (오버나잇 가능). 동시에 (e) UT-Zap random control 1-seed로 시작.
- **(a) 결과 보고**: 1.0 HM 이상 회복 → (b) OT 이식 시도. 회복 미미 → (c) framework 갈아타거나 (d) narrative pivot.

### 11-6. paperhp probe launch (5-11 17:37 KST)

§11-5 (a) 실행. yml `config/cluspro_baseline_mit_l14_paperhp_probe.yml` 신규:

| param | 기존 baseline yml | paperhp probe | paper §4.2 |
|---|---|---|---|
| weight_decay | 1e-5 | **5e-5** | 5e-5 ✓ |
| contrastive_weight (α) | 0.1 | **0.2** | 0.2 ✓ |
| hsic_weight (β) | 0.1 | **0.5** | 0.5 ✓ |
| 그 외 | — | 동일 | 동일 |

- launch: 5-11 17:37 KST, GPU 1, nohup, log `logs/train_cluspro_baseline_mit_paperhp_probe_20260511_173716.log`, pid 4093467
- 속도 첫 측정: ~2.75 it/s × 3793 step → epoch ~22min × 15 ep + 15 × eval = **~7-8h 예상, 5-12 새벽 1-2시 KST 종료**
- save_every_n=100 (epoch_*.pt 미저장, 디스크 압박 방지). val_best.pt + final_model.pt만 남음
- 결과 분석: `bash scripts/check_paperhp_probe.sh` → val_pairs running max + test_pairs (val_best.closed.json 생성됐다면)
- **판정 임계값 (사전 등록)**:
  - HM Δ ≥ +0.010 vs 기존 baseline 0.3893: yml이 의미 있는 기여 → (b) OT 이식 진행 정당화
  - HM Δ in [+0.005, +0.010]: 약한 시그널, OT 이식할지 borderline
  - HM Δ < +0.005: yml 차이는 noise. 갭의 정체는 구조적 메커니즘 누락 → (b) OT 이식 또는 (d) reality-check pivot 결정

### 11-7. paperhp probe 결과 (5-12 KST)

- 학습 완주: 5-11 17:37 → 5-11 23:40 KST (~6h, 15 epoch). 종료 시 GPU 1 idle 확인.
- 학습 끝났을 때 `val_best.closed.json`이 자동 저장되지 않음(train.py는 final evaluate를 호출만 하고 JSON dump 없음). `test.py --load_model val_best.pt`로 5-12에 재평가 → `checkpoint/cluspro_baseline_l14_mit_paperhp_probe/val_best.closed.json` 생성.
- **test_pairs (apples-to-apples):**

| metric | 기존 baseline (4-27) | paperhp probe | Δ |
|---|---|---|---|
| seen | 0.4899 | 0.4912 | +0.0013 |
| unseen | 0.5203 | 0.5228 | +0.0025 |
| **HM** | **0.3893** | **0.3853** | **−0.0040** |
| **AUC** | **0.2169** | **0.2173** | **+0.0004** |
| attr_acc | 0.3856 | 0.3893 | +0.0037 |
| obj_acc | 0.5553 | 0.5604 | +0.0051 |

- val 블록 참고: seen 0.5049 / unseen 0.5707 / HM 0.4185 / AUC 0.2485.
- **판정**: HM Δ=−0.0040, AUC Δ=+0.0004 → 사전 등록 임계 `HM Δ < +0.005`. **yml hyperparameter 차이는 noise**. 갭 0.018 (HM) / 0.021 (AUC)의 정체는 yml이 아니라 구조적 메커니즘 누락임이 1-seed로 확인됨.
- 1-seed 결과의 한계: HM ±0.005 시드 노이즈 범위 안에 들어가 있어 "차이 없음"의 negative 확신은 1-seed로 충분, "조금 더 좋다"의 positive 주장이 필요했다면 3-seed가 필요했음. 현재는 negative이므로 추가 시드 불필요.
- **다음 결정 (§11-5 분기 적용)**: 임계 `< +0.005` 분기 → (b) OT 이식 또는 (d) reality-check pivot. (b)가 단일 컴포넌트 기여가 paper §4.4에서 가장 컸으므로 우선. 즉, **cluspro_baseline.py에 OT (`local_assign` + `distributed_sinkhorn`) 이식**이 다음 작업.

### 11-8. framework pivot 결정 — Troika로 갈아탐 (5-12 KST)

§11-5 (b) vs (c) 사이에서 (c)로 결정. 근거:
- 현 framework의 ClusPro 재현 갭(HM 0.018, AUC 0.021)이 모든 ablation을 confound. "v3_text negative"가 idea 문제인지 framework 재현 갭 문제인지 분리 불가.
- 갭을 닫는 (b) OT 이식 시도해도 paper 수치 산수상 절반만 와서 여전히 재현 마무리 안 될 위험. 그 동안 우리 idea 검증은 계속 정체.
- 공식 코드 + 재현 가능한 baseline 위에서는 *baseline은 paper 수치* / *idea가 추가 ΔHM 주는가*가 깨끗하게 분리됨.

**Framework 후보 비교 (5-12 조사):**

1순위로 검토한 **CDS-CZSL (CVPR 2024)** — 공식 코드 **미공개** 확인:
- 논문 본문: "The supplementary provides codes with detailed parameters"만 언급, GitHub URL 없음
- CVPR open access supplementary: 404 (zip/PDF 다 없음)
- arXiv / Papers with Code / Hugging Face: 코드 링크 없음
- 1저자 Yun Li GitHub (`OgShun`): CDS-CZSL repo 없음
→ 직접 reimplement 비용이 큼, pivot 의미 사라짐.

| 후보 | venue | 별 | 업데이트 | v3_text 겹침 | 비고 |
|---|---|---|---|---|---|
| **Troika** ([bighuang624/Troika](https://github.com/bighuang624/Troika)) | CVPR 2024 | 30 | 26-04 | 적음 | multi-path branches, description ensemble 없음 — 클린 ablation 슬롯 |
| PLID ([Cogito2012/PLID](https://github.com/Cogito2012/PLID)) | ECCV 2024 | 15 | 25-10 | 많음 | language-informed distribution이 v3_text와 컨셉 인접 → idea 흡수 위험 |
| CSP ([BatsResearch/csp](https://github.com/BatsResearch/csp)) | NeurIPS 2022 | 95 | 26-04 | 적음 | 단순/안정, fallback. 단 한 세대 옛날 |

**결정: Troika.** 같은 CVPR 2024, 같은 SOTA 범위, description ensemble 모듈 없음 → v3_text 기여 깨끗히 측정 가능.

**계획:**
1. Troika clone + env 세팅 + dataset 경로 `DPAS/data/mit-states`, UT-Zap 연결
2. Vanilla Troika로 paper 수치 재현 (mit-states, seed=0, ViT-L/14) — paper Table 1과 ±0.005 안에 들어오면 OK
3. v3_text description ensemble 이식 (i: plain ensemble, CDS와 합치는 (ii)는 신호 보고 결정)
4. 3-seed mit-states + UT-Zap

이전 LHP-CZSL/ 디렉토리는 그대로 두고 새 작업은 `/home/student/dongki/Troika/`에. RESEARCH_LOG.md는 계속 LHP-CZSL/ 안에서 이어 적음 — 두 framework를 가로질러 한 연구 라인이므로.

### 11-9. Troika 셋업 + reproduction launch (5-12 KST)

repo `bighuang624/Troika` 클론 → `/home/student/dongki/Troika/`. lhp_czsl conda env 그대로 재활용 (torch 1.11.0+cu113 / clip / einops 등 호환 확인).

**고친 두 가지:**

1. **dataset.py 이미지 경로 패치** (LHP-CZSL과 동일): mit-states 메타데이터는 `coiled_brass` (underscore), 디스크 폴더는 `coiled brass` (space). `ImageLoader.__call__`에서 파일 없으면 첫 path segment의 `_`→space로 fallback. 안 넣으면 학습 시작 직후 `FileNotFoundError`.

2. **train.py CLI override 순서**: 원본은 `parse_args()` 다음에 `load_args(yml, config)`라 yml이 CLI를 덮어씀. 24GB GPU에서 yml default `train_batch_size: 64`가 강제돼서 OOM(21.4 GB 모델+활성화 → 메모리 부족). test.py 패턴을 따라 yml 로드 후 CLI flag (`--train_batch_size`, `--gradient_accumulation_steps`, etc.) 재적용. effective batch = 64 유지 (8 × grad_accum 8).

**메모리 진단** (해결 전):
- 모델 빌드 후: 1.83 GB
- batch=2 fwd peak: 9.78 GB
- batch=8 fwd+bwd peak: 14.63 GB ← 24GB에 맞음
- 그런데 train.py 실행 시 21.4 GB OOM ← yml override가 batch=64 강제한 게 원인
- Troika 저자 issue #2/#5: 원래 A100(80GB) 사용. 24GB에선 batch 작게 + grad_accum 필요.

**Reproduction run 시작:**
- 5-12 12:58 KST launch, GPU 1, nohup, pid 460040
- log: `Troika/logs/train_mit_seed0_repro_20260512_125855.log`
- 명령: `train.py --yml_path config/troika/mit-states.yml --train_batch_size 8 --gradient_accumulation_steps 8 --seed 0` (epochs=10, val_metric=best_loss)
- 첫 step 측정: ~1.73 it/s × 3793 step/epoch × 10 epoch = ~6.1h 학습 + 10 × eval (~40min) = **~7h ETA, 5-12 20:00 KST 종료 예상**
- 합격선: Troika paper Table 2 mit-states CW (ViT-L/14) HM ≈ 39.2 / AUC ≈ 20.1 — **재현치가 ±0.005 이내**면 baseline 인정, 그 위에서 v3_text 이식 진행
- 재현 미달 (예: HM <38) → Troika도 같은 재현 갭에 빠진 거. fallback으로 PLID 또는 CSP 시도

### 11-10. Troika train.py 버그 + fix (5-12 14:00 KST)

**버그**: epoch 1 train (~35min) + val eval (~4min) 끝난 직후 `KeyError: 'best_loss'`로 크래시 — `val_best.pt` 저장도 못 함, 빈 save_path. 1 epoch 손실.

원인: `mit-states.yml`은 `val_metric: best_loss`인데 train.py의 `evaluate()` 반환 dict 키는 `'loss'`만 있고 `'best_loss'`가 없음. train.py:85의 `val_result[config.val_metric]` 접근에서 KeyError.

저자가 issue #3에서 의도 확인: mit-states는 의도적으로 best_loss 기준 선택 (다른 데이터셋은 AUC). 즉 명백한 코드/yml 키 불일치 버그. 다른 repo 사용자들이 안 보고한 건 yml의 val_metric을 안 건드리고 학습 시 CLI로 override했거나 데이터셋별 yml을 안 건드린 듯.

**Fix** (1 line, train.py:130):
```python
test_saved_results['best_loss'] = loss_avg  # for yml val_metric: best_loss
```

**Epoch 1 eval 결과 (죽기 직전 출력)**: best_seen 0.4181 / best_unseen 0.5584 / best_hm **0.3729** / AUC **0.2014** / attr 0.3825 / obj 0.5843. 1 epoch만 돌고도 paper 수치 (HM ~39, AUC ~20) 범위 진입 — 10 epoch 완주 시 재현 가능성 매우 높음.

재시작: 14:02 KST, pid 482043, log `train_mit_seed0_repro_20260512_140226.log`. 새 ETA 5-12 20:27 KST.



### 11-11. Troika reproduction 결과 (5-12 KST)

- 학습 완주: 5-12 14:02 → 5-12 20:34 KST (~6.5h, 10 epoch). 종료 시 GPU 1 idle 확인.
- `val_best.pt` 자동 저장 (val_metric=best_loss, train.py:85-89 + §11-10 fix). 학습 끝에 train.py:99-102가 test set에 자동 evaluate 실행, 결과는 train log 끝부분에 출력.
- **mit-states test_pairs (CW, ViT-L/14, seed 0):**

| metric | Troika paper Table 2 | 재현 | Δ |
|---|---|---|---|
| HM | 0.392 | **0.3940** | +0.002 |
| AUC | 0.201 | **0.2177** | +0.017 |
| seen | (~52) | 0.4676 | — |
| unseen | (~50) | 0.5387 | — |
| attr_acc | — | 0.3952 | — |
| obj_acc | — | 0.5640 | — |

- §11-9 합격선 (`paper 수치 ±0.005`): HM은 그 안, AUC는 오히려 +0.017 상회. **Troika baseline 재현 OK** → §11-8 plan대로 위에 v3_text 이식 진행.
- per-epoch val 추세: 학습 loss는 ep10에 0.033까지 떨어지지만 val_pairs HM은 ep2-4 (HM 0.41~0.42, AUC 0.245~0.247) 고점 후 점진 하락. val_best가 ep2 부근일 가능성 높음(loss로 선택). test_pairs (val_best) HM 0.394 / AUC 0.218 → val 고점보다 test에서 약간 낮은 것은 데이터셋 split 차이로 정상.
- 비교: 우리 LHP-CZSL framework에서 같은 backbone+seed의 cluspro_baseline mit-states test HM 0.3893 / AUC 0.2169 → Troika가 HM +0.0047, AUC +0.0008. paper에서도 Troika가 ClusPro보다 약간 위(약 +1 HM)인데 우리 재현도 같은 방향. **이전 framework의 재현 갭이 사라진 것은 아니지만, Troika 자체가 paper와 ±0.005 안이므로 ablation 기반선으로 손색없음.**
- 결정: §11-8 (c) 완료 (vanilla 재현). 다음은 v3_text (description ensemble) 이식 — Troika의 `construct_token_tensors` + CMT 경로에 sub-meaning embedding 주입.

### 11-12. Troika + v3_text 이식 + launch (5-12 23:48 KST)

**이식 내용** (`Troika/code/model/troika_v3text.py` 신규 + `model_factory.py`에 `troika_v3text` 분기 등록):

- `sub_meanings_path` (`data/descriptions_mit.json`) 로드 → primitive마다 K개 sub-meaning (mit: 거의 전부 K=3).
- `soft_att_obj`을 sub-level로 확장: shape `[sum_Ka + sum_Ko, D] = [345+735, 768] = [1080, 768]` (vanilla는 `[115+245, 768] = [360, 768]`). Init = sub-meaning 문자열의 CLIP token embedding mean.
- `construct_token_tensors`:
  - **comp branch**: sub-embedding을 primitive-level로 mean-pool 후 attr_idx/obj_idx로 indexing (LHP-CZSL과 동일). 출력 shape는 baseline과 같음 (`[B_pair, ctx, D]`).
  - **attr branch**: sub-level 프롬프트 그대로 사용. 출력 shape `[sum_Ka, ctx, D]`.
  - **obj branch**: 동일. 출력 shape `[sum_Ko, ctx, D]`.
- `forward` / `forward_for_open`: `encode_text` 직후 sub-level text feature를 primitive-level로 mean-pool. **CMT 이후 logit 계산까지 shape이 vanilla Troika와 동일** → loss_calu, logit_infer는 baseline과 같은 코드 path.
- `patch_norm` 적용 위치는 vanilla 그대로 (`i_element` loop 안에서 매번 norm). 즉 v3_text 이식은 텍스트 임베딩 단계만 변형, CMT/visual/loss 경로는 byte-identical.

**Sanity 통과** (5-12 23:47):
- 모델 빌드 OK, sum_Ka=345, sum_Ko=735, soft_att_obj=[1080, 768].
- batch=4 단발 forward+loss OK (loss=20.09, peak mem 11.16 GB). batch=8 + grad_accum=8 setting에서도 24GB 안에 들어올 것으로 기대.

**Launch** (5-12 23:48 KST):
- yml `config/troika/mit-states-v3text.yml` 신규. 나머지 hp는 baseline mit-states.yml과 동일 (lr=1e-4, wd=1e-5, attr_dropout=0.3, epochs=10, val_metric=best_loss, batch=8, grad_accum=8).
- log: `Troika/logs/train_mit_v3text_seed0_20260512_234836.log`, pid 743965, GPU 1.
- save: `Troika/save/mit-states_seed0_v3text/`.
- 첫 step 측정: 1.54 it/s × 3793 step/epoch × 10 ep = ~6.85h 학습 + ~50min eval ≈ **~7.7h, ETA 5-13 07:30 KST**.

**판정 임계값 (사전 등록, 1-seed):**
- HM Δ ≥ +0.010 vs Troika baseline 0.3940 → v3_text 효과 의미 있음 → 3-seed 확장 + UT-Zap 이식
- HM Δ in [+0.005, +0.010] → 약한 신호. 3-seed 검증 권장.
- HM Δ in [−0.005, +0.005] → noise. v3_text 자체로는 Troika에서 효과 없음 결론. CDS와 결합한 (ii) image-conditional selection 시도 또는 [[project_lhp_czsl_text_enrich]] 메모에 정리된 다음 axis로 pivot.
- HM Δ < −0.005 → mit-states에서 negative 재현 (LHP-CZSL framework와 같은 방향). Troika에서도 description ensemble이 unseen에 해롭다는 것이 확인됨 → narrative pivot 검토.

### 11-13. Troika + v3_text 결과 + v4_imgsel 이식 launch (5-13 KST)

**v3_text mit-states test_pairs (5-13 07:37 완료, seed 0):**

| metric | Troika baseline | v3_text | Δ |
|---|---|---|---|
| HM | 0.3940 | **0.3314** | **−0.0626** |
| AUC | 0.2177 | **0.1641** | **−0.0536** |
| seen | 0.4676 | 0.4366 | −0.0310 |
| unseen | 0.5387 | 0.4563 | **−0.0824** |
| attr_acc | 0.3952 | 0.3377 | −0.0575 |
| obj_acc | 0.5640 | 0.5206 | −0.0434 |

- §11-12 사전 등록 4번째 case (HM Δ < −0.005) 발생. **mit-states에서 v3_text negative 재현** — Troika 위에서도 LHP-CZSL framework와 같은 방향. unseen이 −0.082로 가장 크게 빠진 패턴 일치.
- 결정: narrative pivot 가기 전에, **v3_text의 핵심 약점인 "mean-pool pre-CMT collapse"가 진짜 원인인지** 확인하는 1-축 ablation 먼저. → v4_imgsel 이식.

**v4_imgsel 설계 (`Troika/code/model/troika_v4imgsel.py` 신규 + model_factory 분기):**

- **변경 축**: v3_text가 attr/obj branch에서 sub-level text feature를 encode 후 mean-pool하던 단계를 **image-conditional softmax 가중합**으로 교체.
  - sub_feat `[sum_K, D]` → primitive 그룹별 normalize → image normalized feature를 query로 cosine score → softmax(τ=0.5) → weighted sum → `[B, num_prim, D]` (image-conditional primitive feature, L2-norm).
  - CMT 이후 logit 계산: `bd, bpd -> bp` einsum.
- **comp branch는 v3_text와 byte-identical** (mean-pool pre-encoder 그대로). → v3_text vs v4_imgsel은 attr/obj aggregation 방식 1축만 다름.
- temperature τ=0.5 fixed config hp. 후속에서 learnable 확장 가능.
- 패딩-mask buffer (`attr_subs_idx/_mask`, `obj_subs_idx/_mask`) 유지해서 K 가변 데이터셋도 동일 코드 path.

**Sanity (5-13 13:30):**
- batch=4 forward+loss+backward OK, peak mem 13.32 GiB. batch=8 16.28 GiB. 24 GiB 안.
- sum_Ka=345, sum_Ko=735, K_max=3 (mit-states 균일). attr_subs_mask sum=345, obj_subs_mask sum=735 → padding 없음 확인.
- comp [4,1262] / attr [4,115] / obj [4,245] shape OK, no NaN, loss=21.6.

**Launch (5-13 13:35 KST):**
- yml `config/troika/mit-states-v4imgsel.yml` 신규. 나머지 hp는 v3_text와 동일 (lr=1e-4, wd=1e-5, attr_dropout=0.3, epochs=10, val_metric=best_loss, batch=8, grad_accum=8).
- log: `Troika/logs/train_mit_v4imgsel_seed0_20260513_133505.log`, pid 895986, GPU 1.
- save: `Troika/save/mit-states_seed0_v4imgsel/`.
- 첫 step 측정: 1.53 it/s × 3793 step/epoch × 10 ep + eval ≈ **~7.7h, ETA 5-13 21:15 KST**.

**판정 임계값 (사전 등록, 1-seed, vs Troika baseline 0.3940 / v3_text 0.3314):**
- HM ≥ Troika baseline (≥0.3940) → image-conditional selection이 mean-pool 문제를 완전히 해결. CDS 결합 등 다음 axis로.
- HM in (v3_text 0.3314, baseline 0.3940) → mean-pool이 부분적 원인. v3_text보다 개선됐지만 baseline 못 미침 → temperature/comp branch 확장 등 후속.
- HM ≤ v3_text (≤0.3314) → mean-pool은 원인이 아님. mit-states 자체가 LLM-text augmentation에 적대적 → narrative pivot 근거 확보.

**왜 mit-states부터?** §11-4에서 mit-states는 v3_text negative였고 UT-Zap은 mild positive였음. Troika 위에서도 같은 데이터셋 의존성 패턴이 재현되는지가 (e) 진단의 핵심. mit-states에서 +가 나오면 framework 종속성, − 또는 tie면 데이터셋 종속성 → 두 framework가 일관됨.

### 11-14. Troika + v4_imgsel 결과 — mean-pool 무죄, narrative pivot 근거 확보 (5-13 23:43 KST)

**v4_imgsel mit-states test_pairs (seed 0):**

| metric | Troika baseline | v3_text | **v4_imgsel** | Δ(v4 vs base) | Δ(v4 vs v3_text) |
|---|---|---|---|---|---|
| HM | 0.3940 | 0.3314 | **0.3354** | **−0.0586** | +0.0040 |
| AUC | 0.2177 | 0.1641 | **0.1669** | **−0.0508** | +0.0028 |
| seen | 0.4676 | 0.4366 | 0.4395 | −0.0281 | +0.0029 |
| unseen | 0.5387 | 0.4563 | 0.4582 | **−0.0805** | +0.0019 |
| attr_acc | 0.3952 | 0.3377 | 0.3377 | −0.0575 | **±0.0000** |
| obj_acc | 0.5640 | 0.5206 | 0.5243 | −0.0397 | +0.0037 |

- val_pairs (참고): HM 0.3795, AUC 0.2055, best_unseen 0.5089, best_seen 0.4691 (val_best.closed.json `val` block).
- 실행: `test.py` on `save/mit-states_seed0_v4imgsel/val_best.pt`, `--open_world False`, log `Troika/logs/test_mit_v4imgsel_valbest_20260513_233328.log`. 첫 실행은 `--dataset_path` 누락으로 즉시 fail → `--dataset_path /home/student/dongki/DPAS/data/mit-states` 추가 후 재실행 OK.
- 학습은 epochs=10까지 정상 완료 (train loss 1.54 → 0.038, e1 → e10). val_best.pt는 21:47 final 저장 전 시점에서 ep 0~3 사이 마지막 갱신 — 일찍 peak 후 overfit 표시 (§11-9 Troika baseline과 동일 패턴).

**§11-13 사전 등록 case 3 (HM ≤ v3_text) 적중:**
- v4_imgsel HM 0.3354 vs v3_text HM 0.3314: Δ +0.004 = noise. **image-conditional softmax selection이 mean-pool을 대체했음에도 의미 있는 회복 없음**.
- attr_acc는 정확히 동일 (0.3377). obj_acc도 ±0.004. → image-conditional aggregation이 attr/obj 인식 자체를 거의 못 바꿈.
- unseen drop −0.080 패턴이 v3_text(−0.082)와 사실상 동일. → seen-prior 강화로 unseen이 깎이는 패턴은 aggregation 방식과 무관.

**결론: mean-pool은 v3_text 실패의 원인이 아니다.** v3_text/v4_imgsel은 같은 description content를 다른 aggregation으로 주입했는데 결과가 동일 → 변인은 **(content) descriptions_mit.json 품질** 또는 **(dataset) mit-states 자체가 LLM-text augmentation에 적대적**.

**다음 결정 분기:**
- (a) descriptions를 다른 LLM(Claude/GPT-4)으로 재생성 후 v3_text 재실험 (~3h LLM API + ~8h Troika 학습) — content 변인 마지막 확인.
- (b) UT-Zap에 v4_imgsel 재실행 (~8h) — dataset-dependency 봉인 (mit-states − / UT-Zap + 패턴이 framework 무관하게 재현되는지).
- (c) axis G pivot (compute 0) — 5개 LLM 진입점 모두 실패 evidence 충분, narrative 작성.

**즉시 액션 (5-13 23:50)**: sanity check — `data/descriptions_mit.json`에서 primitive description 표본 검토하여 hallucination/평이성 여부 판단. content 품질이 명백히 낮으면 (a) 우선, OK면 (b) 또는 (c).

### 11-15. Sanity check → Claude descriptions regenerate → v3_text-claude launch (5-13 23:50 → 5-14 00:35 KST)

**Sanity check 결과 (descriptions_mit.json, Gemini 2.5 Flash Lite 생성):**

구조적으로는 OK (K=3 균일, 11-12 단어 평균, CLIP 77-token 여유). 그러나 결정적 content 문제 발견 — **ATTR descriptions가 사실상 (attr × object) 페어를 묘사함**:

- `"wide"` → "wide river / wood slab / highway" (3 host object 등장)
- `"broken"` → "shattered mug / cracked phone screen / torn backpack"
- `"painted"` → "acrylic on table / paint on door / paint on car fender"

대칭적으로 OBJ descriptions에 attr word 누설:
- `"cotton"` → "fluffy white cotton balls" (fluffy/white = attr)
- `"shorts"` → "frayed, worn casually" (frayed/worn = attr)
- `"wall"` → "weathered, rough surfaces"

**메커니즘**: `_construct_soft_prompt`에서 sub-meaning string의 CLIP token embedding을 mean pool → `soft_att_obj` parameter의 **초기값**. 학습 가능하지만 frozen CLIP + 10-epoch 학습으로 init pollution을 못 씻어냄. → attribute prototype이 seen pair의 host object feature로 polluted → unseen pair (attr, new_obj)에서 wrong object prior 주입 → unseen drop −0.08 패턴이 v3_text/v4_imgsel 동일.

**원인 = tools/llm_descriptions.py:54-65 `ATTR_PROMPT_TEMPLATE` 디자인**: "leather has smooth/grained/patent" 예제 + "concrete and visual" system prompt가 LLM을 항상 host object 호출로 유도. "wide", "tiny", "broken" 같은 추상 attribute는 자연어로는 object 없이 묘사 거의 불가능.

**Claude regeneration (5-14 00:35):**
- Opus 4.7로 conversation에서 직접 생성 (API 미사용, 비용 0).
- 새 prompt 디자인: attr → abstract carrier만 사용 ("a surface", "a form", "a region", "a feature"), host object naming 금지, 다른 attr word 금지, focus on (texture, edge, color, light, pattern, scale).
- obj → canonical/neutral form, attribute/state word 금지, focus on (shape, parts, posture, material category, composition).
- 1080 descriptions 자동 validator로 leakage 0건 확인 (`tools/build_descriptions_claude.py`).
- 출력: `data/descriptions_mit_claude.json` (109KB, LHP-CZSL + Troika 양쪽).

**예시 비교 (wide):**
- Gemini old: "A very wide, flat river with calm water, reflecting the sky."
- Claude new: "a horizontally-extended planar surface dominating the frame with low aspect ratio"

**Launch (5-14 00:35 KST):**
- yml `config/troika/mit-states-v3text-claude.yml` (v3text와 동일 hp, `sub_meanings_path: data/descriptions_mit_claude.json`만 차이).
- exp `mit-states_seed0_v3text_claude`, GPU 1, fp16, batch=8/grad_accum=8.
- 첫 step 예상 1.5 it/s × 3793 step/epoch × 10 ep + eval ≈ **~7.7h, ETA 5-14 08:15 KST**.

**판정 임계값 (사전 등록, 1-seed, vs Troika baseline 0.3940 / v3_text old 0.3314 / v4_imgsel 0.3354):**
- HM ≥ 0.3940 → content가 변인, axis 부활. CDS combo / 3-seed 확장.
- HM in [0.3700, 0.3940) → content가 주요 lever지만 완전 회복 못함. K/temperature/loss 등 후속 tuning.
- HM in [0.3500, 0.3700) → content가 부분적 lever. axis G pivot 가되 mixed-evidence 정리 필요.
- HM < 0.3500 → content도 lever 아님. 5개 LLM 진입점 모두 negative 봉인 → axis G pivot 확정, narrative 작성.

### 11-16. v3_text-claude mit-states 결과 — case 4 적중, 5개 LLM 진입점 모두 negative (5-14 08:29 학습 종료, 13:38 test eval)

**학습 완주**: 5-14 00:35 → 5-14 08:29 KST (~7.9h, 10 epoch). val_best.pt 저장 시점 02:11 (early epoch).

**test_pairs (apples-to-apples, `save/.../val_best.closed.json`의 `test` 블록):**

| metric | Troika baseline | v3_text (Gemini) | v4_imgsel | **v3_text-claude (leak 0)** | Δ vs baseline | Δ vs v3_text-old |
|---|---|---|---|---|---|---|
| HM | 0.3940 | 0.3314 | 0.3354 | **0.3242** | **−0.0698** | **−0.0072** |
| AUC | 0.2177 | 0.1641 | 0.1669 | **0.1613** | **−0.0564** | **−0.0028** |
| seen | 0.4676 | 0.4366 | 0.4395 | 0.4387 | −0.0289 | +0.0021 |
| unseen | 0.5387 | 0.4563 | 0.4582 | 0.4517 | −0.0870 | −0.0046 |
| attr_acc | 0.3952 | 0.3377 | 0.3377 | 0.3474 | −0.0478 | +0.0097 |
| obj_acc | 0.5640 | 0.5206 | 0.5243 | 0.5199 | −0.0441 | −0.0007 |

- val 블록 (참고): HM 0.3664, AUC 0.1967, best_seen 0.4626, best_unseen 0.5055.
- val_pairs running max best_hm (train log epoch-by-epoch 출력): 0.3826 — 이건 §11-1 convention상 비교 불가 숫자. 5-14 오전 사용자에게 일차 보고 시 이 숫자를 잘못 인용했음(§10-5.1 split mismatch 실수의 재발).

**§11-15 사전 등록 case 4 적중 (HM < 0.3500)**:
- Claude leak-free desc가 Gemini old desc와 사실상 tie (HM Δ −0.007, AUC Δ −0.003 — 시드 노이즈 ±0.005 범위 내).
- **content leakage(attr description의 host object 누설)는 v3_text negative의 원인이 아님.** content 자체가 lever가 아님.
- mit-states에서 5개 LLM 진입점 모두 negative 봉인 확정: v1, v1_init_only, v2, v3_text(Gemini), v3_text(Claude leak-free), v4_imgsel.

### 11-17. UT-Zap chain launch — Troika baseline + v3_text-claude (5-14 14:15 KST)

**동기 (페이지 9 면담 결과 재해석)**: 5-14 점검에서 면담 자료 page 9의 "v3_text +1.5pp HM (UT-Zap, 1.3σ mild positive)"이 사실 **mit-states v3_text와 다른 mechanism**임을 발견. 비교:

| dataset | sub_meanings_path | content 형식 | K 분포 |
|---|---|---|---|
| mit-states | `descriptions_mit.json` (Gemini) | LLM long visual desc | K=3 균일 |
| UT-Zap (LHP-CZSL §1.1) | `sub_meanings_utzap.json` | short categorical sub-name (사람 만든 듯) | **K=1 대부분, K=2 일부** |

UT-Zap "v3_text"는 사실상 v1_init_only와 거의 같음 (mostly K=1, K=2짜리 몇 개만 sub-name ensemble). v1_init_only HM 0.5458 → v3_text HM 0.5523 = **+0.65pp 추가는 'K=2 sub-name 추가' 효과지 'LLM long description' 효과가 아님**. 따라서 사용자 가설("LLM이 데이터셋 의존적")은 면담 evidence로는 검증되지 않은 상태였음.

**가설 직접 검증 실험**: mit-states에서 망한 그 mechanism (LLM long description ensemble)을 UT-Zap에서 똑같이 돌려서 dataset dependency가 진짜인지 확인. 또한 Troika UT-Zap baseline 자체도 미재현 상태라 둘을 chain.

**UT-Zap용 Claude leak-free descriptions 생성 (5-14 14:08~14:14)**:
- `tools/build_descriptions_utzap_claude.py` 신규. 16 attr (material) + 12 obj (shoe-type) × K=3 = **84 sub-prompt**.
- ATTR rule: shoe-type tokens 금지, 다른 material tokens 금지, "a textile/coating/finish" 같은 abstract carrier 사용. OBJ rule: material tokens 금지, silhouette/shaft/sole 등 form 단어만.
- multi-token primitive name 처리 위해 per-primitive forbidden set validator (`tokens_of(name) - GENERIC_NAME_TOKENS`). 첫 패스 5개 leak (`grain`, `synthetic`, `mid` 등) 발견 후 수동 수정 → leakage 0.
- 출력: `data/descriptions_utzap_claude.json` (10KB), Troika 쪽으로도 복사.

**Chain script** (`Troika/scripts/run_utzap_chain.sh`): GPU 1에서 baseline → v3_text-claude 순차 실행, 각 학습 후 `test.py --load_model val_best.pt`까지 자동 호출 → `val_best.closed.json` 두 개 생성. batch=8 + grad_accum=16 (eff 128, paper와 동일), 15 epoch, val_metric=AUC.

**Launch (5-14 14:15 KST)**:
- baseline: `config/troika/ut-zappos.yml` (model_name=troika), pid=2708413, log `train_utzap_baseline_seed0_20260514_141547.log`
- v3text-claude: `config/troika/ut-zappos-v3text-claude.yml` 신규 (sub_meanings_path만 차이), 동일 model_name=troika_v3text
- 첫 step 측정: 3.4 it/s × 2875 step/epoch × 15 ep ≈ 3.5h 학습 + eval ≈ **~4h per run**. **Chain 종료 ETA: 5-14 22:00~23:00 KST**

**판정 임계값 (사전 등록, 1-seed, baseline은 우리 재현치 기준):**
- baseline 자체 sanity: paper Troika UT-Zap CW HM ≈ 0.541 ± 0.005 안에 들어오면 OK (UT-Zap에서 LHP-CZSL framework는 −4.8 HM 갭이었음, Troika가 더 가까운지 확인)
- v3_text-claude vs Troika baseline:
  - **HM Δ ≥ +0.010**: LLM long desc가 UT-Zap에서도 양의 효과 → 사용자 가설 ✓ (LLM dataset-dependent), thesis narrative "LLM-text effectiveness is dataset-granularity dependent" 가능
  - **HM Δ in [−0.005, +0.010]**: tie. framework 종속성 가능. 추가 seed 또는 mit-states/UT-Zap 차이 mechanism 분석 필요
  - **HM Δ < −0.005**: UT-Zap에서도 negative → 면담 page 9의 +1.5pp가 LHP-CZSL reproduction gap이 만든 noise. axis 완전 봉인, narrative pivot 굳힘

### 11-18. UT-Zap chain 결과 — v3_text-claude UT-Zap에서도 음, dataset-dependency 가설 기각 (5-14 21:27 KST)

학습 시간: baseline 14:15→17:57 (3h 42m, val_best.pt + final test eval 포함), v3_text-claude 17:58→21:26 (3h 28m). Chain 종료 21:27.

수치 출처: `Troika/logs/train_utzap_{baseline,v3text_claude}_seed0_*.log` 마지막 줄 (Troika test.py가 마지막에 `evaluating on the test set` 블록을 자동 실행).

| Run | seen | unseen | **HM** | **AUC** | attr | obj |
|---|---|---|---|---|---|---|
| Troika baseline | 0.6559 | 0.6959 | **0.5139** | **0.3782** | 0.4859 | 0.7306 |
| Troika + v3_text-claude | 0.6442 | 0.6526 | **0.4612** | **0.3252** | 0.4571 | 0.6905 |
| **Δ vs baseline** | **−0.0117** | **−0.0433** | **−0.0527** | **−0.0530** | −0.0288 | −0.0401 |
| 논문 Troika UT-Zap CW | 0.667 | 0.737 | **0.541** | **0.418** | — | — |

**판정 vs §11-17 사전 등록 임계값:**
- baseline sanity: 우리 0.5139 vs paper 0.541 ± 0.005 → **paper 구간 못 들어옴, −2.7pp 갭**. Troika UT-Zap 재현도 갭 존재. 다만 LHP-CZSL framework UT-Zap 갭(−4.8 HM)보다는 가까움 → Troika가 paper-faithful 정도에서 framework로서 더 가까움 재확인.
- v3_text-claude Δ HM = **−0.0527** → "HM Δ < −0.005" bucket 적중 → **사전 등록 판정: UT-Zap에서도 negative**.

**핵심 의미 (dataset-dependency 가설 기각):**
1. mit-states에서 망한 LLM long visual description axis가 **UT-Zap에서도 망한다**. v3_text "axis" 자체가 dataset 종속이 아니라 framework/방향성 자체가 잘못된 쪽으로 결론.
2. 면담 page 9의 "UT-Zap v3_text +1.5pp HM mild positive"는 §11-17 진단대로 **LHP-CZSL §1.1에서 short categorical sub-name(`sub_meanings_utzap.json`, K=1~2 위주)을 v3_text 이름으로 돌린 효과**였음. 같은 axis(LLM long description, K=3)를 Troika framework + Claude leak-free descriptions로 통제했을 때 +1.5pp 신호 사라지고 −5.3pp 음.
3. v3_text가 unseen pair를 더 크게 떨어뜨리는 패턴이 일관됨: mit-states unseen Δ ≈ −0.048, UT-Zap unseen Δ = −0.043. **text-side description ensemble이 systematically unseen generalization 저하** → 매커니즘은 "description token이 seen 분포에 더 강하게 fit해서 unseen 일반화 손해" 가설이 가장 자연스러움.
4. **v3_text axis 완전 봉인**. [[project_lhp_czsl_text_enrich]] 메모의 "image-conditional selection 결합" 후속 아이디어도 v3_text base가 무용해진 이상 같은 framework 위에서는 진행 안 함.

**narrative 함의:**
- "LLM이 dataset-granularity에 따라 효과 달라진다" 류 thesis로는 가지 못함. 면담 +1.5pp 신호의 출처가 framework noise(LHP-CZSL ClusPro 재현 갭) + sub-name 형식(categorical short name) 두 가지 혼합이었다는 게 §11-17/18로 드러남.
- 새 narrative 후보 정리 필요 (다음 advisor 미팅 자료). 현재 시점에서 살아있는 유일한 양의 신호는 [[project_lhp_czsl_direction]]에서 mit-states v1/v1_init_only가 baseline tie 유지(망하진 않음)라는 것뿐 — "방법론은 baseline 동치, 다만 unseen 일반화를 해치지 않는다"는 weak claim. 거기서 더 가져갈 방향은 본 결과로는 보이지 않음.

---

## 12. mit-states baseline post-fix 3-seed 결과 — 5-16 (학습 5-15 03:55~22:34 KST)

### 12-1. 결과 수치 (test_pairs, val_best.pt, train.py 자동 final-epoch eval)

수치 출처: 각 seed의 `logs/train_baseline_mit_v2_seed{0,1,2}_20260515_035508.log` 마지막 줄. train.py:111-115에서 `i+1 == config.epochs`에 val_best.pt를 load해 test_dataset으로 closed-world 평가 → 정확히 §1 표의 `test_pairs` 정본.

| seed | seen | unseen | HM | AUC | attr | obj |
|---|---|---|---|---|---|---|
| 0 | 0.4933 | 0.5226 | 0.3879 | 0.2176 | 0.3842 | 0.5590 |
| 1 | 0.4857 | 0.5276 | 0.3889 | 0.2172 | 0.3858 | 0.5561 |
| 2 | 0.4874 | 0.5220 | 0.3894 | 0.2165 | 0.3852 | 0.5546 |
| **mean ± std** | **0.4888 ± 0.0040** | **0.5241 ± 0.0031** | **0.3887 ± 0.00076** | **0.2171 ± 0.00056** | 0.3851 | 0.5566 |

학습 시간: seed 0 = 5h 41m, seed 1 = 5h 43m, seed 2 = 7h 15m (seed 2만 GPU contention 영향 가능, 결과는 정상 범위).

### 12-2. 의미

1. **AMP NaN 오염 우려 해소 (mit-states 한정)**: 5-2 fix 적용 후의 baseline mean이 4-27 pre-fix single-seed와 시드 std 안에서 동률. 즉 mit-states/ViT-L/14에서는 NaN으로 인한 silent train-step skip이 metric에 측정 가능한 영향을 안 남겼다. UT-Zap에서 봤던 HM +0.150 같은 fix 효과는 mit-states에서는 안 나타남 — UT-Zap의 epoch당 step 수가 mit-states보다 훨씬 많아 NaN 발생 빈도/누적 영향 차이가 컸을 것으로 추정. ⚠ **UT-Zap에는 여전히 fix가 필요**.

2. **시드 노이즈가 종전 추정의 1/6 수준**: 4-29 시점에 K=3/K=5/LLM-K 차이를 "ΔHM 0.004는 시드 노이즈 안"이라 판정했는데, 진짜 시드 std는 0.00076 → 0.004는 사실 **5σ**. 다만 1-seed 비교라 noise floor 추정 자체가 부정확했음 → 결론 자체는 (변경 아래) 그대로 유지하되 "tie 판정 기준은 1-seed std가 아니라 3-seed std로 봐야 한다" 메모.

3. **v3_text의 음의 격차가 진짜로 큼**: HM −0.0271 / std 0.00076 = **35σ**. AUC −0.0244 / std 0.00056 = **44σ**. 무시할 수 없는 시스템적 회귀.

4. **paperhp probe(5-11)와 baseline 비교 재해석**: 5-11 표에서 paperhp(0.3853) vs baseline(0.3893)을 "HM −0.004, 거의 동률"로 정리했는데, 시드 std 0.00076 기준 약 **5σ 음의 격차**. paperhp는 사실상 회귀. 폐기 판단 자체는 맞았지만 "거의 동률" 어휘는 과소표현이었음.

5. **논문 재현 갭이 진짜 갭**: 0.4888 → 논문 0.521 (seen Δ -3.2pp), 0.3887 → 0.407 (HM Δ -1.8pp). 시드 std로는 도저히 못 닫는 거리. §11-2 진단 항목(λ_h 누락 + initialization 차이 + dropout 등)이 여전히 open question. 단, 우리는 5-12 이미 Troika 프레임워크로 갈아탄 상태라 mit-states ClusPro 재현 갭 자체를 닫는 것은 더 이상 priority 아님 — narrative상 "framework noise floor가 1.8pp 있다"는 fact로만 유지.

### 12-3. 다음 단계

- **§11-17 UT-Zap chain (5-14 launch) 결과** 점검 우선순위가 더 높음 — 그쪽은 thesis narrative 판정용 (v3_text dataset-dependent 가설 검증).
- 본 5-15 baseline은 **mit-states 비교 기준선으로 사용**할 수 있는 신뢰 가능한 3-seed 값으로 §1 표에 안착. 향후 mit-states에서 새 axis를 시도하면 이 3-seed 분포와 직접 비교.
- 단일 axis로 진행 시: 최소 3-seed 필수. 1-seed Δ 0.005 미만은 모두 noise 영역으로 보면 안 됨 (실제 noise는 0.0008).

---

## 13. R2D2 axis 진입 — Region-Routed Description Distillation (5-16 KST)

### 13-1. 진입 결정 배경

§11-18까지 LLM-text 계열 6연패 (mit-states 5번 + UT-Zap 1번) 확정. 공통 mechanism: description을 text encoder로 통과시켜 K=3 ensemble → image와 cosine matching → **inference에서 description 정보가 logit 경로에 직접 들어가서 seen 분포에 overfit, unseen 떨어짐**. 두 dataset에서 unseen Δ가 −0.043 ~ −0.048로 일관 → 매커니즘.

사용자(2026-05-16) 결정: axis를 봉인하지 말고 **"description 정보는 살리되 vision-side 구조를 새로 설계해서 그 정보를 흡수시키는" 방향으로 pivot**. 단순 add-on 아니라 ground-up architecture. SOTA 야망.

### 13-2. Design principle (mechanism level)

1. **Description은 vision representation에 distill하고, inference에서 logit 경로에 두지 않는다** → 6연패의 seen-overfit mechanism을 차단.
2. **CLIP ViT patch tokens(N≈257)를 활용한다**. Troika/ClusPro 모두 CLS feature(`img_feature[:, 0, :]`) 1개만 쓰고 patch grid 256개는 버림. Troika code의 `encode_image_with_adapter`는 이미 `(CLS, full_patch_feat)` 둘 다 반환 — 통합 지점 free.
3. **LLM description의 spatial 정보 ("glossy droplet on red surface" 등)를 patch-level binding의 ground truth 후보로 활용**.

### 13-3. Architecture spec (R2D2 v0.1)

#### Components

**(A) Patch aggregator** — `model/troika_r2d2.py`
- Input: full patch feature `(B, N, D)` = 1 CLS + 256 patches, N=257, D=768 (ViT-L/14 projection 후)
- Learnable query token `q ∈ ℝ^D` + multi-head cross-attention (queries=q, keys/values=patch tokens)
- Output: aggregated image feature `(B, D)` — Troika baseline의 `img_feature[:, 0, :]` 대체

**(B) Description bank** — offline precompute
- Source: `data/descriptions_mit_claude.json` (28,175 pairs × K=3, leak-free)
- 각 description을 CLIP text encoder의 `encode_text(..., return_all_tokens=True)`로 token-level embedding 추출 → `(num_pairs, K=3, L=77, D=768)` cache
- 또는 EOT projection 후 single vector도 함께 cache (`L_distill` target용): `(num_pairs, K=3, D)`

**(C) Description routing module** *(training only)* — `model/r2d2_routing.py`
- Input: patch tokens `(B, N, D)`, GT pair의 K=3 description token sequence `(B, K, L, D)`
- Routing cross-attention: queries=patches, keys/values=description tokens (B*K*L 통째로 concat)
- Output: enriched patch features `(B, N, D)` = patches + α·attended_desc (residual)
- Patch aggregator는 enriched_patch에 동작

**(D) Losses**
- `L_pair`: aggregated image feat과 pair prototype matching — Troika base contrastive 그대로
- `L_distill` *(NEW)*: aggregated feat을 GT pair의 description EOT embedding mean `(B, D)`에 cosine alignment. `1 - cos(F.normalize(agg), F.normalize(desc_mean))`
- `L_route` *(NEW)*: routing attention map의 diversity contrastive
  - within-image: 다른 patch는 다른 description sub-token에 분산되게 (entropy floor on per-patch attention distribution)
  - across-image: same GT pair의 다른 image들은 비슷한 routing pattern (cross-entropy or InfoNCE)
- `L_decorr`: Troika baseline 그대로

#### Inference path
- Patch aggregator + pair prototype matching만 동작
- Routing module은 **completely disabled** at eval. description bank 미사용.
- → "vision-only at inference" → 6연패 mechanism 회피 보장

### 13-4. Pre-registered judgment (사전 등록)

**Sanity gate**:
- λ_distill = λ_route = 0으로 학습 (= 그냥 patch aggregator만) → Troika baseline HM 0.394 ± 0.005 안에 들어와야 architectural integration 정상. 안 들어오면 patch aggregator 자체가 깨진 것.

**Axis viability (mit-states seed 0, test_pairs from `val_best.closed.json`)**:
- HM Δ ≥ **+0.010** vs Troika baseline → axis 살아있음 → 3-seed 본런 + UT-Zap chain 진행 + SOTA 비교 단계
- HM Δ in **[−0.005, +0.010]** → tie. ablation 1-2개 (routing 빼기 = distill-only, mean-pool로 aggregator 대체 등)로 contribution 격리
- HM Δ < **−0.005** → 7연패. mechanism 진단(text-side 자체가 문제) 더 강화 → narrative pivot G("CZSL baseline tie는 backbone/protocol 지배") 또는 새 axis

**Aux signal (정성 진단용, 1-seed 충분)**:
- attn_acc (attr accuracy) 보다 obj_acc 향상 → description spatial이 obj region을 더 잘 잡았다는 신호
- unseen Δ가 seen Δ보다 음의 폭이 작으면 → mechanism 차단 가설 부분 검증

### 13-5. Stages

| stage | 내용 | 예상 시간 | 산출물 |
|---|---|---|---|
| **0** | CLIP ViT patch token attr-bearing probe (linear probe on patch mean vs CLS for attr classification; 1-2 attention maps) | 2-3h | `notebooks/probe_patch_attr.ipynb` 또는 `tools/probe_patch_attr.py` + 결과 dump |
| **1** | `troika_r2d2.py` skeleton: patch aggregator + `L_distill` only (routing 빼고 distillation 효과 isolated 측정). mit-states seed 0 학습 + eval | 1d | `model/troika_r2d2.py`, `config/troika/mit-states-r2d2-distillonly.yml`, seed 0 결과 |
| **2** | Routing 모듈 추가 (`r2d2_routing.py`) + `L_route`. mit-states seed 0 학습 + eval | 1d | Full R2D2 module, `mit-states-r2d2.yml`, seed 0 결과 |
| **3** | seed 0 결과 vs §13-4 threshold 판정 → 3-seed 본런 또는 ablation 또는 pivot | 2-3d (seed 0 결과 따라) | 3-seed 본런 결과 or pivot 결정 + RESEARCH_LOG §13-6 작성 |

Stage 0이 **gating**: CLIP patch token이 attr를 carrying 안 하면 routing 자체가 noise 학습. 1-2시간이면 빠르게 확인 가능.

### 13-5a. Stage 0 결과 (5-16 KST, n=2000 mit-states train sample, 115 attrs)

`Troika/code/tools/probe_patch_attr.py` 실행 결과 (`Troika/logs/probe_patch_attr_results.json`):

| representation | train acc | **test acc** | Δ vs CLS |
|---|---|---|---|
| CLS feature | 1.0000 | **0.2600** | — |
| Patch mean (256 patches) | 1.0000 | **0.2200** | −0.040 |
| Patch max (256 patches) | 1.0000 | **0.2125** | −0.048 |

- chance ≈ 1/115 = 0.0087 → 모든 representation이 **chance × 25 이상**. patch tokens도 attribute information을 의미 있게 carry.
- CLS가 가장 좋지만 patch_mean과 격차 작음 (−0.04) → routing이 patch-level에서 attr-related signal을 학습할 여지 있음.
- train acc 1.0은 N=2000, D=768 logistic regression saturation. 정량 차이 신뢰는 test acc.

**Verdict: R2D2 valid → Stage 1 진행**.

### 13-5b. Stage 1 launch (5-16 01:46 KST)

`Troika/code/model/troika_r2d2.py` 신규: Troika baseline 위에 PatchAggregator(learnable query attention pool) + L_distill(blended image feat을 GT attr/obj description embedding mean에 cosine align). `r2d2_alpha` init=0 → 학습 초기에는 Troika baseline 동일. Routing module은 Stage 2에서 추가.

Description bank precompute: `data/descriptions_mit_claude.json`의 attr/obj subdescriptions(K=3) → CLIP text encoder → mean → L2 norm → buffer로 저장. 학습 중 indexing만, no gradient.

Config: `config/troika/mit-states-r2d2-distillonly.yml` (model_name=troika_r2d2, distill_weight=1.0, r2d2_alpha_init=0.0). 나머지는 mit-states.yml 기본값.

Launch command:
```
CUDA_VISIBLE_DEVICES=1 python -u train.py \
  --yml_path config/troika/mit-states-r2d2-distillonly.yml \
  --clip_arch /home/student/.cache/clip/ViT-L-14.pt \
  --dataset_path /home/student/dongki/DPAS/data/mit-states \
  --save_path /home/student/dongki/Troika/save/mit-states_seed0_r2d2_distillonly \
  --seed 0
```

PID=3724633, log `train_mit_r2d2_distillonly_seed0_20260516_014624.log`. mit-states Troika baseline 학습 시간 reference ~5h → 종료 ETA ~07:00 KST.

**OOM 두 번 → batch tuning (5-16 01:46~01:49)**: batch=64+accum=1 → OOM (PatchAggregator + autograd graph 추가로 baseline 마진을 초과). batch=32+accum=2 → CMT layer에서 OOM (CMT가 train_pairs=1262 × patches=257에 attention하는 게 본래 무거움). 최종 **batch=16 + accum=4 (eff 64) + `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`** 로 안정 진행. 첫 25 step 기준 1.29 it/s, train loss 5.02 → 4.45 정상 감소. 1 epoch ≈ 24분, 10 epochs ≈ 4h. PID=3726106, log `train_mit_r2d2_distillonly_seed0_20260516_014914.log`.

### 13-5c. Stage 2 코드 준비 완료 (Stage 1 학습 중 백그라운드, launch 대기)

`troika_r2d2.py`에 `DescriptionRouter` 모듈 + `use_routing` flag + `_route_loss` (slot-usage entropy reg) 추가. 같은 파일 유지(설정 ablation 깔끔). `config/troika/mit-states-r2d2.yml` 신규 (`use_routing: true`, `route_alpha_init: 0.0`, `route_weight: 0.0` — Stage 2a: 단순 routing residual). Stage 1 결과 보고 launch.

**자율 plan (사용자 자고 일어날 때까지)**:
- Stage 1 (distill-only) seed 0 — 02:00~06:00 KST
- Stage 2 (distill + routing) seed 0 — 06:00~10:00 KST
- 두 결과 모두 사전 등록 threshold(§13-4) 대비 정리해 RESEARCH_LOG §13-6에 보고
- 3-seed 확장 결정은 사용자 깬 후 (irreversible scale-up)

### 13-5d. Stage 1 (distill-only) seed 0 결과 — 5-16 (학습 5-16 01:49~06:47 KST)

수치 출처: `Troika/logs/train_mit_r2d2_distillonly_seed0_20260516_014914.log` 마지막 줄 (train.py:111-115에서 val_best.pt를 load해 `test_pairs`에 closed-world 평가). val_best.pt mtime 02:48 KST → `val_metric='best_loss'` 기준 ep2 직후 저장된 ckpt.

| 모델 | seen | unseen | HM | AUC | attr | obj |
|---|---|---|---|---|---|---|
| Troika baseline 3-seed mean (§12) | 0.4888 ± 0.0040 | 0.5241 ± 0.0031 | **0.3887 ± 0.00076** | **0.2171 ± 0.00056** | 0.3851 | 0.5566 |
| R2D2 distill-only seed 0 | 0.4626 | 0.5254 | **0.3737** | **0.2044** | 0.3868 | 0.5530 |
| Δ (R2D2 − baseline) | −0.0262 | +0.0013 | **−0.0150** | **−0.0127** | +0.0017 | −0.0036 |

**§13-4 사전 등록 판정**: HM Δ −0.0150 << threshold −0.005 → **7연패 (axis 회생 실패)**. 시드 노이즈(0.00076) 대비 약 20σ 음의 격차. AUC 격차도 약 23σ → 통계적 정착 의문 없음.

**Per-epoch val trajectory (`best_hm` 컬럼 = epoch내 running-best HM on val)**:
- ep1 0.3767 / ep2 0.3975 / **ep3 0.4099 (peak)** / ep4 0.4063 / ep5 0.3943 / ep6 0.3969 / ep7 0.3881 / ep8 0.3808 / ep9 0.3844 / ep10 0.3761
- train loss 2.90 (ep1) → 0.14 (ep10): distill target에 과도하게 수렴.
- ep3에서 baseline mean(0.3887)을 +0.021 초과 → distillation 자체는 **early signal positive**, 그러나 학습이 계속되며 image feature가 description EOT mean에 collapse하여 contrastive prototype matching 능력 손실(unseen은 −5%~ −1% 사이로 비교적 유지되지만 seen 하락이 큼 — overfitting pattern).
- val_metric을 `best_loss`로 잡았더니 val_best.pt가 ep2(HM 0.3975) 시점에 고정 — HM/AUC peak인 ep3를 못 가져옴. **model selection metric 미스매치**.

**즉시 함의**:
1. Stage 2(routing 추가) 자동 launch는 보류 — Stage 1 distill-only가 sanity gate를 통과 못 했고, routing은 distill 위에 쌓는 구조라 같은 collapse를 상속할 가능성이 큼.
2. 단, ep3 peak는 architecture가 **틀린 게 아니라 over-distillation**의 증거. 두 갈래 후속 옵션:
   - (a) `distill_weight` sweep (1.0 → {0.1, 0.3, 0.5})로 collapse 완화. 1-seed 빠르게 ablation.
   - (b) `val_metric='best_hm'`(또는 AUC)로 바꿔 model selection을 metric-aligned로. 동일 ckpt에서도 결과 달라질 가능성.
3. 두 ablation은 cheap (각 ~5h). Stage 2는 (a)+(b) 결과 본 후 결정.

**Stage 2 상태**: 코드(§13-5c) 준비 완료, launch 안 함. GPU 1 idle (5-16 18:25 KST 기준).

### 13-5e. distill_weight sweep launch — 5-16 18:33 KST

§13-5d의 over-distillation 진단을 받아, 같은 모델/하이퍼파라미터에서 `distill_weight`만 1.0 → {0.1, 0.3, 0.5}로 sweep. seed=0, 10 epoch, 동일 batch=16/accum=4 + `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`. 순차 실행 (1 GPU). 예상 ETA: 5h × 3 ≈ 5-17 09:30 KST.

- config: `config/troika/mit-states-r2d2-distillonly-dw{01,03,05}.yml`
- launcher: `Troika/scripts/run_r2d2_distill_sweep.sh`, summary log `logs/run_r2d2_distill_sweep_20260516_183329.log`
- 개별 학습 log: `logs/train_mit_r2d2_distillonly_dw{01,03,05}_seed0_*.log`
- save dir: `save/mit-states_seed0_r2d2_distillonly_dw{01,03,05}/`

**판정 (사전 등록)**:
- 어느 한 dw에서 HM Δ ≥ +0.005 vs Troika baseline 3-seed mean(0.3887) → axis 살아남, dw 최적값에서 (a) `val_metric` ablation(best_hm) 추가 + (b) 3-seed 본런 검토.
- 셋 다 HM Δ < −0.005 → axis 봉인. 단, dw=1.0의 ep3 peak가 sweep 어느 곳에서 final-epoch까지 유지되는지 trajectory도 함께 본 후 판정 (모델 select 미스매치 가능성 §13-5d-2 잔존).
- 셋 다 [−0.005, +0.005] tie → over-distillation 가설 부분 확인(collapse는 덜 됨) but 의미 있는 신호 없음. (b) val_metric 교체로 ep3 peak 회수 가능성 추가 점검.

`val_metric='best_loss'`는 이번 sweep에서도 유지(미스매치 가설 따로 isolating 위해 둘을 한 번에 안 바꿈). 본 sweep 결과 후 별도 ablation.

### 13-5f. distill_weight sweep 결과 — 5-17 09:46 KST 종료

수치 출처: `/home/student/dongki/Troika/logs/train_mit_r2d2_distillonly_dw{01,03,05}_seed0_*.log` 마지막 줄 (val_best.pt → `test_pairs` closed-world).

| 모델 | seen | unseen | HM | AUC | attr | obj | val peak (ep) |
|---|---|---|---|---|---|---|---|
| Troika baseline 3-seed mean (§12) | 0.4888 ± 0.0040 | 0.5241 ± 0.0031 | **0.3887 ± 0.00076** | **0.2171 ± 0.00056** | 0.3851 | 0.5566 | — |
| R2D2 dw=1.0 (§13-5d) | 0.4626 | 0.5254 | **0.3737** | **0.2044** | 0.3868 | 0.5530 | 0.4099 (ep3) |
| R2D2 dw=0.5 | 0.4773 | 0.5306 | **0.3818** | **0.2130** | 0.3942 | 0.5575 | 0.4106 (ep3) |
| R2D2 dw=0.3 | 0.4773 | 0.5329 | **0.3829** | **0.2146** | 0.3942 | 0.5589 | 0.4103 (ep3) |
| R2D2 dw=0.1 | 0.4748 | 0.5346 | **0.3832** | **0.2146** | 0.3954 | 0.5581 | 0.4177 (ep2) |
| Δ dw=0.1 − baseline | −0.0140 | +0.0105 | **−0.0055** | **−0.0025** | +0.0103 | +0.0015 | — |

**§13-5e 사전 등록 판정 적용**:
- 셋 다 HM Δ < −0.005 → **axis 봉인 조건 적중** (dw=0.1: −0.0055, dw=0.3: −0.0058, dw=0.5: −0.0069). 시드 std 0.00076 대비 dw=0.1조차 약 7σ 음의 격차.
- 그러나 단서 조항(§13-5e "단,") 미해소: (i) trajectory 점검 + (ii) `val_metric='best_hm'` ablation 통과 전엔 완전 봉인 아님.

**Over-distillation 가설 부분 확인**:
- dw=1.0 → 0.1로 갈수록 test HM이 monotone 회복 (0.3737 → 0.3818 → 0.3829 → 0.3832, total +0.0095). Val peak도 0.4099 → 0.4177로 +0.0078 상승. **collapse는 분명히 완화됨**.
- 그러나 dw=0.5 → 0.1 구간 추가 개선은 +0.0014에 불과(시드 σ ≈ 2배 수준). **diminishing return → dw↓ 만으로는 baseline parity 회수 불가**.
- unseen은 baseline보다 +0.01 우위(0.5346 vs 0.5241). seen은 −0.014 열위. **distillation이 unseen 일반화에는 미미한 양의 효과, seen에는 손해** 패턴 — over-distillation 잔재.

**Trajectory 비교** (val running-best HM, val_metric=best_loss): 모든 dw에서 ep2-3에 peak, 이후 monotone 하락. dw=0.1만 peak가 ep2로 1 epoch 앞당겨졌고 magnitude도 가장 큼. dw=1.0의 ep3 peak(0.4099)는 dw=0.1에서 ep2 peak(0.4177)로 *유지+개선*되지만, 그 peak ckpt가 best_loss로는 선택 안 됨 → val_metric 미스매치(§13-5d-2)가 sweep에 걸친 systematic gap의 원인일 가능성 농후.

**즉시 함의 / 다음 step**:
1. **봉인 확정 전 1개 ablation 필수**: `val_metric='best_hm'`로 dw=0.1, seed 0, 10 epoch 재학습 (≈5h). 사전 등록 §13-5e 단서 조항 해소.
   - 만약 best_hm으로 ep2 ckpt(val HM 0.4177) 가져와 test HM ≥ 0.3937 (= baseline + 0.005) → axis 살아남, 3-seed로 확장.
   - test HM이 여전히 < 0.3837 (= baseline − 0.005) → 봉인 확정.
   - 사이 구간 tie → architecture 자체 한계로 판단.
2. 만약 봉인 확정시 §13-5c Stage 2(routing) launch는 보류 → R2D2 framework 자체에 sealing decision 적용. axis 후보는 [[project_lhp_czsl_text_enrich]]의 image-conditional selection 방향 재검토.

**Save artifacts**: `/home/student/dongki/Troika/save/mit-states_seed0_r2d2_distillonly_dw{01,03,05}/`에 `val_best.pt`, `final_model.pt`, `epoch_4.pt`, `epoch_9.pt`만 저장 (save_every_n=5). **ep2/ep3 peak ckpt 없음** → val_metric='best_hm' ablation은 학습 재실행 필수, ckpt 재선택만으로 불가.

### 13-5g. best_hm val_metric ablation launch — 5-17 23:29 KST

§13-5f의 단서 조항(model-select 미스매치) 해소용 1런. dw=0.1(sweep best), seed=0, 10 epoch, val_metric='best_hm'으로만 sweep와 차이. 나머지 하이퍼 동일 (batch=16, accum=4, `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`, GPU 1). ETA ≈ 5h → **5-18 04:30 KST 예상**.

- config: `/home/student/dongki/Troika/code/config/troika/mit-states-r2d2-distillonly-dw01-besthm.yml`
- launcher: `/home/student/dongki/Troika/scripts/run_r2d2_dw01_besthm.sh`
- log: `/home/student/dongki/Troika/logs/train_mit_r2d2_distillonly_dw01_besthm_seed0_20260517_232902.log`
- save dir: `/home/student/dongki/Troika/save/mit-states_seed0_r2d2_distillonly_dw01_besthm/`
- nohup wrapper: `/home/student/dongki/Troika/logs/r2d2_dw01_besthm_nohup.log`

**판정 적용 (§13-5e 단서 해소)**:
- 만약 test HM ≥ 0.3937 (= baseline 0.3887 + 0.005) → **axis 살아남**, 3-seed 본런 launch 검토.
- 만약 test HM < 0.3837 (= baseline 0.3887 − 0.005) → **봉인 확정**.
- 사이 구간(tie) → architecture 자체 한계로 판단, R2D2 framework sealing 결정.

**예상 (사전 등록)**: val HM peak ckpt(ep2 sweep에서 0.4177)에서 test 평가하면 val/test generalization gap이 sweep dw=0.1 평균 수준(val 0.4177 vs val_best=ep1 ckpt test 0.3832, gap ≈ −0.035)을 따른다면 test HM ≈ 0.4177 − 0.035 = 0.383 근처 — 즉 tie ~ 봉인 경계. 만약 ep2 ckpt가 ep1보다 generalization도 좋으면 (0.39+) → axis 회생.

### 13-5h. best_hm val_metric ablation 결과 — 5-18 04:36 KST 종료

수치 출처: `/home/student/dongki/Troika/logs/train_mit_r2d2_distillonly_dw01_besthm_seed0_20260517_232902.log` 마지막 줄 (val_best.pt → test_pairs closed-world). 학습 시간 23:29 → 04:36 KST ≈ 5h7m. `val_best.pt` mtime 00:59 KST → **ep3 ckpt** (val HM 0.4172 peak에서 저장).

| Run | seen | unseen | **HM** | **AUC** | attr_acc | obj_acc |
|---|---|---|---|---|---|---|
| baseline post-fix 3-seed mean | 0.4888 | 0.5241 | 0.3887 | 0.2171 | 0.3851 | 0.5566 |
| R2D2 dw=0.1 best_loss (§13-5f) | 0.4748 | 0.5346 | 0.3832 | 0.2146 | 0.3954 | 0.5581 |
| **R2D2 dw=0.1 best_hm (5-18)** | 0.4971 | 0.5295 | **0.3926** | **0.2231** | 0.3948 | 0.5581 |
| Δ (best_hm − best_loss) | +0.0223 | −0.0051 | +0.0094 | +0.0085 | −0.0006 | 0.0 |
| Δ (best_hm − baseline) | +0.0083 | +0.0054 | **+0.0039** | **+0.0060** | +0.0097 | +0.0015 |

**Val trajectory (epoch별 best HM on val, bias-sweep best)**: ep1 0.3826 → ep2 0.4057 → **ep3 0.4172 (peak)** → ep4 0.4149 → ep5 0.4067 → ep6 0.3992 → ep7 0.3990 → ep8 0.3929 → ep9 0.3955 → ep10 0.3927. peak가 ep3로 sweep(ep2, 0.4177)보다 1 epoch 늦지만 magnitude는 사실상 동일(Δ −0.0005). ep5+ 점진 하락 → overfit 신호. AUC peak는 ep4 (val 0.2459).

**§13-5g 사전 등록 판정 적용**:
- test HM ≥ 0.3937 cutoff (axis 살아남): **NO** (0.3926, 0.0011 short)
- test HM ≤ 0.3837 cutoff (봉인 확정): NO (0.0089 well above)
- **사이 구간(tie) → architecture 자체 한계 판단, R2D2 framework sealing ✓**

**해석**:
- val_metric 미스매치(§13-5d-2)는 sweep 격차의 **일부 원인 확정**: best_loss → best_hm 단일 변경만으로 test HM +0.0094, AUC +0.0085 (seed std 0.00076 환산 ~12σ on HM). 즉 dw=0.1 sweep의 −0.0055 회귀의 약 70%가 model-select 미스매치였음.
- 그러나 사전 등록 cutoff(+0.005) 진입은 0.0011 short. baseline 대비 HM +0.0039 (~5σ)는 noise를 명확히 넘는 신호이긴 하나 사전 등록 룰은 strict → 봉인.
- val HM 0.4172 vs test HM 0.3926 → val-test gap 0.0246 (sweep best_loss의 ~0.035 gap보다 작음). val-test generalization 자체는 살아 있고, AUC가 +0.0060 (~10σ)로 HM보다 더 명확히 위 → R2D2 distillation이 약하지만 진짜 정보를 추가하긴 함. 다만 architecture 한계가 cutoff 통과 막을 정도임.
- ClusPro 재현 갭(§11-2, −1.8pp HM)을 우리 R2D2가 −1.4pp까지 좁혔지만 절대 SOTA 도달은 아님.

**SOTA 대비 위치 (mit-states ViT-L/14 closed-world test_pairs)**:

| 모델 | HM | AUC | vs 우리 |
|---|---|---|---|
| ClusPro (ICLR'25, 논문 Table 1) | 0.407 | 0.238 | **+0.0144 HM / +0.0149 AUC 위** |
| CDS-CZSL / Troika / PLID (논문 range) | 0.390~0.393 | 0.221~0.224 | 사실상 동등 / 미세하게 우리가 위 |
| 우리 ClusPro 재현 baseline (3-seed mean) | 0.3887 | 0.2171 | — (reference) |
| **우리 R2D2 dw=0.1 best_hm** | **0.3926** | **0.2231** | **+0.0039 HM / +0.0060 AUC vs 우리 baseline** |

→ **Troika/CDS-CZSL/PLID 보고치 range(0.390~0.393)에 도달**, ClusPro 논문 SOTA 0.407에는 −1.4pp HM 미달. 우리 자체 ClusPro 재현 갭(−1.8pp)을 0.4pp 좁힘. 절대 성능 측면에서 "Troika SOTA-class 재현"으로는 valid한 결과이지만, **이 프로젝트의 의도(ClusPro 위로 axis push)** 기준으로는 cutoff 미달 → 봉인.

**다음 결정 (사전 등록 §13-5g 적용)**:
1. **R2D2 framework sealing** — Stage 2 routing(§13-5c, 코드 준비 완료) launch 보류. UT-Zap cross-dataset 검증도 unnecessary (mit-states에서 architecture sealing 결정 났으므로).
2. 다음 axis 후보: [[project_lhp_czsl_text_enrich]]의 image-conditional selection 방향 재검토. CDS-CZSL 차별화 위해 image-conditional desc selection (d) + visual description 결합 안 (사용자 제안 axis).
3. R2D2 결과는 advisor 미팅 narrative(§13-7)에 "vision-side distillation 시도 → Troika-range 회복하지만 cutoff 미달, axis push 실패 확정"으로 정리.

**Save artifacts**: `/home/student/dongki/Troika/save/mit-states_seed0_r2d2_distillonly_dw01_besthm/`에 `val_best.pt` (ep3 ckpt, 5-18 00:59), `epoch_4.pt`, `epoch_9.pt`, `final_model.pt`, `config.json`.

### 13-6. Open questions (사전 메모)

- **CLIP ViT-L/14 patch tokens의 spatial discriminability 미검증**: contrastive 학습이라 patch가 진짜 region-semantic 있는지 불확실. Stage 0이 이걸 확인.
- **Routing supervision의 GT 부재**: description은 image-conditional이 아닌 (attr, obj) pair-conditional. patch-desc alignment ground truth 없음. → 직접 alignment supervision 안 하고 diversity contrastive로만 학습 (Stage 2의 L_route 설계).
- **λ_distill vs λ_pair 균형**: distillation이 너무 강하면 image feature가 desc embedding 따라가서 prototype matching 망함. seed 0에서 sweep 필요.
- **VRAM**: Troika baseline은 batch=8+accum=16에서 ~12GB. Patch 257 tokens 사용 + routing K=3 desc × L=77 tokens cross-attention → 추가 메모리. 첫 step에서 batch 줄여야 할 수도.

### 13-7. 다음 advisor 미팅 narrative draft

"6연패 검증으로 LLM-text axis의 mechanism이 'text-side ensemble의 seen-overfit'에 있음을 확정. 이를 정면으로 회피하는 R2D2 구조 제안: description은 vision representation에 distill, inference는 vision-only. 동시에 기존 CZSL이 안 쓰는 CLIP patch grid를 활용. Stage 0 probe + Stage 1-2 implementation 진행 중."

→ 미팅 일정 정해지면 §13-5 진행상황 + 가장 최근 seed 0 결과를 page-9 후속으로 보여줌.

---

## 14. Image-conditional description selection axis (5-18 진입)

### 14-1. 진입 동기 + 가설

§13 R2D2 framework sealing (5-18) 후 다음 axis. [[project_lhp_czsl_text_enrich]] Phase B 제안의 직접 실행.

**가설**: v3_text(LLM description ensemble)의 negative 결과 (mit-states HM Δ=−0.028, UT-Zap Δ=−0.053; §1, §11-18)에서 **description content 자체는 정보가 있는데 mean-pool이 image-agnostic이라 unseen 일반화를 손상**시켰다. sub-meaning aggregation을 image-conditional softmax로 바꾸면 (a) seen-pair에서는 baseline 회복, (b) unseen-pair에서는 image와 가장 잘 맞는 description만 강조 → seen-overfit 완화.

**falsifiable**: 이 가설이 맞으면 mit-states에서 HM ≥ baseline + 0.005 (= 0.3937) 도달. 틀리면 description content 자체가 mit-states에서 부정적 → **text-side LLM enrichment axis 6연패 → 7연패로 완전 봉인** (남은 변형 없음, project_lhp_czsl_direction 영구 봉인).

### 14-2. CDS-CZSL (CVPR'24)과 차별화

CDS-CZSL: pair-wise LLM description + image-conditional state weight selection. 그러나:
- single description per state (K=1), state-side만 (object-side는 LLM enrichment 없음)
- 코드 미공개 (§11-2)

우리 차별점 (Phase B):
1. **Variable K_p per primitive** (LLM이 결정한 K, mit-states에서 ~3 평균). multi-prototype + multi-description.
2. **attr+obj 양쪽 image-cond** (LHP-CZSL framework 자체가 두 prototype branch 분리되어 있음).
3. **Logit-level image-cond pool** (sub-prompt feature를 sub-level prompt로 그대로 인코딩 + image feature와 inner product → softmax pool). text encoder forward graph 변경 없음, cost는 baseline mean-pool과 동일.

### 14-3. Design 결정 (8개)

| # | 결정 사항 | 선택 | 비고 |
|---|---|---|---|
| D1 | Framework base | **LHP-CZSL v3_text** (`model/lhp_czsl.py`) | v3_text 봉인은 mean-pool 결정이지 framework 결정이 아님. 인프라 재사용. |
| D2 | Selection scope | **attr-only head + obj-only head** | comp head는 mean-pool 유지 (batch-aware token tensor expansion 비용 회피, §14-6) |
| D3 | Mechanism | **soft softmax weighted** over sub-meanings | hard top-1은 gradient 끊기고 K_p=1 graceful degradation 어려움 |
| D4 | Image feature for cond | `norm_img[i]` (disentangled, i=1 attr / i=2 obj) | 같은 forward의 disentangled feature 자연스러움 |
| D5 | Train-time consistency | **train+test 모두 image-cond** (Phase B1) | mean/cond 불일치는 distribution shift; B2는 cutoff 통과 후 ablation |
| D6 | Description JSON | `data/descriptions_mit.json` (Gemini, v3_text Phase A와 동일 set) / UT-Zap도 동일 정책 | mean-pool vs image-cond 직접 비교를 위해 동일 desc set 필수. Claude version은 cutoff 통과 후 ablation. |
| D7 | val_metric | `best_hm` | §13-5d-2/5g lesson — best_loss는 ep1-2로 선택 편향 |
| D8 | Temperature τ | τ=1.0 (default), τ ∈ {0.5, 0.1} ablation 후순위 | softmax temperature; 작을수록 hard top-1 근사 |

### 14-4. 코드 변경 계획

새 flag `image_cond_select: bool` (config). `text_ensemble=True`일 때만 의미.

**변경 대상 파일**:

1. **`model/lhp_czsl.py`**:
   - `__init__`: `self.image_cond_select = getattr(config, 'image_cond_select', False)` 추가
   - `_image_cond_pool` (신규 helper): sub-level features (S, D) + image features (B, D) + sub_to_prim (S,) → (B, num_prims, D)로 image-cond softmax weighted pool. seg_softmax (per-primitive softmax over its subs) 구현 필요.
   - `train_forward` line 600-614 / `val_forward` line 634-648: text_ensemble=True이고 image_cond_select=True인 경우, `_pool_sub_features`를 skip하고 sub-level feature를 살린 채 image-cond pool 함수 통과. logit이 (B, num_prims) shape이 되도록 (현재 (B, num_prims) feature dot prod와 호환).
   - 핵심 식 (per primitive p, image x):
     - $w_p(x, s) = \text{softmax}_{s \in \text{sub}(p)}(\langle x, v_s \rangle / \tau)$
     - $\text{score}_p(x) = \sum_{s \in \text{sub}(p)} w_p(x, s) \cdot \langle x, v_s \rangle$

2. **`config/`**:
   - 신규: `lhp_czsl_v3_text_imgcond_mit_l14_seed0.yml` — `text_ensemble: True`, `image_cond_select: True`, `sub_meanings_path: data/descriptions_mit_claude.json`, `val_metric: best_hm`, decorr_weight/sem_weight=0 (v3_text와 동일).

3. **`scripts/`**:
   - `scripts/run_v3_text_imgcond_mit_seed0.sh` (Stage 1 single seed)
   - `scripts/run_v3_text_imgcond_mit_3seeds.sh` (Stage 2)

**코드 변경 분량 추정**: ~80-120 LOC (helper 30-50 + forward branch 30-40 + config + script).

### 14-5. 4-stage plan + pre-registered judgment

| Stage | 내용 | 예상 시간 | 산출물 |
|---|---|---|---|
| **0** | Probe: helper 구현 + sanity (mit-states seed 0, **1 epoch**), NaN 검사, loss 정상 감소 확인 | ~25min | `_image_cond_pool` 코드, 1ep log |
| **1** | mit-states seed 0, **15 epoch** (LHP-CZSL 표준, v3_text와 동일), val_metric=best_hm | ~6h | test_pairs HM/AUC. 판정 적용. |
| **2** | Stage 1 ≥ +0.005 통과 시 mit-states 3-seed (seed 0,1,2 sequential) | ~18h | 3-seed mean ± std |
| **3** | Stage 2 통과 시 UT-Zap cross-dataset 3-seed | ~30h | mit-states + UT-Zap 양쪽 신호 확인 |

**Pre-registered judgment (Stage 1)**:
- test HM ≥ 0.3937 (baseline 0.3887 + 0.005) → **axis 살아남, Stage 2 launch**.
- test HM ≤ 0.3837 (baseline − 0.005) → **봉인 확정**. text-side LLM enrichment axis 영구 폐기.
- 사이 구간(tie) → image-cond mechanism이 약하지만 information 있음, τ ablation (τ ∈ {0.5, 0.1}) 1런 추가 후 재판정. τ ablation도 tie면 봉인.

**Pre-registered judgment (Stage 2/3)**:
- mit-states 3-seed mean HM Δ ≥ +0.005 + UT-Zap Δ ≥ +0.005 → **main signal 확정**, paper-class result. 추가 ablation (B2 train mean/test cond, K_p effect, attr-only vs obj-only 분해).
- mit-states 통과 + UT-Zap 실패 → **dataset-dependency** (v3_text 봉인 시점에 reject한 가설 다시 등장 → 신중히 narrative 검토).
- 두 dataset 다 cutoff 미달 → 봉인.

### 14-6. comp head는 왜 제외?

`_construct_token_tensors` (line 386)에서 comp head는 single composition prompt를 만들고 attr_pooled/obj_pooled를 EOS-2/EOS-1 token에 직접 주입 (line 414-415). image-cond 하려면 token tensor를 (B, num_pairs, ctx_len, D)로 expand 후 CLIP text encoder를 B*num_pairs 번 forward 해야 함. mit-states num_pairs ≈ 1900 × B=8 ≈ 15200 prompts × 257 token → VRAM/시간 폭증. cost-benefit 안 맞음.

**대안 (Phase C 후순위)**: comp prompt construction을 batch-agnostic 유지하되 inference 시 logit 합성에 image-cond gate 추가. `logit_infer` (line 682)에서 comp_logits에 image-cond weight 곱하는 식. Stage 1-3 결과 좋으면 검토.

### 14-7. Open questions

- **Disentangler 학습 초기 noise**: `f_attr_proj` / `f_obj_proj`가 초기에는 image-cond 신호로 부족할 수 있음. norm_img를 그대로 쓸지(현재 design) 아니면 ep1-2는 mean-pool로 시작 후 점진 전환할지(curriculum). Stage 0 probe에서 1 epoch loss 곡선 보고 결정.
- **K_p=1 primitive**: mit-states에서 ~20% 추정 (descriptions_mit_claude.json 통계 미확인). softmax over single sub = identity (자동 graceful degradation), 별도 처리 불필요.
- **τ ablation 비용**: tie 영역 진입 시 τ ∈ {0.5, 0.1} × seed 0 = 2런 × 5h ≈ 10h. 사전 등록에 포함.
- **memory pressure**: sub-level feature를 살려두면 logit이 (B, num_pairs)가 아니라 (B, sum_K)가 됨. mit-states sum_K ≈ 360 × 3 ≈ 1080 (attr+obj 합). baseline (B, 360)보다 3배. fp16 + B=8에서 무시할 만함.

### 14-8. 다음 액션

1. **사용자 confirm 받고** Stage 0 probe 시작 — `_image_cond_pool` helper 구현 + config/scripts 작성.
2. Stage 0 sanity 통과 후 Stage 1 launch (GPU 1, ~5h, 5-19 새벽 종료 예상).
3. Stage 1 결과 §14-5 판정 적용 후 §14-9 신설로 결과 정리.

### 14-9. Stage 0 probe 통과 + Stage 1 launch — 5-18 18:05 KST

**Stage 0 (1 epoch sanity, 17:31 ~ 18:05 KST, ~34분)**:

log `logs/train_v3_text_imgcond_mit_probe_20260518_173147.log`. 1차 시도(17:28)는 `Tensor.scatter_reduce` API mismatch로 fail (PyTorch 1.11.0+cu113에는 `scatter_reduce_` 자체가 없음). Helper를 segmented softmax → **padded-indices + masked F.softmax** 방식으로 재구현 후 17:31 retry 성공.

| metric | val | test (val_best=ep1 ckpt) |
|---|---|---|
| best_seen | 0.3845 | 0.3609 |
| best_unseen | 0.4791 | 0.4266 |
| **HM** | **0.3143** | **0.2759** |
| **AUC** | **0.1507** | **0.1217** |
| attr_acc | 0.3239 | 0.3161 |
| obj_acc | 0.5390 | 0.5012 |

**Sanity check 통과 ✓**:
- train loss 2.53 (step 0) → 1.68 (epoch 1 끝). Monotone 감소.
- NaN/inf 표시 0회.
- VRAM peak ~13GB (A5000 24GB 여유).
- forward / loss / backward / eval / save 전부 정상.
- 1 epoch HM 0.27은 절대값 비교 무의미 (baseline 1ep도 비슷 수준). 절대값은 Stage 1 결과에서 판정.

**§14-4 design에서 구현 deviation (semantically 동일)**:
- 원안 helper signature: `(sub_feat, img_feat, sub_to_prim, num_prims, ...)` + scatter_reduce(amax) + scatter_add(sum).
- 변경 signature: `(sub_feat, img_feat, sub_indices_padded, valid_mask, ...)`. `(num_prims, k_max)` padded gather + `masked_fill(~mask, -inf)` + `F.softmax(dim=-1)`. PyTorch 1.11 호환 + 더 적은 LOC.
- `__init__`에 `attr_sub_indices_padded`, `attr_sub_valid_mask`, `obj_sub_indices_padded`, `obj_sub_valid_mask` buffer 추가 (cheap, 약 1800 long + 1800 bool, image_cond_select=False일 때도 항상 build).
- 그 외 §14-3 D1~D8 결정 그대로.

**Stage 1 launch (5-18 18:05 KST)**:
- config: `config/lhp_czsl_v3_text_imgcond_mit_l14_seed0.yml` (15 ep, image_cond_select=True, cond_tau=1.0, val_metric=best_hm, descriptions_mit.json)
- launcher: `scripts/run_v3_text_imgcond_mit_seed0.sh`
- log: `logs/train_v3_text_imgcond_mit_seed0_20260518_180556.log`
- nohup wrapper: `logs/v3_text_imgcond_mit_seed0_nohup.log`
- PID: 4025895, GPU: 1
- ETA: ~6h → **5-19 00:00 KST 예상**

**판정 (§14-5 사전 등록)**:
- test HM ≥ 0.3937 (baseline 0.3887 + 0.005) → Stage 2 3-seed launch.
- test HM ≤ 0.3837 → text-side LLM enrichment axis 영구 봉인 (6연패→7연패).
- tie → τ ∈ {0.5, 0.1} ablation 1런 추가 후 재판정.

**다음 액션**: 5-19 00시 학습 종료 시 final 라인(val_best.pt → test_pairs) 결과 → §14-10 신설.

### 14-10. Stage 1 결과 + axis 영구 봉인 확정 — 5-19 01:18 KST 종료

수치 출처: `/home/student/dongki/LHP-CZSL/logs/train_v3_text_imgcond_mit_seed0_20260518_180556.log` 마지막 줄 (val_best.pt → test_pairs closed-world). 학습 시간 18:05 → 01:18 KST ≈ 7h13m.

| Run | seen | unseen | **HM** | **AUC** | attr_acc | obj_acc |
|---|---|---|---|---|---|---|
| baseline post-fix 3-seed mean (§13-5h) | 0.4888 | 0.5241 | 0.3887 | 0.2171 | 0.3851 | 0.5566 |
| **v3_text image-cond (5-19)** | 0.4819 | 0.4752 | **0.3597** | **0.1909** | 0.3551 | 0.5270 |
| Δ (image-cond − baseline) | −0.0069 | −0.0489 | **−0.0290** | **−0.0262** | −0.0300 | −0.0296 |

**§14-5 사전 등록 판정 적용**:
- test HM 0.3597 ≤ cutoff 0.3837 (baseline − 0.005) → **text-side LLM enrichment axis 영구 봉인 ✓**
- HM −0.029 / AUC −0.026 — 단순 미달 아니라 명확 회귀. 특히 unseen이 −0.049로 더 크게 무너짐 → image-cond selection이 unseen sub-name 선택을 망친 것으로 보임 (training-time selection이 train pair 분포에 overfit, unseen 쌍에서는 잘못된 sub 선택).
- attr_acc/obj_acc도 동시 회귀 → attr/obj head 모두에서 image-cond이 noise로 작동. mechanism 자체가 약함.

**누적 axis 결과 (7연패)**: v3_text(mit) 봉인 → v3_text(utzap) 봉인 → R2D2 framework sealing → image-cond selection 봉인. text/vision/selection 3축 모두 baseline 능가 실패.

**다음 액션**:
1. [[project_lhp_czsl_text_enrich]] axis 완전 close. memory에 봉인 사실 반영.
2. advisor 미팅 narrative (§13-7) 갱신: "vision-side distillation도, image-cond selection도 baseline 미달 — text-augmentation 계열 전체 axis 봉인. 새 axis 탐색 필요."
3. 새 axis 후보: structural/prompt-level alternative (e.g. textual inversion, primitive embedding refinement, train-pair sampling intervention). Phase 분리 검토.

---

## 15. Troika R2D2 test.py 재현 + forward_for_open bug fix (5-19 11:13 KST)

### 15-1. 동기

§13-5h R2D2 dw=0.1 best_hm 수치 검증을 위해 `val_best.pt`로 `test.py` standalone 실행 시도 → 즉시 IndexError 발생.

### 15-2. Bug

`code/model/troika_r2d2.py:489` `forward_for_open` 반환값이 `logits` (list of 3 tensors) 하나뿐인데, `loss_calu` (line 418)에서 `predict[3]`로 image_combined를 참조 → IndexError.

**원인**: train forward (line 534)는 `(logits[0], logits[1], logits[2], batch_combined, route_aux)` 5-tuple 반환하지만 eval forward는 logits만 반환. train/eval 반환 signature 불일치 — `test.py:487`에서 `model.loss_calu(predict, data)` 호출 시 충돌.

**Fix**: `forward_for_open` 반환을 `(logits[0], logits[1], logits[2], batch_combined, None)`로 통일. eval-path는 routing label이 없으므로 `route_aux=None`. `loss_calu` 내부 `route_aux = predict[4] if len(predict) > 4 else None` + `if self.use_routing and self.route_weight > 0` 분기로 None 안전.

`logit_infer`는 `predict[0:3]`만 인덱싱하므로 호환.

### 15-3. 재현 결과

| Source | seen | unseen | HM | AUC | attr | obj |
|---|---|---|---|---|---|---|
| §13-5h train-time auto-test (5-18, test set) | 0.4971 | 0.5295 | **0.3926** | **0.2231** | 0.3948 | 0.5581 |
| **5-19 standalone test.py (val_best.pt, test set)** | 0.4920 | 0.5292 | **0.3887** | **0.2200** | 0.4030 | 0.5614 |
| Δ (standalone − §13-5h) | −0.0051 | −0.0003 | −0.0039 | −0.0031 | +0.0082 | +0.0033 |
| 5-19 standalone — val set (ref) | 0.4935 | 0.5654 | 0.4150 | 0.2423 | 0.4120 | 0.5834 |
| Train log ep3 val reported | 0.4940 | 0.5659 | 0.4172 | 0.2431 | — | — |

**관찰**:
- val 수치는 train log ep3와 ±0.0022 일치 (S/U/HM/AUC) → val_best.pt가 ep3 ckpt 확정.
- test 수치는 §13-5h보다 HM −0.0039, AUC −0.0031 — DataLoader noise 수준보다 약간 큼. 원인 추정: (a) `threshold_trials=50` 재실행 시 bias term 미세 변동 (val biasterm 1.3951 / test biasterm 1.0898), (b) fp16 비결정성, (c) eval-time random state 차이. 의도된 동일 경로지만 완벽한 bit-exact reproducibility는 아님.
- **결론**: §13-5h의 "cutoff +0.005 미달, framework sealing" 판정이 standalone 재평가에서 **오히려 더 명확해짐**. standalone HM 0.3887 = baseline 0.3887 (정확히 동일) → R2D2 distillation의 baseline 대비 개선 사실상 0. AUC만 +0.0029 (~3.7σ on baseline std 0.0008) 미세 양 — distillation 신호가 미세하게 있지만 HM/operating point에 도달하지 못함. **R2D2 framework sealing 결정 무영향, 오히려 강화**.

### 15-4. 영향 범위

- §13-5h 수치 자체는 train script 내부 final test 호출에서 나온 것이므로 영향 없음 (정확히 같은 model.forward 경로).
- 다만 누구든 `test.py`를 R2D2 ckpt에 standalone 적용하면 무조건 fail → 향후 ablation/외부 재현용으로 fix 필요했음. 해당 fix는 본 세션에서 in-place 적용 완료. R2D2 framework sealing 결정에는 무영향 (오히려 standalone HM 0.3887 = baseline으로 더 결정적).

---

## 16. Axis G — Margin loss (CosFace) on comp/attr/obj heads (5-19 진입)

### 16-1. 진입 동기 + 가설

§14-10 text-side enrichment axis 영구 봉인 직후 새 axis. 봉인된 3축(text-side LLM enrichment / vision-side distillation / image-cond selection)을 피하면서 baseline 0.3887을 흔드는 게 목표. text/vision/selection을 모두 건드리지 않고 **loss surface 자체**에 개입하는 가장 cheap한 후보.

**가설**: baseline의 plain CE는 seen pair 사이 boundary가 too soft → seen-overfit (epoch 4-5부터 단조 회귀, §13-5h `R2D2 dw01 best_hm` 학습곡선). CosFace의 additive cosine margin은 seen target class의 cosine을 `m`만큼 낮춰 강제로 boundary tightening → seen logit이 더 강하게 학습되어야만 통과 → over-confident 해소 + unseen pair에 score 여유.

**Why not in 봉인 axis**: input augmentation도 아니고 visual representation도 아니고 selection도 아님. CE 정의 자체의 regularization이라 직교 axis. supervised classification 표준기법 (CosFace ICCV'18, ArcFace CVPR'19)이라 narrative도 깔끔.

### 16-2. 선행 연구 + 차별점

- **CosFace (Wang et al., ICCV 2018)**: face recognition. cos(θ) − m. s=64, m=0.35.
- **ArcFace (Deng et al., CVPR 2019)**: cos(θ + m). angular additive.
- **CZSL에서 margin loss 적용 사례**: 검색 필요 (현재까지 mit-states/UT-Zap CZSL 표준 비교군에서 직접 보고된 적 없음). 있다면 차별화 narrative 조정. **TODO Stage 0 직전 lit search.**

CZSL에서 안 쓴 이유 후보: (a) 1262 pair × 115/245 primitive는 face recognition의 ~수만 class와 다름, m 작아야 함, (b) primitive head와 comp head 사이 margin 상호작용 미검증.

### 16-3. Design 결정 (8개, §14-3 패턴)

| D# | 결정 | 선택 | 근거 |
|---|---|---|---|
| D1 | Margin type | **CosFace** (cos − m, additive cosine) | ArcFace보다 numerical stable, debug 쉬움. m=0.35 face benchmark는 CZSL에 무리지만 일단 cosine-additive로 시작. |
| D2 | 적용 head | **comp + attr + obj 3개 모두** | 셋 다 CE 구조 동일. comp만 적용 시 attr/obj head는 unaffected → 불균형. 전체 일관 처리. |
| D3 | Margin value m | **0.1** (default) | face 0.35는 1262-pair에 강함. mit-states pair similarity 분포상 m=0.1이 cos ~0.3-0.4 평균에서 ~25-30% margin = 합리적 starting point. Stage 1 tie 시 {0.05, 0.2} ablation. |
| D4 | Scale s | **기존 `self.clip.logit_scale.exp()` 유지** (~100, 학습 가능) | learned scale 교체는 baseline 흔들음. margin만 추가, scale 보존. |
| D5 | 적용 시점 | **training only** (loss_calu 내부) | margin은 regularizer. inference logit_infer는 무수정. |
| D6 | val_metric | **`best_hm`** | §13-5g/§14 lesson — best_loss는 ep1-2 over-select. |
| D7 | epochs | **15** | LHP-CZSL 표준 (v3_text / image-cond와 동일). |
| D8 | 구현 | **`loss_calu`에 in-place margin step 삽입** | forward 수정 무. config flag `cosface_margin > 0`일 때만 활성. 봉인 axis(R2D2 forward 수정)와 달리 diff 최소. |

### 16-4. 코드 변경 계획

**대상 파일**: `model/cluspro_baseline.py` (baseline 본체). v3_text / image-cond / R2D2 변형 모델에는 적용 안 함 — pure baseline + margin 단일 변경.

**수정 1**: `loss_calu` (line 402-423).

```python
def _cosface_margin(self, logits, target, m, scale):
    # logits: (B, C) = scale * cos(theta).  target: (B,).
    one_hot = F.one_hot(target, num_classes=logits.size(-1)).float()
    cos = logits / scale
    cos = cos - m * one_hot
    return cos * scale

def loss_calu(self, predict, target):
    loss_fn = nn.CrossEntropyLoss()
    batch_attr, batch_obj, batch_target = target[1], target[2], target[3]
    batch_attr = batch_attr.cuda(); batch_obj = batch_obj.cuda(); batch_target = batch_target.cuda()

    if self.training:
        comp_logits, attr_logits, obj_logits, loss_contras, loss_hsic = predict
    else:
        comp_logits, attr_logits, obj_logits = predict

    m = getattr(self.config, 'cosface_margin', 0.0)
    if self.training and m > 0:
        scale = self.clip.logit_scale.exp().detach()  # detach so margin doesn't gradient back to scale
        comp_logits = self._cosface_margin(comp_logits, batch_target, m, scale)
        attr_logits = self._cosface_margin(attr_logits, batch_attr, m, scale)
        obj_logits = self._cosface_margin(obj_logits, batch_obj, m, scale)

    loss = (self.pair_loss_weight * loss_fn(comp_logits, batch_target)
            + self.attr_loss_weight * loss_fn(attr_logits, batch_attr)
            + self.obj_loss_weight * loss_fn(obj_logits, batch_obj))
    if self.training:
        loss = loss + self.contrastive_weight * loss_contras + self.hsic_weight * loss_hsic
    return loss
```

**수정 2**: `parameters.py`에 `--cosface_margin` argument 추가 (default 0.0).

**수정 3**: 새 config `config/lhp_czsl_baseline_cosface_mit_l14_seed0.yml` — 기존 baseline yml 복제 + `cosface_margin: 0.1` 한 줄 추가.

**수정 4**: 새 launcher `scripts/run_baseline_cosface_mit_seed0.sh` — 기존 baseline 스크립트 복제 + yml 경로만 변경.

**예상 LOC**: model 코드 +15줄, parameters +1줄, config 1줄, launcher 카피.

### 16-5. 4-stage plan + pre-registered judgment

| Stage | 내용 | 예상 시간 | 산출물 |
|---|---|---|---|
| **0** | Probe: 코드 변경 + sanity (mit-states seed 0, **1 epoch**), NaN 검사, loss 정상 감소, margin OFF/ON 둘 다 확인 | ~30min | 1ep log × 2 (m=0, m=0.1) |
| **1** | mit-states seed 0, **15 epoch**, val_metric=best_hm, cosface_margin=0.1 | ~5h | test_pairs HM/AUC. 판정 적용. |
| **2** | Stage 1 통과 시 mit-states 3-seed (0, 1, 2) | ~15h | 3-seed mean ± std |
| **3** | Stage 2 통과 시 UT-Zap cross-dataset 3-seed | ~24h | mit-states + UT-Zap 양쪽 신호 |

**Pre-registered judgment (Stage 1)**:
- test HM ≥ 0.3937 (baseline 0.3887 + 0.005) → **axis 살림, Stage 2 launch**.
- test HM ≤ 0.3837 (baseline − 0.005) → **axis 봉인**. 8연패 누적, margin axis 영구 폐기.
- Tie 구간 (0.3837 < HM < 0.3937) → **m ∈ {0.05, 0.2} ablation (seed 0, 각 5h, 총 10h) 후 best m으로 재판정**. 둘 다 tie면 봉인.

**Pre-registered judgment (Stage 2)**:
- mit-states 3-seed mean HM Δ ≥ +0.005 over baseline 3-seed → Stage 3 (UT-Zap).
- Δ < +0.005 → mit-states-only 효과로 분류, narrative weakening, Stage 3 보류.

**Pre-registered judgment (Stage 3)**:
- UT-Zap 3-seed mean HM Δ ≥ +0.005 → **main signal 확정**. paper-class. 추가 ablation (m sweep, ArcFace 비교, attr-only vs obj-only vs comp-only).
- mit-states 통과 + UT-Zap 실패 → **dataset-dependency** 표기 (axis G도 v3_text와 같은 함정에 빠지는지 확인).
- 둘 다 실패 → 봉인.

### 16-6. Open questions

- **logit_scale × m 상호작용**: scale ≈ 100, m=0.1 → margin subtraction ≈ 10 logit unit. CE softmax에서 사실상 target 강제 무한정 멀어짐 효과 → gradient blow-up 위험. Stage 0 probe에서 첫 step loss와 gradient norm 점검 필수. 위험 시 `m → m / scale` (target cos에서 직접 차감) 대신 `effective_margin = m × scale_clip_max` 등 clip 방식 검토.
- **attr/obj head 동시 margin**: comp pair = attr × obj 관계라 셋 다 margin 시 double-penalize 가능. Stage 0에서 attr/obj head 따로 OFF/ON 비교 검토 (D2 변경 가능성).
- **best_loss val_metric도 같이 봐야?**: §13-5g lesson 따라 best_hm 사용하지만, margin loss는 train loss 자체에 추가 페널티 → val loss 패턴이 baseline과 달라질 수 있음. Stage 1 학습 곡선 보고 사후 분석.
- **HSIC / contrastive loss와 충돌?**: 기존 baseline에 `loss_contras`, `loss_hsic` 있음. margin은 main CE만 수정하므로 충돌 없지만, 셋이 다 regularizer라 효과 중첩 가능. Stage 1 결과 후 ablation (margin only vs margin + hsic 등) 필요.

### 16-7. 다음 액션

1. **사용자 confirm 받고** Stage 0 probe 코드 변경 시작 — `loss_calu` 패치 + parameters + config + launcher.
2. Stage 0 sanity (1ep × 2 — m=0 OFF / m=0.1 ON 비교, OFF 결과는 기존 baseline과 bit-exact 동일해야 함, 코드 변경 무영향 보장).
3. Stage 0 통과 후 Stage 1 launch (GPU 1, ~5h).
4. Stage 1 결과 §16-5 판정 적용 → §16-8 신설.

**ETA**: 5-19 13:00 KST 코드 작업 시작 → 13:30 Stage 0 시작 → 14:00 Stage 0 종료 → 14:00 Stage 1 launch → **5-19 19:00 KST 결과**.

### 16-8. Stage 0 probe 통과 + Stage 1 launch — 5-19 22:19 KST

**코드 변경 (§16-4)**:
- `model/cluspro_baseline.py:402-406` — `_cosface_margin(logits, target, m, scale)` helper (one_hot에 `scale*m` 차감).
- `model/cluspro_baseline.py:420-425` — `loss_calu` 내부 `cosface_margin > 0 and self.training`일 때만 comp/attr/obj 3 head 모두에 margin 적용. `scale = self.clip.logit_scale.exp().detach()` (gradient → scale로 흐르지 않게 분리, §16-6 risk 대응).
- `parameters.py:56` — `--cosface_margin` float default 0.0.
- configs: `lhp_czsl_baseline_cosface_m{00,01}_mit_l14_probe.yml` (1ep), `lhp_czsl_baseline_cosface_m01_mit_l14_seed0.yml` (15ep).
- launchers: `scripts/run_baseline_cosface_mit_probe.sh`, `scripts/run_baseline_cosface_mit_seed0.sh`.

**Stage 0 결과 (5-19 11:31–12:29 KST, 1ep × 2)**:

| Probe | start loss | end loss (step 7586) | val HM | val AUC | test HM | test AUC | NaN/inf |
|---|---|---|---|---|---|---|---|
| m=0.0 OFF | 2.42 | 1.36 | 0.3716 | 0.1972 | 0.3423 | 0.1744 | 0 |
| m=0.1 ON | 6.16 | 5.01 | 0.3763 | 0.2026 | 0.3472 | 0.1804 | 0 |

- m=0.1 시작 loss 6.16 = m=0 시작 2.42 + ~3.74 (scale*m=10 logit unit이 target class에서 빠지므로 CE에 약 +log(softmax 분모 비율) 추가 — 정량적으로 예상 범위, gradient 폭주 아님). 둘 다 monotone 감소, NaN/inf 0회.
- m=0.0 OFF run의 학습 시간 (28m 46s)이 m=0.1 ON (28m 38s)과 동일 수준 → margin 추가 cost 무시할 만함.
- m=0.0 1ep test HM 0.3423은 §13-5h baseline 15ep mean 0.3887보다 당연히 낮음 (1ep만). 의미는 OFF/ON 비교에 있음 — ON이 1ep에서 +0.0049 HM. 단 1ep noise라 Stage 1 결과 대기.

**Stage 0 통과 판정**: §16-6 risk 모두 클리어 — (a) NaN 없음, (b) loss bounded, (c) margin OFF 분기가 forward 무수정 (m>0 분기만 활성).

**Stage 1 launch (5-19 22:19:50 KST)**:
- config: `config/lhp_czsl_baseline_cosface_m01_mit_l14_seed0.yml` (15ep, cosface_margin=0.1, val_metric=best_hm, batch=8/accum=8)
- launcher: `scripts/run_baseline_cosface_mit_seed0.sh`
- log: `logs/train_baseline_cosface_m01_mit_seed0_20260519_221950.log`
- nohup wrapper: `logs/baseline_cosface_m01_seed0_nohup.log`
- PID: 4128877, GPU 1 (88% util, 11.6GB)
- ETA: ~5h → **5-20 03:20 KST 예상**

**판정 (§16-5 사전 등록 재인용)**:
- test HM ≥ 0.3937 (baseline 0.3887 + 0.005) → Stage 2 (mit-states 3-seed) launch.
- test HM ≤ 0.3837 (baseline − 0.005) → axis 봉인 (8연패).
- tie (0.3837 < HM < 0.3937) → m ∈ {0.05, 0.2} ablation 후 재판정.

**다음 액션**: Stage 1 종료 후 final 라인(val_best.pt → test_pairs)으로 §16-5 판정 적용 → 본 절 update.

### 16-9. Stage 1 종료 + 판정 — 5-20 04:25 KST

**학습 종료**: 2026-05-20 04:25:38 KST (시작 22:19:50, 총 ~6h 06m, ETA 5h보다 +1h).

**Val 학습 곡선 (15 epoch)** — best_hm 우상향 후 평탄:

| epoch | best_seen | best_unseen | best_hm | AUC | attr_acc | obj_acc |
|---|---|---|---|---|---|---|
| 1 | 0.4208 | 0.5564 | 0.3789 | 0.2051 | 0.3973 | 0.5819 |
| 2 | 0.4572 | 0.5648 | 0.3920 | 0.2209 | 0.3994 | 0.5859 |
| 3 | 0.4713 | 0.5595 | 0.3897 | 0.2231 | 0.3904 | 0.5841 |
| 4 | 0.4875 | 0.5527 | 0.3957 | 0.2287 | 0.4001 | 0.5803 |
| 5 | 0.4740 | 0.5612 | 0.3932 | 0.2250 | 0.3883 | 0.5799 |
| 6 | 0.4897 | 0.5542 | 0.3954 | 0.2289 | 0.3918 | 0.5808 |
| 7 | 0.4810 | 0.5556 | 0.3932 | 0.2267 | 0.3919 | 0.5791 |
| 8 | 0.4875 | 0.5582 | 0.4017 | 0.2320 | 0.3936 | 0.5791 |
| 9 | 0.4978 | 0.5562 | 0.4041 | 0.2347 | 0.3926 | 0.5790 |
| 10 | 0.4989 | 0.5522 | 0.4056 | 0.2343 | 0.3940 | 0.5812 |
| 11 | 0.5000 | 0.5515 | 0.4078 | 0.2353 | 0.3931 | 0.5788 |
| 12 | 0.5027 | 0.5496 | 0.4072 | 0.2359 | 0.3930 | 0.5764 |
| 13 | 0.5011 | 0.5497 | 0.4066 | 0.2348 | 0.3916 | 0.5774 |
| 14 | 0.5016 | 0.5471 | 0.4058 | 0.2338 | 0.3911 | 0.5727 |
| **15** | **0.5043** | **0.5459** | **0.4086** | **0.2355** | 0.3932 | 0.5745 |

- val best: **epoch 15, HM 0.4086** (val_best.pt 04:21 KST 저장 → val_best 시점 일치).
- AUC는 epoch 12 (0.2359) 최고 — best_hm metric 기준이라 epoch 15 선택.
- monotone 우상향: epoch 1→15 HM Δ +0.0297. NaN/inf 무관측, 학습 안정적.

**Test (Closed World, val_best 모델 = epoch 15)**:

| | seen | unseen | HM | AUC | attr_acc | obj_acc |
|---|---|---|---|---|---|---|
| **cosface m=0.1 seed 0** | 0.4819 | 0.5177 | **0.3763** | 0.2088 | 0.3934 | 0.5514 |
| baseline (§13-5h, 3-seed mean) | — | — | 0.3887 | — | — | — |

- Δ test HM vs baseline: **−0.0124** (0.3763 − 0.3887)
- val→test gap: −0.0323 (val 0.4086 → test 0.3763). §13-5h baseline의 val→test gap과 유사 수준이라 overfit 신호 아님 → margin이 test에서도 무효.

**§16-5 사전 등록 판정 적용**:
- 임계 1 (Stage 2 launch): HM ≥ 0.3937 → **불충족** (0.3763 < 0.3937).
- 임계 2 (axis 봉인): HM ≤ 0.3837 → **충족** (0.3763 ≤ 0.3837, Δ −0.0124 < −0.005).
- Tie 구간 (0.3837 < HM < 0.3937): 해당 안 됨.

**결론**: ❌ **CosFace margin axis 봉인.** Stage 2 (3-seed), Stage 3 (UT-Zap), m sweep 모두 launch 안 함 — pre-registered가 m={0.05, 0.2} ablation은 tie 구간일 때만 허용, 본 결과는 명확한 fail이라 ablation 면제.

**누적 실패 카운터**: §13-5h 이후 axis 봉인 누적 **8연패** (이전 7 + cosface margin) — pre-registration대로 추가 margin variant 실험 금지.

**왜 망가졌는지 (post-hoc 가설, 결정에 영향 없음)**:
- val HM은 baseline 0.3887 → cosface 0.4086 (+0.0199 val) → **val에서는 보임**.
- test HM은 0.3763 → val→test 일반화 실패. cosface margin이 train pair에 대해서만 decision boundary를 강제로 멀게 만들었지만 unseen comp pair에는 그 margin이 의미 없음. inference 시 m=0 분기로 흐르는 코드 구조라 train/test mismatch가 강해진 듯 (§16-6 open Q "best_loss val_metric"와 연결).
- 단, 이건 후속 ablation 없이는 검증 불가, axis 봉인 결정에 영향 없음. 메모로만 기록.

**산출물**:
- 모델: `checkpoint/lhp_czsl_baseline_cosface_m01_mit_l14_seed0/val_best.pt`, `final_model.pt` (각 1.75GB).
- 로그: `logs/train_baseline_cosface_m01_mit_seed0_20260519_221950.log` (10.7MB).

**다음 액션**:
- Cosface margin axis 사망 → §17 next-axis 선정 필요. 후보:
  - (a) text enrichment via LLM-visual-descriptions (memory: `project_lhp_czsl_text_enrich`) — image-conditional selection 결합.
  - (b) §16-1 sealed 8 axes 외 미탐색 axis brainstorm.
- 사용자 confirm 받고 다음 axis briefing.

---

## 17. Axis H — Multi-seed ensemble (H2) + inference-weight calibration (H4) (5-20 진입)

### 17-1. 진입 동기 + 가설

§16-9 cosface 봉인 직후 (axis 9연패는 아님 — H2/H4는 retrain 없는 inference-time intervention, mechanism-level sealed 8 axes와 직교). 사용자가 "어떻게든 SOTA 달성" 지시 → 가장 cheap한 inference-time 두 axis를 우선 평가.

**Initial reading 오류 정정**: H4 framing 시 test.py:215+ `evaluator.evaluate_predictions`가 **이미 test data에서 bias sweep**해서 `best_hm`을 산출하고 있음을 발견 (test.py:305 `correct_unseen_score_diff` from test data → biaslist iteration). 따라서 단순 bias 캘리브레이션은 dead axis. 진짜 미탐색 lever는:
- **H2**: 3-seed val_best.pt의 raw (comp, attr, obj) logits를 평균 → `logit_infer` 통과 → evaluator. seed std 0.0008 대비 +0.005 cutoff 기대.
- **H4 (revised)**: `model.logit_infer`의 (`pair_inf_w`, `attr_inf_w`, `obj_inf_w`) 가중치 + per-head softmax 온도 (`T_a`, `T_o`) 를 val에서 grid sweep, test에 적용. 현 default (1,1,1,1,1) 외 점 미탐색.

**가설**: 
- H2는 seed-noise를 줄여 val→test gap (cosface에서 −0.032 관측)을 부분 해소.
- H4는 `logit_infer`가 `comp + attr_softmax × obj_softmax` 구조라 가중치 변경이 unseen pair에서 ranking을 재조정할 여지 있음.

### 17-2. Design 결정

| D# | 결정 | 선택 | 근거 |
|---|---|---|---|
| D1 | H2 ensemble 방식 | **raw logit (comp/attr/obj) 평균** | logit_infer 비선형성 보존, H4 grid와 stackable. |
| D2 | 사용 ckpt | `cluspro_baseline_l14_mit_v2_seed{0,1,2}/val_best.pt` (§12-1) | post-fix 3-seed mean baseline 0.3887 ± 0.0008과 직접 비교. |
| D3 | H4 grid (Stage A coarse) | PAIR∈{1,2}, ATTR/OBJ∈{1,3,10}, T=1 → 18 cfg | 빠른 first signal. |
| D4 | H4 grid (Stage B fine) | PAIR∈{0.1,0.5,1,2}, ATTR/OBJ∈{0.5,1,5}, T_a/T_o∈{0.5,1,2} → 324 cfg | coarse no-signal 후 lower pair_w + temperature 확장. |
| D5 | val_metric | val `best_hm` (evaluator 자체가 bias sweep 내장) | inference-time 캘리브레이션에서도 동일 metric. |
| D6 | Stacking | H4 grid를 ensemble val에 sweep → ensemble test에 best cfg 적용 | H2+H4 직교성 검증. |

### 17-3. 코드 + 캐시

**스크립트**:
- `scripts/eval_ensemble.py` — 3-seed forward + raw logit 캐싱 + H2 ensemble + H4 coarse sweep.
- `scripts/eval_sweep_fine.py` — 캐시 재사용 + H4 fine sweep on ensemble.

**캐시**: `checkpoint/ensemble_cache_mit_l14_v2/seed{0,1,2}_{val,test}.pt` — raw (comp, attr, obj) tensors, attr_gt, obj_gt, pair_gt. 재사용 가능.

**검증**: per-seed default 적용 시 §12-1과 정확 일치 (seed 0/1/2 test HM 0.3879/0.3889/0.3894). `apply_logit_infer` 벡터화 재구현이 `model.logit_infer`와 동일함을 sanity 확인.

### 17-4. 결과

**H2 (3-seed ensemble, default weights)**:

| | seen | unseen | HM | AUC | attr | obj |
|---|---|---|---|---|---|---|
| seed 0 val (sanity) | 0.5136 | 0.5680 | 0.4267 | 0.2527 | 0.4023 | 0.5896 |
| seed 0 test (sanity) | 0.4933 | 0.5226 | 0.3879 | 0.2176 | 0.3842 | 0.5590 |
| seed 1 test | 0.4857 | 0.5276 | 0.3889 | 0.2172 | 0.3858 | 0.5561 |
| seed 2 test | 0.4874 | 0.5220 | 0.3894 | 0.2165 | 0.3852 | 0.5546 |
| **baseline 3-seed mean** (§12-1) | 0.4888 | 0.5241 | **0.3887 ± 0.0008** | 0.2171 | 0.3851 | 0.5566 |
| **H2 ensemble VAL** | 0.5217 | 0.5715 | **0.4326** | 0.2589 | 0.4050 | 0.5917 |
| **H2 ensemble TEST** | 0.4912 | 0.5308 | **0.3924** | 0.2218 | 0.3857 | 0.5595 |

- H2 ensemble TEST HM Δ = **+0.0037** vs 3-seed single mean. seed std 0.0008 기준 **~5σ** → 통계적으로 의미 있음.
- AUC Δ = +0.0047 (~8σ).
- 그러나 사전 등록 cutoff **+0.005에 −0.0013 미달** → 엄격 룰로는 tie zone.

**H4 coarse (18 cfg, seed 0)** — best val cfg = **(1.0, 1.0, 1.0, 1.0, 1.0) 즉 default**. test HM 0.3879. **gain 0**.

**H4 fine (324 cfg, ensemble)** — best val cfg = (0.5, 0.5, 1.0, 0.5, 2.0). VAL HM 0.4329, TEST HM **0.3923** (Δ vs H2 default −0.0001, 노이즈). top-10 cfg가 모두 val_hm 0.4329로 동률 → grid 평면에서 logit_infer 변형의 ranking 영향이 미미함.

**Stacked (H2 + H4)**: 0.3923 (== H2 단독).

### 17-5. 판정 + 진단

**H4 axis 사망 확정**: 
- coarse + fine grid 모두 default ≈ optimal.
- 메커니즘: `logit_infer`에서 `comp_logits`의 scale (~logit_scale × cos ≈ 100 × 0.4 = 40)이 `attr_pred × obj_pred` 항 (∈ [0,1])보다 약 1-2 order 크다. 어떤 (pair_w, attr_w, obj_w) 조합도 comp 항을 dominant하게 두는 한 ranking 거의 동일. 
- `pair_w=0.1`로 comp 억제해도 attr×obj는 frequent primitive pair에 bias 걸리고 evaluator의 bias sweep이 이를 다시 보정 → net 변화 0.
- 결론: **inference-weight/temperature 캘리브레이션 axis는 mit-states/ViT-L/14에서 effective space가 거의 비어있다**.

**H2 axis 부분 성공**:
- +0.0037 statistically real (5σ), 사전 등록 cutoff +0.005에는 0.0013 short.
- mechanism: 3-seed val_best.pt가 서로 다른 train 데이터 순서로 다르게 overfit → 평균하면 high-variance 영역만 cancel, 공통 신호 잔존. val→test gap 0.032 → ensemble val 0.4326 / test 0.3924로 gap이 0.040으로 약간 커짐 → ensemble은 val에서 더 많이 얻고 test에서 비례적으로 얻음.
- **strict cutoff 미달이지만 axis 자체는 dead 아님**. 추가 다양성 (multi-epoch, multi-config) 결합 시 +0.005 진입 가능성 있음.

**SOTA 대비 현 위치**:
- 3-seed mean: 0.3887 → 우리 최고 (R2D2 dw01 best_hm 0.3926, §13-5h) → H2 ensemble 0.3924 (R2D2 dw01과 거의 동률, 단 retrain 없음).
- ClusPro paper 0.407까지 **Δ +0.0146 필요**. H2 단독으로는 미달.

### 17-6. 다음 액션 — pivot 결정 필요

H2 +0.0037을 baseline의 새 reporting standard로 채택 (no retrain cost) + 다음 axis는 큰 신호를 노려야 SOTA 도달 가능.

후보:
- **H1 (hard-pair sampling, `same_prim_sample=True`)** — data-level 새 axis, retrain 필요 (5h × n seed). CGE/SymNet 등 표준 트릭, 한 번도 안 시도. 사전 등록 cutoff +0.005 vs (H2 0.3924) → ≥ 0.3974.
- **H3 (ClusPro 재현 갭 §11-2 직접 닫기)** — 1.8pp gap (λ_h, init, dropout) 진단 활용. 성공 시 0.407 SOTA. debug-heavy.
- **multi-checkpoint × multi-seed ensemble** — 3 seed × {ep13, ep14, ep15} = 9-12 ckpt 평균. forward만 ~60분 추가. cheap 확장.

**현 권장**: multi-ckpt ensemble을 H2 확장으로 먼저 시도 (cheap), 그 후 H1 / H3 큰 axis 결정.

**산출물**:
- `scripts/eval_ensemble.py`, `scripts/eval_sweep_fine.py`
- `checkpoint/ensemble_cache_mit_l14_v2/seed{0,1,2}_{val,test}.pt` (재사용 가능)
- `checkpoint/ensemble_cache_mit_l14_v2/results.json`, `results_fine.json`
- `logs/eval_ensemble_h2_h4_coarse.log`, `logs/eval_sweep_fine.log`

### 17-7. Multi-ckpt × multi-seed ensemble 결과 — H2 ceiling 확정 (5-20 23:27 1차 crash, 5-21 18:09 재실행 → 18:31 종료)

**1차 (5-20 23:27)**: `SEED_EPOCHS[2] = [15, 14, 11]`이었으나 `epoch_15.pt` 미존재 (train epochs=15는 0-indexed로 0..14 저장)으로 seed 2 ep 15 load 시 FileNotFoundError. seed 0/1 forward 결과만 캐시에 남고 종료.

**2차 (5-21 18:09)**: `SEED_EPOCHS[2] = [14, 13, 11]` (trajectory 다음 best로 교체) 후 재실행. seed 0/1 cache hit, seed 2만 새로 forward. 총 22분.

| cfg | TEST seen | unseen | **HM** | AUC | Δ vs baseline mean (0.3887) | Δ vs H2 top1 (0.3924) |
|---|---|---|---|---|---|---|
| M0 top1×3 seed (3 ckpts: s0e11, s1e13, s2e14) | 0.4916 | 0.5293 | **0.3921** | 0.2214 | +0.0034 | −0.0003 |
| M1.5 top2×3 seed (6 ckpts) | 0.4903 | 0.5293 | **0.3906** | 0.2209 | +0.0019 | −0.0018 |
| M2 full top3×3 seed (9 ckpts) | 0.4912 | 0.5301 | **0.3907** | 0.2213 | +0.0020 | −0.0017 |
| M1 within-seed top3 (per seed) | s0 0.3861 / s1 0.3885 / s2 0.3885 | — | — | — | (각 단독 ckpt 평균) | — |

**판정**:
- M0 (3 ckpt) ≈ §17-4 H2 ensemble (0.3924) 거의 동치 — 작은 차이는 val_best.pt vs explicit epoch_*.pt selection 사이의 ckpt 약간 다름에 기인 (0.0003).
- **6 ckpt / 9 ckpt 확장은 오히려 성능 하락** (M1.5 −0.0018, M2 −0.0017 vs H2 top1). 추가 epoch 평균이 regularization 과해 ensemble peak를 깎음. top epoch 외 epochs가 noise만 보탬.
- 사전 등록 cutoff `+0.005 vs baseline mean` → 필요 HM ≥ 0.3937. **M0/M1.5/M2 모두 미달**.

**H2 axis ceiling 확정**:
- inference-time ckpt ensemble로 추출 가능한 신호는 H2 top1 (0.3924)에서 saturate. multi-epoch 확장은 dead.
- mit-states / ViT-L/14 / `cluspro_baseline_l14_mit_v2` framework 한정 결론. UT-Zap이나 다른 framework에서는 다를 수 있음 (검증 X).

**산출물**:
- `scripts/eval_multi_ckpt.py` (D5 best_hm metric 기준, M0/M1/M1.5/M2 모두 평가)
- `checkpoint/ensemble_cache_mit_l14_v2/results_multickpt.json`
- `logs/eval_multi_ckpt_20260521_180954.log`

### 17-8. Pivot 결정 — H1 (hard-pair contrastive) 진입 (5-21)

§17-7로 H2 axis(no retrain cost ensemble) 봉인. SOTA(0.407)까지 Δ +0.0146 갭 남았고, §11-7에서 cheap yml-level H3 (paper hp probe) 이미 dead 확인 (HM −0.004), R2D2/imgsel/cosface 4 axis sealed. 남은 living 후보:

| axis | 코드 작업 | retrain 시간 | 예상 gain | 평가 |
|---|---|---|---|---|
| **H1** hard-pair contrastive | model+train 100-200 LOC | 3-seed ~15h | +0.003 ~ +0.008 (CGE/SymNet 표준) | cheap engineering, bounded ceiling |
| **H3** ClusPro 구조적 메커니즘 포팅 | OT(`local_assign`+`distributed_sinkhorn`) + CrossAttn + queue contrast, 300-500 LOC | retrain ~15h | paper §4.4 단일 최대 (+3.3 AUC) | 1-2주 엔지니어링, 통합 리스크 큼 |

**결정: H1 먼저.**

근거:
- 시간-신호: 1-2일 코드 + 15h 학습 → ~3일 내 axis 생사 판정.
- §16 (cosface) sealing 후 8 sealed axes 누적 — 다음 axis도 fail 가능성을 통계적으로 합리적으로 가정해야 함. 빠른 turnover가 EV 높음.
- 사전 등록 cutoff +0.005 통과 시: stack with H2 (0.3924 baseline) → target 0.397+, paper 0.407까지 추가 0.010 갭 → H3로 진입 정당화.
- 사전 등록 cutoff 미달 시: 9th sealed axis 추가. H3 OT 본격 진입 (남은 유일한 high-ceiling 후보).

**H1 Design (v0.1)**:

(A) **Dataset (이미 구현됨)**: `dataset.py:145-154, 218-226`이 `same_prim_sample=True`일 때 (same_attr/diff_obj img, same_obj/diff_attr img) 쌍을 batch에 자동 추가. anchor와 같은 attr이지만 obj 다른 이미지 + 같은 obj이지만 attr 다른 이미지. mask는 sampling 가능 여부 (training pair에 해당 pivot이 존재할 때만 True).

(B) **Model change** (`model/cluspro_baseline.py:train_forward`):
- batch[4..7], batch[9..12]에서 보조 이미지 unpack → encode_image → disentangler 통과 → f_attr_same, f_obj_same 등 추출.
- Hard-pair contrastive loss: 
  - `L_hard_attr = -log( exp(sim(f_attr_anchor, f_attr_same_attr)/τ) / Σ_neg exp(sim(f_attr_anchor, f_attr_neg)/τ) )` — same_attr 이미지가 positive (attr 공유), same_obj 이미지가 negative (attr 다름).
  - `L_hard_obj`: 대칭.
- Total: `loss += λ_hard * (L_hard_attr + L_hard_obj)`, τ=0.1, λ_hard ∈ {0.05, 0.1, 0.2} (D2 ablation 후보).

(C) **Train.py change** (`train.py:160`): `same_prim_sample=config.same_prim_sample` 이미 통과. batch 길이 동적이라 model_factory의 dataloader collate 확인 필요.

(D) **YML**: `config/cluspro_baseline_mit_l14_v2.yml` 복사 → `config/cluspro_baseline_mit_l14_v2_hardpair.yml`, `same_prim_sample: true` + `hard_pair_weight: 0.1` + `hard_pair_temperature: 0.1` 추가.

(E) **사전 등록 판정**:
- 1-seed probe (seed 0, ~5h): HM Δ ≥ +0.005 vs single-seed baseline (0.3879) → 3-seed 본런 + H2 stack.
- HM Δ ∈ [−0.005, +0.005]: tie, ablation (λ_hard sweep: 0.05/0.1/0.2) 1-2회.
- HM Δ < −0.005: dead, 봉인 → H3 OT 진입.

**다음 액션**: 
1. `model/cluspro_baseline.py`에 hard-pair branch 추가
2. `dataset.py` mask 처리 확인 + collate 호환
3. smoke test (1 epoch + 1 val) → loss curve check
4. seed 0 본 학습 launch

### 17-9. H1 구현 + smoke test + seed 0 launch — 5-21 18:37 KST

**코드 변경**:
- `parameters.py`: `--hard_pair_weight` (default 0.0), `--hard_pair_temperature` (default 0.1) 추가.
- `model/cluspro_baseline.py:train_forward`: `len(batch) >= 14 and hard_pair_weight > 0`일 때 batch[4] (same_attr_img), batch[9] (same_obj_img) 같이 encode_image (3B concat → split). 각각 disentangler + projection 통과.
- `model/cluspro_baseline.py:_hard_pair_loss`: InfoNCE 형식.
  - L_attr: `anchor_attr_proj` ↔ `sa_attr_proj` positive, within-batch `so_attr_proj` (다른 attr) negatives.
  - L_obj : `anchor_obj_proj` ↔ `so_obj_proj` positive, within-batch `sa_obj_proj` (다른 obj) negatives.
  - mask로 sampling 실패 샘플 제외.
- `loss_calu`: `predict` 6-tuple (loss_hard 포함), `loss += hard_pair_weight * loss_hard`.
- `config/cluspro_baseline_mit_l14_v2_hardpair_seed0.yml` 신규: baseline yml 기반 + `same_prim_sample: true`, `hard_pair_weight: 0.1`, `hard_pair_temperature: 0.1`. 메모리 제약 (3× I/O → micro-batch 12)로 `train_batch_size: 4`, `gradient_accumulation_steps: 16` (effective bs 64 유지).

**Smoke (5-21 18:37)**:
- launch: pid 217696, log `logs/train_hardpair_mit_seed0_20260521_183706.log`
- 첫 1분: loss finite (1.2), GPU 13.06 GiB, 2.6 it/s. OOM/NaN 없음.
- 1 epoch ETA ~49분 (7585 step), 15 epoch ≈ **13h ETA**, 5-22 08:00 KST 종료 예상
- val_metric=best_hm로 매 epoch val 후 val_best.pt 저장.

**사전 등록 판정 (5-21 등록, 결과 5-22)**:
- HM Δ ≥ +0.005 vs single-seed baseline 0.3879 → 0.3929 이상 → 3-seed 본런 launch + H2 stack 시도.
- HM Δ ∈ [−0.005, +0.005] → tie, λ_hard sweep {0.05, 0.2} 1-2회.
- HM Δ < −0.005 → 9th sealed axis 추가, H3 OT 본격 진입.

### 17-10. H3 OT 코드 분석 — 포팅 비용 재추정 (5-21 18:50 KST, H1 학습 중 백그라운드)

§11-2에서 "1-2주 엔지니어링" 추정했던 H3 OT 포팅을 H1 학습 도는 동안 공식 `cluspro_baseline.py` (root, 34KB, byte-identical to official ClusPro)를 다시 읽어 재추정함. **실제 변경 범위는 50-80 LOC 수준**으로 훨씬 작음.

**핵심 발견 1: `local_assign`은 우리도 이미 갖고 있다 (`model/otgcc.py`)**
- 공식 train_forward (line 698-725) `local_assign(batch_attr, init_q, top_percent=1)` 호출 → import 경로 `from .otgcc import *` (root cluspro_baseline.py:18).
- 우리 `model/otgcc.py:7-38` 존재. 그러나 **softmax-only stripped 버전**. 진짜 Sinkhorn 없음.
- 진짜 Sinkhorn 구현 (`distributed_sinkhorn`, line 163-187)은 root cluspro_baseline.py에 있으나 **호출처 없음** (dead code in official 파일). 즉 공식 ClusPro의 진짜 OT lever = `local_assign`이 Sinkhorn을 내부 호출하는 미공개 버전이었음. paper §4.4의 +3.3 AUC 기여가 여기서 옴.

**핵심 발견 2: 우리 `_update_prototypes`와 공식 train_forward의 차이는 2가지뿐**

| 항목 | 우리 (model/cluspro_baseline.py:246-262) | 공식 (cluspro_baseline.py:698-725) |
|---|---|---|
| Coupling 계산 범위 | 클래스 k 샘플만 (`feats_k = batch_attr_f[mask]`) | **전체 batch** (`init_q = attr_masks[..., k]`) over B 샘플 |
| Coupling 분포 | `F.softmax(sim / 0.5)` row-wise softmax | `local_assign(...)` → Sinkhorn double-stochastic balance |
| Hard assign | `F.gumbel_softmax(couplings, tau=0.5, hard=True)` | 동일 |
| EMA prototype update | 동일 (`queue * μ + new_proto * (1-μ)`) | 동일 |
| attr_labels 생성 | 별도 패스 `_get_cluster_labels` | 같은 루프에서 `indexs[attr_idx==k] + cluter_num*k` |

→ **Sinkhorn(전체 batch)이 핵심**. 단일 클래스의 B_k = 1~3 샘플로는 marginal balance 의미 없음. 전체 B=64 위에서 cluster usage 균형을 강제할 때 prototype collapse 방지 + soft assignment 정보 보존이 동시 성립.

**핵심 발견 3: §11-2에서 우려했던 다른 메커니즘들은 실제 train_forward에서 안 쓰임**

| 메커니즘 | 정의 위치 | 공식 train_forward (664-820) 호출 | 결론 |
|---|---|---|---|
| `CrossAttentionLayer` + `MulitHeadAttention` | line 110-162 | **호출 없음** (`cross_attn`로 grep) | dead code |
| `pos_neg`, `_sample_negative` | line 611-661 | **호출 없음** | dead code |
| `_dequeue_and_enqueue` | line 630-646 | 호출 있음 (line 806-807) BUT `self.attr_queue` (2D 샘플 queue)는 어디서도 loss에 안 들어감 | dead 업데이트 |
| `self.attr_queue` 2D | line 392 | line 787-793에서 projection 계산되지만 어디에도 안 쓰임 | dead code |
| `distributed_sinkhorn` standalone | line 163-187 | 호출 없음 | dead code (그러나 우리가 `local_assign`에 이식해야 할 알고리즘) |
| `entropic_COT_*` | line 188-314 | 호출 없음 | dead code |
| `loss_contrastive` 구성 | line 800-803 | 사용. `featattr_memory_con1 = [attr_protos, obj_protos]` concat을 negative로 nceloss. obj prototypes를 attr nceloss의 추가 negative로 mixing. | 우리 nceloss는 `all_protos_flat = [attr_protos; obj_protos]`로 이미 같음 (model/cluspro_baseline.py:326-329). 동치. |

§11-2 (1)에서 "구조적 메커니즘 누락"이라 한 항목 중 **실제로 train_forward 신호 경로에 들어가는 것은 `local_assign`의 OT 부분 하나뿐**. 나머지는 공식 코드의 vestigial.

**포팅 plan (LOC 추정 60-80)**

1. **`model/otgcc.py` 재작성** (~30 LOC):
   - `local_assign_ot(features, similarity_scores, sinkhorn_iters=3, epsilon=0.05, top_percent=1.0)` 신규 또는 기존 `local_assign` 교체.
   - 본체: `distributed_sinkhorn` 알고리즘을 (B, K) coupling matrix에 적용.
     ```python
     L = torch.exp(similarity_scores / epsilon).t()  # K x B
     L /= L.sum()
     for _ in range(sinkhorn_iters):
         L /= L.sum(dim=1, keepdim=True); L /= K
         L /= L.sum(dim=0, keepdim=True); L /= B
     L *= B
     return L.t(), selected_mask  # (B, K)
     ```
   - top_percent gate (필요 시 high-confidence 샘플만 사용).
   - Backward compat: default behavior로 softmax 유지 옵션 + `use_ot=True` flag.

2. **`model/cluspro_baseline.py:_update_prototypes` 재구조화** (~30-40 LOC):
   - 현재: 클래스 k 루프 안에서 `feats_k = batch_attr_f[mask]` (subset) → sim → softmax → gumbel.
   - 변경: 루프 밖에서 `attr_protos_all = [attr_queue0..N]` `(num_attrs, cluster_num, D)` 구성 → `attr_masks = einsum('bd, kmd -> bmk', batch_attr_norm, attr_protos_all_norm)` `(B, cluster_num, num_attrs)`.
   - 루프 안: `init_q = attr_masks[..., k]` `(B, cluster_num)` → `local_assign_ot(batch_attr.detach(), init_q.detach())` → Sinkhorn 균형 couplings → gumbel hard assign → `q[attr_idx==k]` subset → EMA proto update.
   - attr_labels도 같은 루프에서 채움 (별도 `_get_cluster_labels` 호출 제거 가능, 또는 보존).
   - obj 측 대칭.

3. **config flag** (~5 LOC):
   - `--use_ot_assignment` (bool, default False) `parameters.py`에 추가.
   - yml에 `use_ot_assignment: true` 토글.

4. **새 config** (~동일 yml 1개):
   - `config/cluspro_baseline_mit_l14_v2_ot_seed0.yml` — baseline 기반 + `use_ot_assignment: true`.

5. **Sanity gate (사전 등록)**:
   - `use_ot_assignment=False`로 학습 → baseline HM 0.3879 ± 0.0008 안에 들어와야 통합 정상.
   - `use_ot_assignment=True` seed 0 → HM Δ ≥ +0.005 vs 0.3879 → 0.3929 이상 → 3-seed 본런 + paper 0.407 대비 갭 측정.
   - HM Δ ∈ [−0.005, +0.005] → ε ∈ {0.03, 0.1}, sinkhorn_iters ∈ {3, 10} 4-cell ablation.
   - HM Δ < −0.005 → paper §4.4 ablation과 모순. ε/iter 한 번 확장 후에도 음이면 sealing (10th).

**실제 코딩 작업 일정**:
- H1 학습 13h (5-22 08:00 KST 종료) 동안 design 확정 + LOC budgeting 끝.
- H1 결과가 negative 면 H3 OT 본격 코딩 진입. 코딩 + smoke ~6h, 15 epoch retrain ~6h (baseline 속도, same_prim_sample 없음) → 5-23 늦은 오후 결과.
- H1 결과가 positive면 H3 OT는 H1 후속(3-seed 본런 + H2 stack) 이후 별도 axis로 진입.

**리스크**:
- Sinkhorn ε=0.05 hyperparam이 우리 batch=8, cluster_num=5 setup에 맞는지 검증 필요. 너무 작으면 mode collapse (one-hot 수렴), 너무 크면 uniform (정보 손실).
- AMP autocast 안에서 Sinkhorn 안정성 (이미 우리 `_update_prototypes`는 `@torch.autocast(..., enabled=False)`로 float32 강제하고 있음 — Sinkhorn도 같은 보호 받음).
- top_percent < 1.0 사용 시 어떤 confidence 기준으로 cutoff할지. 일단 1.0으로 시작.

### 17-11. H1 seed 0 학습 종료 — tie band 음의 가장자리, 사실상 fail (5-21 18:37 → 5-22 07:23 KST, 약 12h46m)

**최종 TEST 결과 (val_best.pt @ ep14)**:
- best_seen 0.4815 | best_unseen 0.5225 | **best_hm 0.383** | AUC 0.2137 | attr_acc 0.3878 | obj_acc 0.5575
- 정상 종료 (Traceback/OOM/NaN 없음).

**VAL trajectory (epoch 1 → 15)**:
- ep1 0.3687 → ep2 0.3907 → ep3 0.3956 → ep4 0.4021 → ep5 0.4057 → ep6 0.4112 → ep7 0.4111 → ep8 0.4126 → ep9 0.4148 → ep10 0.4169 → ep11 0.4188 → ep12 0.4189 → ep13 0.4193 → **ep14 peak 0.4225** → ep15 0.4175
- val_metric=best_hm로 ep14에서 val_best.pt 저장됨.

**판정 (사전 등록 §17-9 기준, single-seed baseline 0.3879 대비)**:
- Δ HM = 0.383 − 0.3879 = **−0.0049**
- cutoff `< −0.005`까지 **0.0001 부족**. 명목상 [−0.005, +0.005] tie band 진입.
- baseline std ±0.0008 고려 시 베이스라인 동일 ~ 약간 손해. positive 신호 없음.

**핵심 관찰 — val/test divergence 확대**:
- val peak 0.4225 vs test 0.383 → val→test gap **0.0395**.
- 베이스라인 단일 시드 val→test gap ≈ 0.032 (§17 영역) → hardpair에서 +0.008 확대.
- 이는 hard-pair contrastive 신호가 training/val에서 같은-attr/같은-obj 쌍의 분리를 잘 학습하지만 test pair (unseen comp)로 일반화 안 됨을 시사. attr 표현이 batch 내 hard negative에 과적합되어 unseen attr-obj 조합의 attr 일반화 약화.

**Pre-registered next step**:
- tie band 형식 충족 → λ_hard ∈ {0.05, 0.2} 1-2 cell ablation 의무.
- BUT val/test gap 확대가 mechanism-level 우려 신호 → λ만 바꿔도 같은 패턴 반복할 가능성 큼.

**결정 (5-22 사용자 승인)**: λ_hard sweep skip, **H1 = 9th sealed axis로 즉시 봉인**, H3 OT 본격 진입.
- 근거: val→test gap +0.008 확대는 mechanism-level 결함 (hard negative overfitting). λ 줄여도 같은 메커니즘이 약해진 채 잔존할 뿐 일반화 결함 자체는 안 사라짐. EV < cost.
- 9th sealed axes 누적: R2D2, imgsel, cosface, v3_text, vision_distill, image_cond, ckpt-ensemble (multi-epoch 확장), feasibility, **hardpair**.
- 남은 living axis: **H3 OT (last high-ceiling 후보)**. fail 시 framework 전환 검토 단계 진입.

**산출물**:
- checkpoint: `checkpoint/cluspro_baseline_l14_mit_v2_hardpair_seed0/val_best.pt` (epoch 14)
- log: `logs/train_hardpair_mit_seed0_20260521_183706.log` (21MB, CR 진행률 포함)
- config: `config/cluspro_baseline_mit_l14_v2_hardpair_seed0.yml`

**SOTA 갭 (변동 없음)**:
- 우리 최고: H2 ensemble 0.3924 (§17-4). H1 seed 0 단독 0.383은 그 아래.
- ClusPro paper 0.407까지 Δ +0.0146 갭. H1으로 못 메움. H3 OT가 남은 유일한 high-ceiling 후보.

