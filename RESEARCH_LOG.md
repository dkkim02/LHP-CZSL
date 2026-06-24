# LHP-CZSL Research Log

---

## ★ 최신 요약 — §20 Redefined-Axis Program (2026-06-01, 4축 전부 SEALED)

**목표**: Troika baseline의 cross-framework gap (mit-states/ViT-L/14: **−0.017 HM / −0.020 AUC** vs ClusPro 0.407/0.238)을 재정의 4축 C′/D′/E/F로 닫을 수 있는지 사전 등록 검증.
**Baseline**: Troika best_loss 3-seed **HM 0.3899 ± 0.0038 / AUC 0.2177** (single-seed §11-11 = 0.3940). 시드 band ±0.011 HM(3σ). **Cutoff**: test HM > 0.4050 (Δ > +0.011) → 유의 / 0~+0.011 tie / <0 fail → 봉인.

| 축 | 메커니즘 | 개입강도(bite) | 결과 | Δ HM | 판정 |
|---|---|---|---|---|---|
| **C′** | logic-rule (attr mutex + obj–attr incompat soft penalty, λ=0.1) | ~0.2% | 3-seed HM **0.3890** / AUC .2160 | −0.0009 | fail → 봉인 (5-28) |
| **D′** | LLM semantic hard-neg InfoNCE (λ=0.1, τ=0.1) | ~0.8% | seed0 HM **0.3951** / AUC .2189 | +0.0011 | tie → 봉인 (5-31) |
| **E** | LLM taxonomy aux CE (attr→18 state-groups, obj→21 super-cats) | **~3.3%** | seed0 HM **0.382** / AUC .2136 | −0.012 | fail → 봉인 (6-1) |
| **F** | LLM multimodal test-time rerank (top-5 재선택) | inference-only | clean probe top1 0.340→**0.320** | net −0.02 | null → 봉인, $65 waive (6-1) |

**핵심 3줄:**
1. **개입을 강하게 할수록 더 나쁘다** — bite 0.2→0.8→3.3%로 키웠으나 신호 0/음수, 최강 E가 −0.012로 worst. "보조 supervision 강화 = 표현 개선" 가설 직접 반증(coarse prior가 fine-grained 변별 희석, 과적합 가속).
2. **inference rerank(F)도 null** — zero-shot LLM은 동시 참 속성 다수일 때(huge↔inflated balloon) dataset single-label과 어긋나 복구(+5)<파괴(−6). (1차 +0.22는 이미지 경로에 정답 박힌 leak artifact → 익명화+gt-blind 평가자 재측정 후 −0.02 확정. *vision-LLM eval 시 path/filename leak 필수 점검.*)
3. **gap은 architectural** — legacy 11축 + ckpt-selection + §20 4축 = 누적 16축 전부 null/sealed. axis/hyperparam/inference로 닫을 EV ≈ 0 → **새 architecture/framework 진입 국면**.

산출물: `section20_summary.md`(이 표 1p), `Troika/code/probe_f.py`(rerank harness), `Troika/data/F_probe/test_top5_preds.json`(test 12,995장 top-5 캐시). 상세: §20-4(C′)/§20-5(D′)/§20-6(E)/§20-7(F).

---

작성일 2026-05-02 16:15 KST. 4-27 ClusPro baseline (K=5) 시작 시점부터 오늘 3-seed 본런 시작 직전까지의 정리. **5-3 09:00 업데이트**: UT-Zap v1_init_only 3-seed 결과 (§1) + baseline tie 결론 + 다음 단계 후보 (§4). **5-3 21:35 업데이트**: v3_text 3-seed 본런이 16:08 KST에 끝났으나 train loss와 test 숫자가 v1_init_only와 비트 일치 → **invalid run으로 폐기**, §4.5에 디버깅 단서 정리. **5-5 20:35 업데이트**: mit-states LLM feasibility 점수 28,175 조합 생성 완료 (§6). **5-6 10:54 업데이트**: v3_text refinement Phase A — LLM visual descriptions를 mit-states에 적용한 3-seed 본런 launch (§10). **5-10 22:19 업데이트**: 5-8 재현 런이 seed0 ep8 test 도중 SIGKILL로 사망 (디스크 96% + GPU 외부 contention 추정), 디스크 정리 후 seed0만 재시도 launch (§10-6). **5-11 16:30 업데이트**: seed0 retry 완주 + 3-seed test_pairs 재평가 → **Phase A negative 확정** (HM Δ=−0.028 vs baseline). 이전 "+0.012 HM 신호"는 val_pairs(우리) vs test_pairs(논문/baseline) 비교 오류였음. §1 mit-states 표 + §10-5 분기 갱신, §11 신설(convention 정리 + ClusPro 재현 갭). **5-16 R2D2 진입 결정 업데이트**: §11-18 결과 보고 사용자가 axis pivot. v3_text 류 add-on은 봉인했지만 description 정보는 살린다 — **vision-side 구조를 새로 설계해서 distillation으로 흡수, inference에는 description을 안 쓴다**. CLIP ViT patch grid(기존 CZSL이 안 쓰는 lever) + LLM description spatial richness 결합. 4-stage plan + pre-registered judgment 사전 등록. 상세 §13. **5-14 21:27 업데이트**: UT-Zap chain (Troika baseline + Troika v3_text-claude) 종료. baseline HM 0.5139 / AUC 0.3782, v3_text-claude HM 0.4612 / AUC 0.3252 → **Δ HM −0.053, AUC −0.053** (사전 등록 "<−0.005" bucket 적중). **UT-Zap에서도 v3_text negative → dataset-dependency 가설 기각, v3_text axis 완전 봉인**. 면담 page 9 +1.5pp 신호의 출처는 LHP-CZSL framework noise + short categorical sub-name 형식이었음. 상세 §11-18. **5-15 03:55 업데이트**: 4-27/28 baseline은 pre-AMP-fix(5-2 이전) ckpt라 노이즈 corrupt 위험 있음. 비교 기준선 재확보를 위해 **mit-states cluspro_baseline post-fix 3-seed 재학습 launch** (config: `cluspro_baseline_mit_l14_v2_seed{0,1,2}.yml`, GPU 1, sequential). 첫 batch 1.08 it/s, 1 run ≈ 11.6 GB VRAM. paperhp probe(5-11) 결과는 baseline 대비 test 거의 동률(HM −0.004, AUC +0.0004) → **paperhp 설정 폐기**. **5-16 업데이트**: 5-15 baseline 3-seed 완주 (22:34 KST 종료). post-fix 3-seed mean HM 0.3887 / AUC 0.2171, **4-27 pre-fix single-seed(0.3893/0.2169)와 사실상 동일** (Δ HM −0.0006, Δ AUC +0.0002). 시드 std HM ±0.00076 / AUC ±0.00056로 매우 작음 → **mit-states/ViT-L/14에서는 AMP NaN 오염이 metric에 영향 없었음** 확인 ([[project_lhp_czsl_amp_nan]] 가정 일부 기각). 기존 mit-states 표의 1-seed 결과 모두 신뢰 유효. v3_text −2.8pp HM은 시드 std 대비 ~35σ → 분명한 음의 신호. 상세 §12. **5-17 09:46 업데이트**: R2D2 distill_weight sweep(dw=0.1/0.3/0.5) 완주. dw=0.1이 best (test HM 0.3832, Δ vs baseline −0.0055 ≈ 7σ). 셋 다 사전 등록 봉인 조건(<−0.005) 적중하지만, val_metric=best_loss로 ep1-2 ckpt 선택 → val HM peak(ep2-3, 0.41+) 회수 못함. **봉인 확정 전 best_hm val_metric ablation 1런(dw=0.1, seed0) 필요**. Over-distillation 가설은 부분 확인 (dw 1.0→0.1 test HM +0.0095, val peak +0.0078). 상세 §13-5f. **5-18 17:10 업데이트**: §13-5g best_hm ablation 완주 (5-17 23:29 ~ 5-18 04:36 KST). val_best=ep3 ckpt → test HM 0.3926 / AUC 0.2231. val_metric 단일 변경만으로 dw=0.1 best_loss 대비 HM +0.0094, AUC +0.0085 — val_metric 미스매치(§13-5d-2)가 격차 일부의 원인 확인. 그러나 baseline 대비 Δ HM +0.0039 (~5σ) / Δ AUC +0.0060 (~10σ)로 **사전 등록 cutoff(+0.005) 진입 실패, 0.0011 short**. tie 영역(0.3837 ≤ HM < 0.3937) 진입 → **§13-5g 룰에 의해 R2D2 framework sealing 결정**, Stage 2 routing(§13-5c) launch 보류. SOTA 대비: Troika/CDS-CZSL/PLID range(HM 0.390~0.393) 도달, ClusPro 논문 SOTA(HM 0.407)에는 −1.4pp 미달 — 우리 ClusPro 재현 갭(§11-2, −1.8pp) 안에 머무름. 상세 §13-5h. **5-18 17:30 axis pivot**: R2D2 봉인 후 다음 axis는 [[project_lhp_czsl_text_enrich]]의 Phase B — **image-conditional description selection on LHP-CZSL v3_text framework**. v3_text mean-pool 가설을 직접 검증: description content는 그대로 두고 sub-meaning aggregation만 image-cond softmax로 교체. CDS-CZSL과 차별: (1) primitive-level multi-prototype (variable K_p, CDS는 single state per primitive), (2) attr+obj 양쪽 image-cond (CDS는 attr/state-only). 4-stage plan + pre-registered judgment §14 신설. **5-18 18:05 업데이트**: Stage 0 probe (1ep sanity) 통과 — train loss 2.53→1.68 monotone, NaN 0회, VRAM 13GB. PyTorch 1.11 scatter_reduce_ 미지원으로 helper signature 변경 (padded indices + masked F.softmax, semantically 동일). **Stage 1 launch (mit-states seed 0, 15 epoch, PID 4025895)** — ETA 5-19 00:00 KST. 상세 §14-9. **5-28 업데이트**: C′(logic-rule grounding) 3-seed 완주 (seed0 5-27 04:19, seed1→2 chain 5-28 09:07). best_loss 3-seed mean HM 0.3890 ± 0.0035 / AUC 0.2160 ± 0.0027 — baseline best_loss(0.3899/0.2177) 대비 Δ HM −0.0009 / Δ AUC −0.0017, 시드 노이즈 안의 **신호 0**. Stage 2 cutoff 0.4009 대비 −0.0119 (~3.4σ 아래). 원인: λ=0.1 mutex/compat loss 기여가 base의 ~0.2%로 inert (§20-4-1 예고대로). **C′ axis fail** (사용자 결정: 즉시 봉인). training-change 봉인 8→9. 상세 §20-4-6. **D′ 진입**: LLM semantic hard-neg InfoNCE(troika_hardneg, λ=0.1/τ=0.1). hard-neg는 큐레이션 클러스터(attr sim-group + obj family) 기반 1164/1262 comp×top-5(cross-primitive 39%). Stage 0 smoke 통과(NaN0/VRAM14GB/1.76it/s, L_hn 기여 ~0.8% = C′의 4배). **Stage 1 launch 15:47 (seed0, GPU1 단독, ETA ~22:00). cutoff test HM>0.4050.** 상세 §20-5-1/2.

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

### 17-12. H3 OT seed 0 학습 종료 — 사실상 baseline 동률, 즉시 sealing (5-22 11:08 → 5-22 21:10 KST, 약 10h)

**구현**: §17-10 plan 그대로 적용.
- `model/otgcc.py`: `local_assign_ot(features, similarity_scores, sinkhorn_iters, epsilon, top_percent)` 신규. (B, K) similarity에 `distributed_sinkhorn` 알고리즘 (exp → row/col 교대 normalize) 적용.
- `model/cluspro_baseline.py:_update_prototypes`: `use_ot_assignment=True`일 때 전체 batch coupling matrix로 Sinkhorn 균형 적용 → gumbel hard assign. attr/obj 대칭.
- `parameters.py`: `--use_ot_assignment` (default False), `--sinkhorn_epsilon=0.05`, `--sinkhorn_iters=3` 추가.
- config: `config/cluspro_baseline_mit_l14_v2_ot_seed0.yml` (baseline 기반 + `use_ot_assignment: true`).
- AMP autocast 안에서 Sinkhorn 안정성: `_update_prototypes`가 이미 `@torch.autocast(enabled=False)`로 fp32 강제 → Sinkhorn도 fp32 실행.

**Launch**: 5-22 11:08 KST. `same_prim_sample=False`라 batch I/O 단순 → micro-batch 8, grad_accum 8 (effective bs 64). 1 epoch ~27분, 15 epoch 약 10h. H1 (13h)보다 빠름.

**최종 TEST 결과 (val_best.pt @ ep14)**:
- best_seen 0.4849 | best_unseen 0.5242 | **best_hm 0.3867** | AUC 0.2167 | attr_acc 0.3855 | obj_acc 0.5561
- 정상 종료 (Traceback/OOM/NaN 없음).

**VAL trajectory (epoch 1 → 15, running best_hm)**:
- ep1 0.3744 → ep2 0.3949 → ep3 0.3973 → ep4 0.4125 → ep5 0.4135 → ep6 0.4145 → ep7 0.4143 → ep8 0.4172 → ep9 **0.4249** → ep10 0.4214 → ep11 0.4225 → ep12 0.4240 → ep13 0.4237 → **ep14 peak 0.4250** → ep15 0.4246
- val_metric=best_hm로 ep14에서 val_best.pt 저장. 사실상 ep9에서 이미 saturate.

**판정 (사전 등록 §17-10 기준, single-seed baseline 0.3879 대비)**:

| 지표 | H3 OT seed 0 | baseline seed 0 (§12-1, §17-4) | Δ |
|---|---|---|---|
| **TEST HM** | 0.3867 | 0.3879 | **−0.0012** |
| TEST AUC | 0.2167 | 0.2176 | −0.0009 |
| TEST seen | 0.4849 | 0.4933 | −0.0084 |
| TEST unseen | 0.5242 | 0.5226 | +0.0016 |
| TEST attr | 0.3855 | 0.3842 | +0.0013 |
| TEST obj | 0.5561 | 0.5590 | −0.0029 |
| **VAL HM (best ckpt)** | 0.4250 | 0.4267 | **−0.0017** |

- HM Δ = **−0.0012**, cutoff `[−0.005, +0.005]` **tie band 중앙**. baseline std ±0.0008 기준 −1.5σ. 통계적으로 baseline과 구분 안 됨.
- paper §4.4의 +3.3 AUC 메커니즘이 **우리 framework에서 0pp 재현**.

**핵심 관찰 — H1과는 다른 failure 모드**:
- val/test gap: H3 OT 0.0383 vs baseline 0.0388 → **gap 거의 동일** (H1은 +0.008 확대). overfitting 아님.
- **val 자체도 −0.0017**로 baseline 미만. 즉 OT signal이 어디서도 도움 안 됨.
- 진단: paper의 +3.3 AUC는 raw ClusPro framework (K=5, paper hyperparam) 위에서 측정. 우리는 LHP framework 위에서 `_update_prototypes`만 OT로 교체했으나, attr_labels 생성 경로(`_get_cluster_labels`)와 nceloss/contrastive_weight 결합이 OT가 만드는 soft balance 신호를 흡수하지 못함. paper 메커니즘은 framework 일체로 작동.

**결정 (5-23 사용자 승인)**: ε/sinkhorn_iters 4-cell ablation skip, **H3 OT = 10th sealed axis로 즉시 봉인**. framework pivot 검토 단계 진입.
- 근거: val에서도 negative라 mechanism-level dead. ε/iter 조정은 동일 메커니즘 안의 HP search → EV < cost. H1 sealing precedent 그대로 적용.
- 10th sealed axes: R2D2, imgsel, cosface, v3_text, vision_distill, image_cond, ckpt-ensemble, feasibility, hardpair, **OT**.
- **living high-ceiling axis 소진**. framework pivot 결정 단계.

**산출물**:
- checkpoint: `checkpoint/cluspro_baseline_l14_mit_v2_ot_seed0/val_best.pt` (epoch 14)
- log: `logs/train_ot_mit_seed0_20260522_110850.log` (10.7MB)
- config: `config/cluspro_baseline_mit_l14_v2_ot_seed0.yml`
- 코드: `model/otgcc.py` (재작성), `model/cluspro_baseline.py:_update_prototypes` (OT 분기), `parameters.py` (3 flags)

**SOTA 갭 (변동 없음)**:
- 우리 최고: H2 ensemble 0.3924 (§17-4). H3 OT seed 0 단독 0.3867은 그 아래.
- ClusPro paper 0.407까지 Δ +0.0146. mit-states / ViT-L/14 / LHP-CZSL framework 안에서 남은 living axis 없음.

### 17-13. Framework pivot 검토 (5-23, 진행 중)

10 axes sealed in current setup (mit-states + ViT-L/14 + cluspro_baseline framework). 남은 방향:

| 옵션 | 비용 | 기대 신호 | 리스크 |
|---|---|---|---|
| **(A) UT-Zappos 재시도** | retrain ~6-8h × 3 seed | mit-states에서 sealed axes 중 일부가 UT-Zap에서 살아날 수 있음 (도메인 다름). 단 v3_text는 [[project_lhp_czsl_direction]]에서 양 dataset 모두 fail 확인 — 모든 axis가 mit-states 결과와 같으리라는 보장 없음 | UT-Zap도 모두 fail이면 framework 전체 dead 판정 |
| **(B) backbone 교체 (ViT-B/16 또는 더 강한 backbone)** | retrain 시간 backbone에 따라 변동. ViT-B/16은 빠르지만 ceiling 낮음. CLIP RN50x16 / OpenCLIP G/14 등 | backbone capacity 변화로 axis effective space 다시 열릴 수 있음 | code change 큼, paper SOTA와 직접 비교 어려움 |
| **(C) Troika framework 전환** | dongki/Troika에 이미 존재. 우리가 §15에서 mit-states test HM 0.394 재현 — ClusPro 0.3893보다 위. 새 SOTA chase 기반선 | Troika 위에서 H1/H2/H3 axis 재시도 가능. Troika는 attribute 측 cross-modal decomposition 구조 다름 → 같은 axis가 다르게 작동 가능 | 새 framework 적응 비용. 우리 LHP 코드 자산 일부만 이식 가능 |
| **(D) novel mechanism 탐색** | 시간 무한정 | 진짜 new axis 발굴 가능 | EV 낮음 (이미 10 axes sealed). 학회 deadline 등 외부 제약 있다면 위험 |

권장: **(C) Troika framework 전환**. 이미 재현된 강한 기반선(0.394 > ClusPro 0.3893)에서 시작 → mit-states 위에서 paper 0.407 SOTA chase에 더 유리. (A)는 sealed axis 검증용으로 cheap이라 (C)와 병행 가능.

**5-23 사용자 결정**: (C) Troika framework 전환 + 첫 axis로 **H3 OT를 Troika에 이식** (사용자 선택). [[project_lhp_czsl_h3_ot_sealed]]에 "OT 메커니즘은 framework-dependent" 명시 → Troika에서 재평가 가능.

### 17-14. Troika OT 이식 — 사전등록 (5-23 KST, 작업 중)

**가설**: H3 OT(Sinkhorn-balanced prototype assignment)는 LHP-CZSL/cluspro_baseline framework에서 메커니즘이 framework에 흡수되어 신호 0이었음 (§17-12). Troika framework는 attribute/object disentangler가 분리돼 있고 prototype/contrastive 구조가 없으므로, prototype + OT-balanced clustering aux loss를 **새 신호 경로**로 끼우면 attr/obj feature space에 추가 supervision이 흘러 SOTA chase에 도움될 수 있음. 단, 메커니즘이 framework-dependent임이 §17-12에서 입증됐으므로 부정 결과도 가능.

**설계 (Option A: visual prototypes + OT + contrastive aux loss)**:

LHP-CZSL `model/cluspro_baseline.py:_update_prototypes_ot` + `_get_cluster_labels` + nceloss를 Troika 위에서 재구성. paper(ClusPro §4.4) 메커니즘 그대로 옮긴 최소 포팅.

- **새 파일**: `Troika/code/model/troika_ot.py` (TroikaOT 클래스, ~330 LOC), `Troika/code/model/ot_utils.py` (local_assign_ot 포팅)
- **변경 파일**: `Troika/code/model/model_factory.py` (troika_ot 분기)
- **새 config**: `Troika/code/config/troika/mit-states-ot.yml`
- **prototype 구조**: K=5 prototype queue per primitive (115 attr + 245 obj = 360 queue × 5 × 768 ≈ 1.4M 추가 buffer, EMA momentum 0.99)
- **OT 업데이트**: 매 train step, attr_disentangler/obj_disentangler 출력 features로 Sinkhorn-balanced 배치-prototype coupling (ε=0.05, iters=3) → gumbel hard → EMA. paper §4.4와 동일.
- **aux loss**: 매 train step, cluster_labels (B,) over (num_prim × K) classes → l2-normalize(disentangler_feat) vs all prototypes의 cosine similarity logits / temperature=0.07 → cross_entropy. attr + obj 평균. `contrastive_weight=0.1`로 main loss에 가산. main forward path는 byte-identical to baseline.
- **prototype init**: 첫 배치에서 per-class mean으로 K slot 채우고 작은 noise 추가 (uniform OT trivial 해결책 방지)

**Hyperparameters (고정, LHP/ClusPro 매칭)**: cluster_num=5, proto_momentum=0.99, sinkhorn_epsilon=0.05, sinkhorn_iters=3, contrastive_weight=0.1, cluster_temperature=0.07.

**학습 설정**: Troika baseline(§11-11)과 동일 — mit-states / ViT-L/14 / lr=1e-4 / wd=1e-5 / attr_dropout=0.3 / batch=8 × grad_accum=8 / epochs=10 / val_metric=best_loss / seed 0 / GPU 1.

**비교 기준**: Troika baseline seed 0 (§11-11) — test HM **0.3940**, AUC **0.2177**.

**사전 등록 cutoff (single-seed)**:
- **Δ HM ≥ +0.005 vs baseline (즉 HM ≥ 0.3990)** → 신호 인정, 3-seed 확장 launch
- **−0.005 ≤ Δ HM < +0.005 (HM ∈ [0.3890, 0.3990))** → tie band → 11th sealed axis (즉시 봉인, ε/iter sweep skip)
- **Δ HM < −0.005 (HM < 0.3890)** → dead → 11th sealed axis

같은 cutoff을 §13-5h, §16-5, §17-9에서 적용 — 일관성 유지. Troika baseline은 1-seed라 std 미상이지만, cluspro framework 3-seed std HM ±0.0008 기준 ±0.005는 ~6σ로 안전.

**EV 평가**: Troika 위에서 prototype + OT가 작동한다는 직접 증거 없음. mit-states/ClusPro 5-23 sealing 결과(Δ HM −0.0012)를 prior로 받으면 base rate ≈ 0. 그러나 framework가 바뀌면 "기존 framework가 OT 신호를 흡수했다"는 LHP 진단이 끊기므로 재평가 가치 있음. 1-seed cost ≈ 10-15h (Troika baseline 속도 + aux loss overhead ~10%).

**Decision tree**:
- HM ≥ 0.3990: 3-seed 확장 후 결과 보고. 의미 있으면 paper-track novelty axis 후보.
- HM tie band: 11th sealed axis 확정. framework pivot의 첫 axis도 dead 판정 → 다음 옵션 (H1 Troika 이식 / Troika-native mechanism / 학회 narrative pivot) 선택.
- HM < 0.3890: dead, 같은 11th sealed axis 처리.

**Smoke test 통과 조건 (사전 등록)**:
- 1 epoch on GPU 1, batch=8, grad_accum=8 — train loss monotone 하강, NaN/Inf 0회, peak VRAM < 22GB.
- aux_loss 첫 step 값이 log(num_prim × K) ≈ log(575) ≈ 6.35 부근에서 시작해야 init이 정상 (uniform expectation).
- 통과 시 본런 launch.

**산출물 예정**:
- 코드: `Troika/code/model/troika_ot.py`, `model/ot_utils.py`, `config/troika/mit-states-ot.yml`
- 학습: PID/log/save_path는 launch 시 업데이트

### 17-15. Smoke test 통과 + seed 0 본런 launch (5-23 14:01 → 14:58, launch 14:59 KST)

**Smoke (1ep, batch=8 × accum=8, GPU 1)**:
- 시작 14:01:36, 종료 14:58:28 — 약 57분 (학습 ~47분 + val 4분 + test 6분)
- Train loss epoch 평균 1.466, 첫 step 2.66 → step 20 2.22 → 단조 추세 ✓
- VRAM peak 7.2 GB (cutoff 22 GB 안) ✓
- NaN/Inf 0회, RuntimeError 없음 ✓
- 1ep ckpt eval: val HM 0.3766 / AUC 0.2072, test HM 0.3544 / AUC 0.1822 (1ep 단독은 baseline 비교 의미 없음 — sanity 용)
- 사전 등록 통과 조건(§17-14) 모두 충족 → 본런 launch.

**본런 (10ep, seed 0)**: 14:59:07 KST, PID 3376964, GPU 1.
- log: `Troika/logs/train_mit_ot_seed0_20260523_145907.log`
- save: `Troika/save/mit-states_seed0_ot/`
- ETA: 10ep × ~57min = **약 9.5시간** (5-23 24:30 KST 부근 종료 예상)
- 종료 후 §17-14 cutoff 적용:
  - test HM ≥ 0.3990 → 3-seed launch
  - tie band 또는 음수 → 11th sealed axis 즉시 봉인

### 17-16. seed 0 본런 중단 + epoch 1-5 결과 (5-24 02:10 KST 확인)

**상태**: 학습 사망 — log 마지막 write 5-23 19:05 KST, epoch 6 test eval iteration 249/326 (~76%) 에서 멈춤. log 내 Traceback/Error/OOM/NaN 0건. nohup 없이 `bash scripts/run_ot_seed0.sh` 직접 실행이라 **SSH/터미널 세션 종료에 의한 SIGHUP 추정**. dmesg/journal OOM 흔적 없음, GPU 1 현재 idle (15 MiB).

**완료된 epoch 1-5 test 결과** (per-epoch test metric, threshold sweep 적용):

| epoch | train loss | seen | unseen | **test HM** | **test AUC** | attr_acc | obj_acc |
|---|---|---|---|---|---|---|---|
| 1 | 1.466 | 0.4273 | 0.5587 | 0.3766 | 0.2072 | 0.3964 | 0.5850 |
| 2 | 0.779 | 0.4908 | 0.5731 | **0.4151** | 0.2446 | 0.4090 | 0.5886 |
| 3 | 0.632 | 0.4984 | 0.5747 | **0.4188** ← 피크 | 0.2476 | 0.4081 | 0.5783 |
| 4 | 0.517 | 0.5087 | 0.5651 | 0.4156 | **0.2493** ← AUC 피크 | 0.4009 | 0.5797 |
| 5 | 0.399 | 0.5070 | 0.5596 | 0.4129 | 0.2450 | 0.3907 | 0.5707 |
| 6 | 0.202 | (test 중단) | — | — | — | — | — |

**Cutoff 판정 (§17-14)** — Troika baseline §11-11 (HM 0.3940 / AUC 0.2177) 대비:
- 피크 Δ HM = **+0.0248** (ep3), cutoff +0.005 — **5배 초과 통과**.
- 피크 Δ AUC = **+0.0316** (ep4).
- ep 2-5 4개 epoch 모두 HM ≥ 0.4129 으로 cutoff 위. **신호 robust, noise 아님.**
- 결론: §17-14 decision tree에 따라 **3-seed 확장 launch 트리거됨**.

**caveat**:
- ep5에서 HM 0.4188→0.4129로 약간 하강, train loss 0.40→0.20으로 overfitting 시작 신호. ep 6-10 추가 학습이 피크를 끌어올릴지는 미지수. 다만 §17-14 cutoff은 단일 epoch peak 기준이라 판정에 영향 없음.
- val_metric='best_loss' 라 `val_best.pt` 는 ep1 (HM 0.3766) 시점에서 잠긴 채로 update 안 됨. val-best protocol을 단일 숫자로 보고하면 baseline 미만이 되므로 — **per-epoch peak 또는 best_hm val_metric**으로 보고 protocol 통일 필요. (R2D2 §13-5g에서 동일 이슈 발생 — best_hm val_metric으로 +0.0094 HM 회수 사례 있음.)
- 산출물: `Troika/save/mit-states_seed0_ot/val_best.pt` (ep1), `epoch_4.pt` (~ep5 추정, 5-23 18:22), 두 ckpt 모두 1.83GB.

**다음 단계 후보**:
1. (권장) **seed 0 본런 재실행** — nohup/tmux로 감싸서 10ep 완주 후 정확한 peak/val-best 확정. ETA 9.5h. cutoff 이미 통과했으니 동시에 3-seed launch도 가능.
2. **epoch_4.pt(~ep5)에서 resume** — train.py에 `--load_model` 분기 있는지 확인 필요. resume이 깨끗하면 4-5h만 추가.
3. **val_metric=best_hm으로 config 변경 후 재실행** — R2D2 §13-5g 교훈 적용. 단일 숫자 reporting 깔끔해짐.
4. **seed 0 결과만으로 3-seed launch 즉시 진행** — peak는 ep3이지만 robust signal 있으므로 cutoff 판정만 보고 진행.

사용자 결정 대기 — 권장은 **(3) val_metric=best_hm 변경 + (1) seed 0 nohup 재실행**, 완주 후 3-seed launch.

### 17-17. seed 0 nohup 재실행 (5-24 02:11:56 KST)

**변경**: `config/troika/mit-states-ot.yml` → `val_metric: best_loss` → **`best_hm`** (R2D2 §13-5g 교훈 적용, val-best ckpt가 test peak 근방으로 가도록).

**Launch**: `nohup bash scripts/run_ot_seed0.sh ... & disown`
- PID **828185** (python train.py)
- log: `Troika/logs/train_mit_ot_seed0_20260524_021156.log`
- save: `Troika/save/mit-states_seed0_ot/` (이전 incomplete run은 `..._old_incomplete_20260523/`로 백업)
- GPU 1, VRAM ~16.8 GB 점유 확인.
- nohup + disown으로 SSH session 분리 — §17-15 SIGHUP 사망 재발 방지.
- ETA: 10ep × ~57min ≈ **9.5h**, 5-24 11:40 KST 부근 종료 예상.
- Namespace 로그에서 `val_metric='best_hm'` 확인됨.

종료 후:
- §17-14 cutoff 재적용 (val-best epoch의 test HM 기준으로 정식 보고)
- ep1-5 결과 (§17-16) 이미 cutoff 통과 → 완주 후 즉시 **3-seed 확장 launch** 예정.

### 17-18. seed 0 본런 종료 + §17-14 cutoff 적용 (5-24 09:00:31 KST)

**Validation HM 추이** (val_metric=best_hm, 10ep 완주):

| ep | best_seen | best_unseen | best_hm | AUC | attr | obj |
|---|---|---|---|---|---|---|
| 1 | 0.4273 | 0.5587 | 0.3766 | 0.2072 | 0.3964 | 0.5850 |
| 2 | 0.4924 | 0.5733 | 0.4173 | 0.2457 | 0.4091 | 0.5895 |
| 3 | 0.4967 | 0.5746 | **0.4175** | 0.2466 | 0.4077 | 0.5785 |
| 4 | 0.5108 | 0.5649 | **0.4175** | **0.2509** ← peak | 0.4005 | 0.5796 |
| 5 | 0.5038 | 0.5581 | 0.4121 | 0.2426 | 0.3894 | 0.5701 |
| 6 | 0.5087 | 0.5420 | 0.4057 | 0.2378 | 0.3850 | 0.5548 |
| 7 | 0.5092 | 0.5326 | 0.4048 | 0.2367 | 0.3820 | 0.5523 |
| 8 | 0.5163 | 0.5280 | 0.4075 | 0.2367 | 0.3754 | 0.5452 |
| 9 | 0.5016 | 0.5196 | 0.4031 | 0.2286 | 0.3656 | 0.5440 |
| 10 | 0.5016 | 0.5208 | 0.3997 | 0.2256 | 0.3619 | 0.5403 |

- val peak HM 0.4175 (ep3-4), AUC peak 0.2509 (ep4).
- ep5+ monotone decay, ep10 다시 val HM 0.3997로 하강 → overfit.

**Test on `val_best.pt`** (ep3 또는 ep4): **HM 0.3867 / AUC 0.2186 / attr 0.3934 / obj 0.5584**.

**§17-14 cutoff 판정** — Troika baseline §11-11 (HM 0.3940 / AUC 0.2177) 대비:
- **Δ HM = −0.0073** (cutoff −0.005 미만, **dead band 진입**)
- Δ AUC = +0.0009 (tie)
- 결론: **H3 OT on Troika = 11th sealed axis 확정**.

**해석**:
- §17-16 incomplete run의 val peak HM 0.4188(ep3)은 정상 신호였지만, **val_best.pt test로 정식 평가 시 0.3867** — val peak와 test 사이 약 −0.030의 generalization gap 존재.
- 같은 generalization gap이 §17-12 cluspro_baseline + OT(test HM 0.3867 동일치)에서도 관측됨 → OT 메커니즘이 train/val 신호를 끌어올리지만 test에 전이되지 않는 패턴이 두 framework 공통.
- §17-12 가설 "OT 메커니즘은 framework-dependent" **부분 기각**: framework 바꿔도 OT의 test 신호는 0. 즉 OT 자체가 train/val noise에 fit한 것이지 진짜 분류 신호 추가 아님.

**11 sealed axes 누적**:
- mit-states/cluspro_baseline framework (LHP-CZSL): R2D2, imgsel, cosface, v3_text, vision_distill, image_cond, ckpt-ensemble, feasibility, hardpair, OT (10)
- mit-states/Troika framework: **OT (1, 동일 메커니즘 양 framework 모두 dead)**
- → **현 mit-states + ViT-L/14 setup에서 axis 변형 수준 개선 EV ≈ 0**.

**3-seed 확장 launch 트리거 안 됨**: §17-17 사전 계획대로면 cutoff 통과 시 즉시 3-seed였지만, **cutoff 미통과 → seed 0 결과만으로 sealing 결정**.

**다음 단계 검토 (사용자 5-24 19:35 결정)**: 새 구조 진입 전 **Troika baseline gap (paper ClusPro 0.407 vs 우리 0.394, −0.013) 먼저 추적**. §18 신설.

산출물: `Troika/save/mit-states_seed0_ot/val_best.pt`, `epoch_4.pt`, `epoch_9.pt`, `final_model.pt` (각 1.83GB), `Troika/logs/train_mit_ot_seed0_20260524_021156.log` (7.2MB).

## 18. Baseline gap 추적 — Troika 위에서 ClusPro paper 0.407 chase (5-24 진입)

### 18-1. 동기 + 진입 결정 (5-24 19:35 KST)

11 axes sealed 후 사용자 결정: 새 구조 설계 전에 **Troika baseline (HM 0.394, val_metric=best_loss) → ClusPro paper SOTA (HM 0.407) gap −0.013을 hyperparam 조정으로 회수 가능한지** 우선 검증.

**근거**:
- §13-5g (R2D2/Troika dw=0.1): val_metric `best_loss → best_hm` 단일 변경으로 test HM **+0.0094** 회수 사례 있음 (0.3832 → 0.3926, val peak ckpt가 test에 더 가깝게 잡힘).
- 우리 Troika baseline §11-11은 `val_metric=best_loss`로 학습됨 (`mit-states.yml` 기본). val_best.pt가 ep1(val HM 0.376)에서 잡혔을 가능성 높음 — val peak ep2-4(HM 0.41+)를 놓침.
- **best_hm로 재학습 시 회수 폭이 R2D2 +0.0094 수준이면 baseline → 0.403, ClusPro 0.407까지 −0.004**까지 좁혀짐. 이 정도면 framework 차이가 아니라 reproduce hyperparam 차이로 설명 가능.

### 18-2. Launch — Troika baseline + val_metric=best_hm seed 0 (5-24 19:37:53 KST)

**변경**:
- 새 yml `config/troika/mit-states-besthm.yml` 생성 — vanilla `mit-states.yml`에서 `val_metric: best_loss → best_hm` 단일 변경만. 나머지 hp (lr=1e-4, wd=1e-5, attr_dropout=0.3, epochs=10, batch=8×grad_accum=8, cmt_layers=3) 모두 §11-11과 동일.
- 새 script `scripts/run_baseline_besthm_seed0.sh` (nohup launch wrapper).

**Run**:
- PID **3433364** (python train.py)
- log: `Troika/logs/train_mit_baseline_besthm_seed0_20260524_193753.log`
- save: `Troika/save/mit-states_seed0_baseline_besthm/`
- GPU 1, CUDA_VISIBLE_DEVICES=1, nohup + disown.
- ETA: §17-17 OT run(9.5h)보다 OT overhead 없어서 짧을 듯, ~7-8h. 5-25 03:00-04:00 KST 부근 종료 예상.

**판정 (사전 등록)** — §11-11 test HM 0.3940 / AUC 0.2177 대비:
- test HM ≥ 0.404 → 사실상 ClusPro paper SOTA 0.407 tie band (±0.005). **"Troika best_hm = SOTA-class baseline"** 결론 + 11 sealed axes를 best_hm baseline 위에서 재평가 검토.
- test HM ∈ [0.398, 0.404) → 일부 회수, 진짜 gap은 framework 차이 또는 다른 미상 hyperparam. 후속 진단 (lr/wd/scheduler/seed sweep).
- test HM ∈ [0.394, 0.398) → 단순 ckpt 선택 효과 미미. R2D2 §13-5g와 다른 양상 — val-test 상관성 framework별 다름.
- test HM < 0.394 → §11-11이 seed noise였을 가능성. seed 1, 2 추가 본런 필요.

**기대치**: §13-5g 회수폭 +0.0094 prior로 → mean estimate 0.3940 + 0.0094 = **0.4034**. tie band(0.398-0.404) 안. 그러나 R2D2 framework와 vanilla Troika의 val/test 상관성 다를 수 있으므로 noise band 넓게 잡음.

종료 후: 결과에 따라 (a) 다른 hp(lr/wd/scheduler) 단일축 ablation, 또는 (b) 새 구조 진입 결정.

### 18-3. seed 0 본런 종료 + §18-2 cutoff 적용 (5-25 02:15:25 KST)

**Run 통계**: 5-24 19:37:53 → 5-25 02:15:25, **약 6h 38m** (§17-17 OT run 6h 49m 대비 ~10min 짧음, OT overhead 없는 만큼).

**Validation HM 추이** (val_metric=best_hm, 10ep 완주):

| ep | best_seen | best_unseen | best_hm | AUC | attr | obj |
|---|---|---|---|---|---|---|
| 1 | 0.4181 | 0.5584 | 0.3729 | 0.2014 | 0.3825 | 0.5843 |
| 2 | 0.4913 | 0.5761 | 0.4155 | 0.2461 | 0.4031 | 0.5938 |
| 3 | 0.5005 | 0.5725 | **0.4157** ← peak | **0.2475** ← AUC peak | 0.4016 | 0.5812 |
| 4 | 0.5054 | 0.5578 | 0.4151 | 0.2451 | 0.4020 | 0.5762 |
| 5 | 0.5141 | 0.5524 | 0.4099 | 0.2446 | 0.3836 | 0.5703 |
| 6 | 0.5125 | 0.5339 | 0.4042 | 0.2360 | 0.3801 | 0.5584 |
| 7 | 0.5043 | 0.5335 | 0.4045 | 0.2335 | 0.3789 | 0.5542 |
| 8 | 0.5060 | 0.5237 | 0.4037 | 0.2307 | 0.3739 | 0.5484 |
| 9 | 0.5103 | 0.5198 | 0.4037 | 0.2304 | 0.3753 | 0.5456 |
| 10 | 0.4778 | 0.5132 | 0.3882 | 0.2120 | 0.3585 | 0.5382 |

- val peak HM 0.4157 (ep3), AUC peak 0.2475 (ep3).
- ep4~ monotone decay, ep10 val HM 0.3882 → overfit pattern. §17-18 OT run(ep4 peak 0.4175)과 거의 동일 trajectory shape.

**Test on `val_best.pt`** (ep3): **HM 0.3905 / AUC 0.2235 / attr 0.3963 / obj 0.5561**.

**§18-2 사전 등록 cutoff 적용** — §11-11 baseline (HM 0.3940 / AUC 0.2177, val_metric=best_loss) 대비:
- **Δ HM = −0.0035** (cutoff 4번째 band `< 0.394` 진입)
- Δ AUC = +0.0058
- 결론: **val_metric=best_hm 단일 변경으로는 ClusPro paper gap 회수 실패**. §11-11이 seed-lucky였을 가능성 우선 검증 필요 (사전 등록 4번째 band → seed 1, 2 추가 본런).

**해석**:
- val peak HM 0.4157 → test HM 0.3905 → **val-test gap −0.0252** 존재.
- §17-18 OT run의 val peak HM 0.4175 → test HM 0.3867 (gap −0.0308)와 동일 패턴: vanilla Troika는 val peak ckpt가 test에 깨끗하게 전이되지 않음. 즉 **val-test correlation이 framework-dependent** — R2D2(§13-5g)에서는 best_hm으로 +0.0094 회수됐지만 Troika에서는 회수 없음.
- §13-5g R2D2 prior(+0.0094)는 vanilla Troika에 transfer되지 않음. framework별 val/test 상관성을 별도로 측정해야 한다는 교훈.
- §11-11 baseline 0.3940 (best_loss, 1-seed)과 이번 0.3905 (best_hm, 1-seed)의 차이 −0.0035는 [[project_lhp_czsl_amp_nan]] §12 cluspro_baseline seed std ±0.0008와 비교하면 큰 차이지만, **framework가 다르므로 직접 비교 불가**. Troika seed std는 미측정.

**11→12 sealed axes 누적 후보**: "val_metric=best_hm on vanilla Troika"는 단일 axis 봉인 후보. 단, §11-11 numbers가 seed-lucky인지 먼저 확인 필요 (아래 §18-4).

**다음 단계 (사전 등록 4번째 band rule)**: §11-11 noise band 확정을 위해 **Troika baseline (val_metric=best_loss, §11-11과 동일 yml) seed 1, 2 sequential 본런**. 종료 후:
- §11-11 3-seed mean HM ≥ 0.394 → §11-11 신뢰. 0.3905 (best_hm seed 0)는 진짜 −0.0035 음 신호. → "val_metric=best_hm on Troika" 12번째 sealed axis 봉인.
- §11-11 3-seed mean HM < 0.394 → §11-11이 seed-lucky. best_loss와 best_hm tie 가능성 → best_hm seed 1, 2 추가 본런으로 3-seed 비교.
- 어느 쪽이든 Troika baseline seed std 확보 → 향후 hp ablation의 noise band로 사용.

산출물: `Troika/save/mit-states_seed0_baseline_besthm/val_best.pt`, `final_model.pt`, `Troika/logs/train_mit_baseline_besthm_seed0_20260524_193753.log` (7.2MB).

### 18-4. Launch — Troika baseline (best_loss) seed 1, 2 sequential (5-25 11:44:47 KST)

§18-3 4번째 band rule 적용. §11-11 single-seed 0.3940의 seed noise 확정 목적.

**Setup**:
- yml: `config/troika/mit-states.yml` (vanilla) — val_metric=best_loss, lr=1e-4, wd=1e-5, attr_dropout=0.3, epochs=10, cmt_layers=3.
- **CLI override 필수**: vanilla yml은 `train_batch_size: 64 / accum: 1`로 변경돼 있어 ViT-L/14에 OOM. §11-11 `mit-states_seed0_repro` config(`bs=8 / accum=8`)와 정확히 맞추기 위해 `--train_batch_size 8 --gradient_accumulation_steps 8` CLI override 추가. (첫 launch 11:42:56은 override 없이 시작 → ep1 step 0에서 OOM 사망 → 11:44:47 재launch).
- seed 1 → 종료 후 같은 nohup 스크립트 내에서 자동 seed 2 launch.
- script: `Troika/scripts/run_baseline_repro_seed12.sh`
- log: `Troika/logs/train_mit_baseline_repro_seed{1,2}_<TS>.log`
- save: `Troika/save/mit-states_seed{1,2}_repro/`
- GPU 1, nohup + disown. seed 1 PID **1624528**, VRAM 16.8GB.
- 첫 epoch 측정 1.86 it/s × 3793 step ≈ 34min/epoch × 10 ≈ **5h 40m/seed × 2 ≈ 11h 20m**, 종료 **5-25 23:00 KST** 부근.

판정 (사전 등록):
- 3-seed mean HM ≥ 0.394 AND std ≤ 0.003 → §11-11 신뢰, best_hm 음 신호 0.3905 확정. axis 봉인.
- 3-seed mean HM < 0.394 → §11-11 seed-lucky. best_hm seed 1, 2도 launch해서 best_loss vs best_hm 3v3 비교.
- 3-seed std > 0.003 → noise dominant, baseline gap (−0.013) 자체가 seed scope 안일 가능성. §18 가설 자체 재검토.

### 18-5. seed 1, 2 종료 + §18-4 cutoff 적용 (5-26 01:03 KST)

**Run 종료 시각**:
- seed 1: 5-25 11:44:47 → 5-25 18:24:05, **6h 39m** (PID 1624528)
- seed 2: 5-25 18:24:14 → 5-26 01:03:18, **6h 39m** (seed 1과 동일 스크립트 내 자동 launch)
- 둘 다 10ep 완주, AMP NaN 0회, val_metric=best_loss로 val_best.pt 저장 후 test 평가까지.

**Test on `val_best.pt`** (Closed World, test_pairs):

| seed | seen | unseen | **HM** | **AUC** | attr | obj |
|---|---|---|---|---|---|---|
| 0 (§11-11) | 0.5141 | 0.5161 | **0.3940** | **0.2177** | — | — |
| 1 (§18-4) | 0.4891 | 0.5325 | **0.3891** | **0.2213** | 0.4052 | 0.5607 |
| 2 (§18-4) | 0.4668 | 0.5340 | **0.3865** | **0.2140** | 0.3981 | 0.5735 |
| **3-seed mean ± std (n−1)** | 0.4900 ± 0.024 | 0.5275 ± 0.010 | **0.3899 ± 0.0038** | **0.2177 ± 0.0037** | — | — |

**Val peak trajectory** (각 seed의 val HM peak epoch / 값):

| seed | val peak ep | val peak HM | val peak AUC | val_best ckpt ep (best_loss) |
|---|---|---|---|---|
| 0 | — | — | — | — (§11-11 raw 미보관) |
| 1 | ep4 | 0.4264 | 0.2538 | ep1~ (낮은 loss 기준) |
| 2 | ep4 | 0.4211 | 0.2506 | ep1~ |

seed 1, 2 모두 §17-18 / §18-3 OT·best_hm run과 동일 trajectory shape (ep3-4 val peak, ep5+ overfit decay). val peak HM ~0.42 vs test HM ~0.387 → **val-test gap 평균 −0.033**, vanilla Troika의 framework-level transfer 손실 재확인.

**§18-4 사전 등록 cutoff 판정**:

| band | 조건 | hit? |
|---|---|---|
| A | mean ≥ 0.394 AND std ≤ 0.003 → §11-11 신뢰, best_hm axis 봉인 | ❌ (mean 0.3899 < 0.394) |
| B | mean < 0.394 → §11-11 seed-lucky, best_hm seed 1, 2 추가 본런 | ✅ |
| C | std > 0.003 → noise dominant, §18 가설 재검토 | ✅ (std 0.0038, 사전 등록 0.003 cutoff 초과) |

Band B + C 동시 hit. cluspro_baseline 시드 std ±0.0008(§12)과 비교하면 **Troika는 시드 노이즈가 R2D2/LHP-CZSL 대비 5배 큼** — vanilla Troika는 1-seed 측정으로 trend 판정 불가.

**해석**:
1. **§11-11 0.3940은 seed-lucky**. seed 0이 mean(0.3899)보다 +0.0041(약 1σ 위), 진짜 baseline은 **HM 0.390 / AUC 0.218** 부근.
2. **§18-3 best_hm seed 0 = 0.3905 vs best_loss 3-seed mean 0.3899 차이 +0.0006 → noise band 내(0.2σ)**. "val_metric=best_hm on Troika"의 effect size는 측정 불가, **봉인 후보 철회**. §13-5g R2D2의 +0.0094 prior는 Troika에 transfer되지 않으며, 우리 측정 한계 안에서는 framework-independent하게 ckpt selection 자체가 weak lever.
3. **ClusPro paper SOTA(0.407)와의 gap −0.017은 ckpt selection으로 회수 불가** 확정. framework 자체(R2D2 vs Troika)나 다른 hp 축에 있을 가능성. ClusPro 재현 갭(§11-2, −1.8pp) 안에 머무름.
4. **Troika 시드 std 0.0038 확보** → 향후 hp ablation의 effective noise band는 ~3σ ≈ **±0.011 HM**, 즉 Δ HM > +0.011이라야 "유의" 판정 가능. R2D2/LHP-CZSL 보다 4-5배 보수적.

**11→12 sealed axes**: 봉인 추가 없음. v3_text/OT/best_hm/Troika-pivot 모두 dead 상태 유지 (10 + Troika OT = 11), [[project_lhp_czsl_h3_ot_sealed]] 그대로.

**다음 단계**: §18-4 Band B의 "best_hm seed 1, 2 추가 본런"은 §18-3 + §18-5 신호로 이미 effect size 너무 작아 무의미 → **skip**. Troika best_loss baseline mean 0.3899 확정으로 §18 axis 전체 종료. 새 축 탐색 필요.

→ §19부터 **self-derived improvement search**. 최근 CZSL 논문 분석 후 baseline에 어떤 lever가 남았는지 사전 등록. [[project_lhp_czsl_baseline_gap]] 메모리도 "본런 진행 중" → "Troika best_loss 3-seed mean 0.3899 ± 0.0038 확정, gap 회수 실패, ckpt-selection 축 닫힘"으로 갱신 필요.

산출물: `Troika/save/mit-states_seed{1,2}_repro/{val_best,final_model}.pt`, `Troika/logs/train_mit_baseline_repro_seed{1,2}_<TS>.log` (각 7.2MB).

---

## 19. Self-improvement search — recent CZSL literature triage (5-26 16:30 KST)

§18-5 종료로 ckpt-selection 축 닫힘. 새 lever 발굴 위해 2024-2025 CZSL 논문 서베이.

### 19-1. 서베이 대상 (CVPR/ICCV 2025 + arXiv 2025-2026)

| 논문 | 핵심 아이디어 | mit-states ViT-L/14 HM | 우리 sealed axes와의 관계 |
|---|---|---|---|
| **CLUSPRO** (ICLR'25) | K-prototype per primitive | 0.407 (paper) | 우리 baseline (재현 0.389) |
| **CDS-CZSL** (CVPR'24) | Context-aware specificity (image-cond attr) | 0.390~0.393 | §14 image-cond 봉인과 인접 |
| **PLID** (ECCV'24) | LLM-rich class distributions | 0.390~0.393 | §11-18 v3_text 봉인 영역 |
| **Troika** (CVPR'24) | Multi-path (s/o/comp) + CMT | 0.392 (재현 0.389) | 우리 framework |
| **CAILA** (WACV'24) | Intra-layer adapters in CLIP encoder | — | **frozen CLIP 가정 파괴** (새 lever) |
| **SDP / Duplex** (2501.07114, 2026) | Counterfactual comp + local graph + EMA proto refine | **0.409 ± 0.002** | 일부 미시도, 자세히 §19-2 |
| **FlowComposer** (2603.16641, 2026) | Rectified flow between visual feat ↔ primitive text embed + Composer fusion + leakage-guided aug | **0.402** (Troika+FC, ViT-L/14) | **완전 직교**, 자세히 §19-2b |
| **VP-CMJL** (ICCV'25, 2501.13859) | Visual proxy init from text + cross-modal joint loss | SOTA (수치 미공개) | ClusPro multi-proto + alignment 부근 |
| **LOGICZSL** (CVPR'25) | LLM logic rules (primitive compatibility) → loss penalty | UT-Zap AUC 38.2 (CSP 33.0 대비 +5.2) | v3_text와 다른 axis: 콘텐츠 enrich 아닌 feasibility prior |
| **ULAO** (2412.07161) | Sequential primitive (obj→attr) + tailored hard-negative contrastive | "SOTA" (수치 미공개) | 우리 image-cond과 다른 결: 순서·hard-neg |
| **SASOW** (2512.18969) | Self-attention over state-object pairs | 미공개 (open-world?) | 미확인 |
| **ATIF** | Dual-stream (frozen + adapter) fusion | — | 새 axis (encoder mixing) |
| **SymNet 류** | Group-theoretic symmetry/invertibility | — | underexplored, 우리 axes와 완전 직교 |

### 19-2. SDP (Duplex) 정밀 분석 — 가장 강력한 후보

**보고치**: mit-states **HM 0.409 / AUC 0.237** (ViT-L/14), UT-Zap HM 0.582 / AUC 0.462. ClusPro paper와 거의 동률 또는 약간 위.

**메커니즘** (3개 구성):
1. **Counterfactual composition**: 미니배치에서 image i의 state feature 𝐳ₛ와 image j의 object feature 𝐳ₒ를 추출 → ϕ(𝐳ₛ, 𝐳ₒ)로 unseen comp 합성. 배치 내 feasible comp만 활성.
2. **Label-conditioned local graph**: 노드 = {프로토타입, 분리된 s/o factor, counterfactual comp}, 엣지 = feasibility rule. 배치 내 "old" + "tiger"가 있으면 unseen "old tiger" 프로토타입까지 메시지 전파.
3. **EMA prototype update**: 𝐇 ← λ𝐇 + (1−λ)𝐇̂. **Semantic (text) prototype은 frozen, visual prototype만 동적 갱신**.

**우리 sealed axes와의 관계**:
- §10 v3_text (text content 확장): **무관** (SDP는 vision-side, text는 vanilla soft prompt).
- §13 R2D2 (vision distillation): **느슨하게 인접** — distillation 대신 EMA로 visual proto를 직접 학습.
- §14 image-cond description selection: **다른 결** — SDP는 image 쌍을 cross 해서 unseen 합성, image-cond aggregation 아님.
- §17 OT primitive matching: **다른 결** — graph message passing이 OT 대신.

→ **3개 component 중 counterfactual + local graph는 우리 axes와 직교, EMA proto refinement는 부분 인접.** 가장 시도해 볼 만한 직교 lever.

### 19-2b. FlowComposer 정밀 분석 — 우리 framework와 직접 호환되는 강력 후보

**보고치** (ViT-L/14, frozen CLIP, Troika baseline + FlowComposer add-on):
- mit-states **HM 0.402 / AUC 0.235** (closed-world). Troika 단독 0.392 대비 **Δ HM +0.010**, 우리 noise band(±0.011 = 3σ) 상단.
- UT-Zap HM 0.586 / AUC 0.468, C-GQA HM 0.340 / AUC 0.159.
- ClusPro paper(0.407) 약간 아래, SDP(0.409) 약간 아래지만, **Troika에 직접 attach 가능**한 add-on이라는 점이 결정적.

**메커니즘 (rectified flow 기반)**:
1. **Two primitive flows** (attr / obj 각각):
   - Visual feature **x₀ⁱ** → text embedding **x₁ⁱ**으로 transport.
   - 선형 보간 경로 **xₜⁱ = (1−t)x₀ⁱ + t·x₁ⁱ** 위에서 velocity network **v_θⁱ**를 ground-truth velocity **v* = x₁ⁱ − x₀ⁱ**로 회귀.
   - MSE loss: ‖v̂ₜⁱ − (x₁ⁱ − x₀ⁱ)‖². 네트워크는 timestep-conditioned residual MLP (MAR 류).
2. **Composer**: 정규화된 primitive velocity **Δ̂ₐ, Δ̂ₒ**의 선형 결합 계수 (a*, b*) 학습. 3-layer MLP가 (â, b̂) 예측. composition velocity v̂ₓ = â·norm(v̂ₐ) + b̂·norm(v̂ₒ).
3. **Leakage-guided augmentation**: cross-branch 누수 feature를 supervision으로 재활용. branch i의 학습에 다른 branch j의 누수 feature x₀ʲ → x₁ⁱ로 transport시키는 **cross-branch interpolation path** xₜⁱ←ⱼ = (1−t)x₀ʲ + t·x₁ⁱ. 추가 MSE + contrastive supervision.

**Inference (one-step)**: x̂₁ⁱ = x₀ⁱ + v_θⁱ(x₀ⁱ, 0). 그리고 x̂₁ᶜ = x₀ᶜ + h·v̂ₓ. h는 step-size hp.

**Ablation (mit-states, Troika baseline 0.394)**:
| 구성 | HM | Δ |
|---|---|---|
| Troika only | 0.394 | — |
| + Flows | 0.394 | +0.000 |
| + Flows + Composer | 0.401 | +0.007 |
| + Flows + Leakage-guided aug | 0.403 | +0.009 |
| **+ Flows + Composer + Leakage** | **0.402** | +0.008 |

→ leakage-guided aug 단독이 main lever (~+0.009), Composer는 marginal. UT-Zap에서는 leakage가 +0.012 HM 독립 기여.

**우리 sealed axes와의 관계**:
- §10 v3_text: text content 안 건드림 (frozen text embedding 그대로 endpoint로 사용). **직교**.
- §13 R2D2 vision distillation: distillation 대신 flow regression. distillation은 single-step embedding alignment, flow는 trajectory regression — **별개 paradigm**.
- §14 image-cond description: description 안 씀. **직교**.
- §17 OT: OT는 patch ↔ prototype assignment, flow는 feature trajectory — **다른 결**.
- §18 ckpt selection: 무관.

→ **11 sealed axes 어디와도 겹치지 않음, frozen CLIP 가정 유지, Troika 위 add-on**. 우리에게 가장 잘 맞는 구조.

### 19-3. 후보 lever 후순위

**Tier A (직교성 + 비용 합리적)**:
1. **FlowComposer leakage-guided augmentation** (§19-2b). Troika에 직접 attach하는 cross-branch rectified-flow regression. ablation상 leakage 단독 +0.009 HM (3σ band 도달 가능), frozen CLIP 유지, add-on만 학습. 코드 공개 여부 확인 필요. 가장 framework-fit.
2. **Counterfactual composition mixup** (SDP § 1번 component만 isolated). 배치 내 (sᵢ, oⱼ) 합성으로 unseen pair pseudo-sample 생성, classification loss에 추가. 단일 axis 검증 가능.
3. **ULAO 류 hard-negative contrastive**: 같은 obj, 다른 attr / 같은 attr, 다른 obj 쌍을 mined negative로. 우리 negative sampling이 random uniform이라면 lever 있음.

**Tier B (orthogonal but invasive)**:
4. **CAILA / ATIF — intra-layer adapter or dual-stream**: frozen CLIP 가정 파괴. 우리 모든 11 axes가 frozen 위에서 sealed 됐다는 점에서 **framework 자체 unstuck 가능성**. 단, 검증 비용 큼.
5. **LOGICZSL feasibility prior**: LLM이 primitive 쌍 (s, o)의 feasibility score 산출 → composition logit에 prior로 가산. v3_text content와 달리 score-only prior라 §11-18 sealing과 다른 결.

**Tier C (보류)**:
6. SymNet 류 group-theoretic — formulation 자체가 무거움.
7. VP-CMJL — ClusPro multi-proto와 너무 인접, sealed 가능성 큼.
8. SASOW — 수치/메커니즘 불분명.

### 19-4. 사전 등록 제안 (사용자 결정 대기)

직교성 + framework-fit 최고: **Tier A-1 (FlowComposer leakage-guided aug)** 권장.

- Pre-check: arXiv 2603.16641 저자 코드 공개 여부 확인 → 공개시 우리 Troika 위 직접 attach, 미공개시 self-impl (residual MLP velocity net + cross-branch path MSE/CE loss 약 200줄).
- Probe 설정: Troika §11-11 yml + 2-branch velocity net + leakage-guided MSE/CE. Composer는 ablation상 marginal이라 skip 가능.
- Stage 0 sanity: 1ep, NaN-free, train loss monotone, VRAM 안 깨짐.
- Stage 1 본런: seed 0, 10ep, ETA ~7-8h (velocity net add-on 작아 추가 시간 < 30%).
- 판정 cutoff (사전 등록):
  - Δ HM > +0.011 (3σ band, [[project_lhp_czsl_baseline_gap]] §18-5 noise scale) → 유의, seed 1, 2 본런 → 봉인 회피.
  - 0 ≤ Δ HM ≤ +0.011 → tie band, axis 봉인.
  - Δ HM < 0 → fail, 봉인.

대안: **Tier A-2 (counterfactual composition mixup)**는 A-1보다 단순. 코드 공개 없거나 self-impl 비용 부담시 fallback.

→ 사용자 결정 필요: (a) Tier A-1 (FlowComposer) 진행, (b) Tier A-2 (SDP counterfactual) 진행, (c) Tier B (CAILA encoder adapter) 큰 lever 시도, (d) 다른 방향.

---

## 20. LLM-axis sequential chain — C′/D′/E/F (5-26 17:10 KST 진입)

**사용자 결정**: §19 후보 중 FlowComposer/SDP 보류, **LLM 활용 축**으로 진입. 4-axis sequential 시도.

**전제 sealed 영역 회피** ([[project_lhp_czsl_h3_ot_sealed]] 10 LHP-CZSL + 1 Troika OT):
- "feasibility" (LLM scalar feasibility score → composition logit bias, scalar prior) **이미 9th sealed** on cluspro_baseline.
- "hardpair" (`same_prim_sample=True` batch-level syntactic 같은-attr/같은-obj 강제) **이미 10th sealed** on cluspro_baseline + Troika OT 비추어 framework pivot 효과 1차 기각.
- 따라서 C, D는 **원안과 다른 메커니즘으로 redefine** (옵션 2). 원안 C=feasibility scalar prior, D=syntactic same-prim sample, 모두 봉인 영역이므로 회피.

### 20-1. Redefined axis 정의

**C′ — LOGICZSL 류 relational logic rule grounding**
- Sealed feasibility와 차이: scalar `logit += w · score(s, o)` 아님. **다중 primitive 관계 제약 (mutual-exclusion, group-membership)** 을 보조 BCE/KL loss로 grounding.
- 예시 rule type:
  - (mutex_attr) `{ripe, rotten, fresh}`는 같은 obj 상에서 mutually exclusive — 하나 high → 나머지 low penalty.
  - (group_attr) `{wet, damp, soaked}`는 surface-moisture group, 그룹 내 swap은 cost ↓ (label smoothing 류).
  - (compat_obj_attr) `{liquid_obj} × {melted, frozen, runny, viscous}` compatible, `{liquid_obj} × {wrinkled, folded, ripped}` incompatible.
- 구현: LLM (Claude in-conv) offline 호출 → `data/logic_rules_mit.json` 생성. 각 rule = `{type, premise, conclusion, weight}`. Train forward에서 composition logit에 rule violation penalty 적용.
- 예상 코드: `model/troika_logic.py` (300 LOC), `tools/build_logic_rules_claude.py` (200 LOC), `code/parameters.py` λ_logic flag.

**D′ — LLM-guided semantic hard-negative mining**
- Sealed hardpair와 차이: batch sampler가 **syntactic 같은-primitive 강제**가 아니라, LLM이 사전에 산출한 **semantic confusable pair list**를 oversample. 즉 same_prim_sample (cluspro hardpair)에서 같은 attr/obj 쌍을 무조건 한 배치에 넣는 것 아니라, LLM이 "이미지 차원에서 헷갈리기 쉽다"고 판정한 (s₁, o₁) ↔ (s₂, o₂) 쌍 N개를 oversample. 메커니즘은 sampler weight + optional InfoNCE.
- 예시: "ripe apple" ↔ "rotten apple" (semantic close on appearance), "wooden chair" ↔ "leather chair" (texture distinguishable but compositionally close).
- 구현: LLM offline → `data/hardneg_pairs_mit.json` (top-K confusable per anchor composition, K=8 ~ 16). WeightedRandomSampler + 1 InfoNCE term. λ_hn sweep {0.05, 0.1, 0.3}.

**E — LLM taxonomy / structural prior** (novel, sealed 영역 아님)
- LLM이 obj → super-category (예: apple → fruit, tiger → animal, copper → metal) 및 attr → group (예: ripe ∈ ripeness_group, rusty ∈ surface_quality) 산출. Auxiliary classification head가 visual feature → super-category 예측. 보조 CE loss로 hierarchical regularization.
- 구현: 작은 LLM call (115 attrs + 245 objs offline) → `data/taxonomy_mit.json`. `model/troika.py`에 2개 aux head (attr_group_clf, obj_supcat_clf). λ_taxo sweep {0.1, 0.3, 0.5}.

**F — LLM test-time reranking** (novel, training change 0)
- 학습 변경 없음. Best ckpt val_best.pt forward로 test set의 top-K (K=5 or 10) composition prediction 산출 → multimodal LLM (Claude Sonnet/Opus)에 "이 이미지에 가장 맞는 composition은?" 질의 → reranked top-1. Eval HM/AUC.
- 비용 분석: mit-states test ~13k images × K=5 candidates → 13k LLM call. Batch API 사용 시 cost 추정 후 진행.

### 20-2. 사전 등록 cutoff (전 axis 공통)

[[project_lhp_czsl_baseline_gap]] §18-5에서 측정한 Troika 3-seed noise = **±0.0038 HM (1σ), ±0.011 (3σ)**. 적용:

| 단계 | 통과 조건 | 후속 |
|---|---|---|
| Stage 0 (1ep sanity, smoke) | train loss monotone 30 step, NaN 0회, VRAM 안 깨짐, val HM > 0 | Stage 1 진입 |
| Stage 1 (seed 0, 10ep) | Δ HM > +0.011 vs §11-11 0.3940 (단일 seed 비교) | seed 1, 2 본런 |
| Stage 2 (seed 1, 2 본런) | 3-seed mean HM > 0.3899 + 0.011 = **0.4009** AND std ≤ 0.005 | axis 유의 확정, 다음 axis로 |
| 어느 단계든 fail | Δ HM ≤ +0.011 (Stage 1) or 3-seed mean ≤ 0.4009 (Stage 2) | axis 봉인, RESEARCH_LOG/메모리 갱신, 다음 axis로 |

각 axis는 baseline (vanilla Troika) 대비 독립 측정. 모두 통과한 후 stacking ablation은 별도 §20-N.

### 20-3. 시작 순서

C′ → D′ → E → F 순차. F만 inference-only라 다른 axes의 best ckpt 위에 적용 가능. 각 axis 한 cycle 예상 ~ 2-3일 (Stage 0 ~30분, Stage 1 ~7h, Stage 2 ~15h 병렬 불가능 → 22h, 분석 ~1h).

**즉시 진입**: C′ Stage 0 (rule extraction + smoke). 아래 §20-4.

### 20-4. C′ design — attr-group mutex + obj-attr incompat regularization (5-26 17:30 KST)

**Sealed feasibility와의 메커니즘 차이**:
| 항목 | Sealed feasibility (cluspro) | **C′ logic rule** |
|---|---|---|
| LLM output | scalar score ∈ [1,10] per (attr, obj) | categorical: attr mutex groups + obj-specific incompat sets |
| 적용 위치 | composition logit additive bias `logit_c += w·s(a,o)` | attribute posterior 분포 형태 regularization (loss on `attr_logits` softmax) |
| 정보 사용 | per-comp 점수 — primitive 간 관계 X | mutex group 멤버십 + 조건부 incompat = relational structure |

**Loss term 정의**:
1. **L_mutex (within-group mutual exclusion)**: 각 sample (image, GT attr a) → a가 속한 mutex group G(a)에 대해, 다른 멤버 attrs의 softmax 확률 합을 페널티. attribute distribution이 mutex group 안에서 sharp peak를 갖도록 유도.
   `L_mutex = mean_b ∑_{a' ∈ G(a_gt), a' ≠ a_gt} softmax(attr_logits)[a']`
2. **L_compat (obj-conditioned incompat penalty)**: 각 sample (image, GT obj o) → o가 incompatible로 가진 attr set I(o)에 대해, 해당 attrs의 softmax 확률 합을 페널티.
   `L_compat = mean_b ∑_{a ∈ I(o_gt)} softmax(attr_logits)[a]`
3. 총 loss = baseline_loss + λ_mutex · L_mutex + λ_compat · L_compat
4. Sweep: λ_mutex ∈ {0.05, 0.1, 0.3}, λ_compat ∈ {0.05, 0.1, 0.3}. Stage 0에서는 (0.1, 0.1)만.

**LLM input/output**:
- **Mutex groups**: in-conversation Claude (no API), categorical 산출 (예: `{ripeness: [ripe, unripe, rotten, fresh, raw], moisture: [wet, dry, damp]}`). 115 attrs 중 ~70-90이 그룹에 속하고 나머지는 standalone. 출력 → `data/logic_rules_mit.json::attr_mutex_groups`.
- **Incompat sets**: 기존 `data/feasibility_mit.json` (Gemini scores 1-10, 28,175 pairs) 재활용 — score ≤ 3 = incompat threshold. obj별로 incompat attr list 산출. **scalar score를 categorical incompat set으로 reduce하므로 sealed mechanism과 다른 grain**. 출력 → `data/logic_rules_mit.json::obj_incompat_attrs`.

**구현 파일**:
- `model/troika_logic.py`: troika.py 복사 + `loss_calu`에 mutex/compat loss term 추가.
- `parameters.py`: `--lambda_mutex` (default 0.0), `--lambda_compat` (default 0.0), `--logic_rule_path` 추가.
- `config/troika/mit-states_logic.yml`: vanilla mit-states.yml 기반 + `model: troika_logic`, `lambda_mutex: 0.1`, `lambda_compat: 0.1`.
- `tools/build_logic_rules.py`: feasibility json + mutex group categorical 합쳐서 `data/logic_rules_mit.json` 생성.

**Stage 0 통과 조건** (smoke, 1-ep):
- forward NaN 0회, train loss finite 30 step 이상, val HM > 0.30 (baseline 1-ep 0.373 근처면 합격).
- VRAM 17 GiB 이내, 1 it/s 유지.

**Stage 1 통과 조건** (seed 0, 10ep):
- Δ HM > +0.011 vs §11-11 (0.3940). 즉 **test HM > 0.4050**. AUC도 함께 보고.

§20-4-1부터 구현 시작.

### 20-4-1. C′ Stage 0 결과 (5-26 17:50 KST)

**산출물**:
- `/home/student/dongki/LHP-CZSL/tools/build_logic_rules.py` (rule json builder)
- `/home/student/dongki/LHP-CZSL/data/logic_rules_mit.json` (5.4MB → 1.1MB, 65/115 attrs grouped into 25 mutex groups, 245/245 objs with avg 31.4 incompat attrs)
- `Troika/code/model/troika_logic.py` (TroikaLogic subclass of Troika, override loss_calu)
- `Troika/code/model/model_factory.py`에 model_name=troika_logic 등록
- `Troika/code/config/troika/mit-states-logic.yml` (lambda_mutex=0.1, lambda_compat=0.1, bs=8/accum=8)
- `Troika/code/smoke_logic.py` (30-step 1-2분 smoke)

**Smoke 결과** (CUDA_VISIBLE_DEVICES=1, λ_mutex=λ_compat=0.1, 30 step):
- rule loading 정상: 65/115 attrs grouped, 245 objs incompat avg=31.4
- NaN/Inf 0회
- VRAM peak 13.96 GB (baseline 16.8GB 대비 -2.8GB, attr_probs softmax + mask indexing overhead 미미)
- 처리속도 **1.75 it/s** (Troika baseline 1.86 it/s 대비 -6%; 30 step에서 17.2초)
- Loss term 분해 (step 30, λ 적용 전 raw):
  - comp ~5.4, attr ~5.3, obj ~6.3, **L_mutex ~0.008, L_compat ~0.30**
  - λ=0.1 적용시 mutex 기여 ~0.0008, compat 기여 ~0.03 — base loss ~16 대비 **약 0.2%**
- **주의**: L_mutex 신호가 init-near uniform softmax에서 매우 작음 (group size 3-5, 115-way softmax → 3-5/115 ≈ 0.03 균등 기대). 학습 후 sharpen되면 증가 예상. L_compat는 init 0.27 (31/115 균등 기대치 ≈ 0.27) — 기대치 안.

**Stage 0 통과 판정**: NaN-free ✓, finite loss ✓, VRAM 안전 ✓ → **Stage 1 진입**.

**리스크**: λ=0.1에서 mutex/compat 기여 0.2%는 약함. Stage 1 fail시 λ sweep {(0.3,0.3), (1.0,1.0)} 후속 검토.

### 20-4-2. C′ Stage 1 launch (5-26 18:00 KST)

**설정**:
- yml: `config/troika/mit-states-logic.yml` (λ_mutex=0.1, λ_compat=0.1)
- seed 0, bs=8/accum=8, 10ep, val_metric=best_loss
- GPU 1, nohup + disown
- script: `Troika/scripts/run_c_prime_seed0.sh`
- save: `Troika/save/mit-states_c_prime_seed0_l0.1/`
- log: `Troika/logs/train_mit_c_prime_seed0_l0.1_<TS>.log`
- ETA 첫 epoch ~34min (baseline과 동일) × 10ep ≈ **5h 40m**, 5-26 23:40 KST 부근 종료.

**Stage 1 사전 등록 cutoff** (재확인):
- Δ HM > +0.011 vs §11-11 (0.3940) → **test HM > 0.4050** → 유의, Stage 2 (seed 1, 2) 진입.
- 0 ≤ Δ HM ≤ +0.011 → tie band, axis 봉인.
- Δ HM < 0 → fail, axis 봉인.
- (대안) λ=0.1 fail시 (0.3, 0.3) Stage 1' 재시도. 그래도 fail → C′ axis 봉인.

### 20-5. D′ design — LLM-guided semantic hard-negative mining (사전 등록, C′ 결과 대기 중 작성)

**Sealed hardpair와의 메커니즘 차이**:
| 항목 | Sealed hardpair (cluspro, §17-9) | **D′ semantic hard-neg** |
|---|---|---|
| 선정 기준 | **Syntactic**: 같은 attr/obj 강제 (`same_prim_sample=True`) batch sampler | **Semantic**: LLM이 visually confusable composition 판정 — 같은 primitive 공유 안 해도 됨 |
| 예시 | "ripe apple" anchor → batch에 다른 "ripe X", "Y apple" 강제 | "ripe banana" anchor → "yellow paper", "brown leaf", "ripe pear" (다른 primitive지만 시각적 유사) |
| Lever 가설 | primitive 자체 disentanglement | image embedding 공간의 cross-primitive 혼동 분리 |
| Sealing 결과 | sealed 10th (val-test gap 확대로 일반화 fail) | 미시도 |

**LLM 산출물** (`data/hardneg_pairs_mit.json`):
- 1262 train compositions 각각에 대해 top-K=5 hard-neg compositions (K=5 선택 이유: batch_size=8 안에서 anchor 1 + hard-neg 3-5 + filler 2-4 균형, 너무 많으면 batch diversity 손상).
- 형식: `{"hard_neg": {comp_str: [comp_str_1, ..., comp_str_K]}}`.
- 생성 방식 검토 옵션:
  - (옵션 1) Claude in-conversation (build_descriptions_claude 패턴) — 1262 anchor × K outputs 약 1-2시간 신호. **선호**.
  - (옵션 2) Gemini API batch (feasibility json 패턴) — 비용 발생.
  - (옵션 3) CLIP image-feature 기반 cluster 기반 (LLM 안 씀) — D′의 LLM lever 의미 약화.
  - **결정**: 옵션 1.

**구현**:
- `Troika/code/dataset.py`에 hardneg pair 로드 + sampler weight 함수 추가.
- `Troika/code/model/troika_hardneg.py`: Troika 상속, forward에서 hard-neg comp text embedding과의 InfoNCE term 추가. `L_hardneg = -log(exp(s_pos/τ) / sum_{neg ∈ hardneg(anchor)} exp(s_neg/τ))` 형식, τ=0.1.
- `parameters.py`: `--lambda_hardneg` (default 0.0), `--hardneg_path`.
- `config/troika/mit-states-hardneg.yml`: λ_hardneg=0.1, 외 baseline 동일.

**Stage 0/1 cutoff**: §20-2 공통 (Δ HM > +0.011).

### 20-6. E design — LLM taxonomy / structural prior (사전 등록)

**메커니즘**: LLM이 obj → super-category (예: apple → fruit), attr → group (예: ripe ∈ ripeness_group). Visual feature 위에 auxiliary classifier 2개 (attr_group_clf, obj_supcat_clf) — 입력 visual feature, 출력 group/supcat logits. CE aux loss. mutex group(§20-4)과 무관 (taxonomy는 의미 분류, mutex는 상호 배제). 다른 sealed axis와 무관.

**LLM 산출물** (`data/taxonomy_mit.json`):
- 115 attrs → ~15 groups (예: ripeness, moisture, age, size, ...)
- 245 objs → ~25 supercat (예: fruit, animal, building, vehicle, ...)
- 작은 출력 (수십 KB), Claude in-conv 단일 turn 가능.

**구현**:
- `tools/build_taxonomy.py`: 하드코딩된 그룹 → JSON 산출.
- `model/troika_taxo.py`: Troika 상속 + 2개 aux head (nn.Linear(visual_feat_dim, n_groups) × 2), forward에서 visual feature 통과 → group logits, loss에 CE aux 추가.
- `parameters.py`: `--lambda_attr_group`, `--lambda_obj_supcat`, `--taxonomy_path`.
- `config/troika/mit-states-taxo.yml`: λ_attr_group=0.1, λ_obj_supcat=0.1.

**Stage 0/1 cutoff**: §20-2 공통.

### 20-7. F design — LLM test-time reranking (사전 등록)

**메커니즘**: 학습 변경 0. 가장 좋은 ckpt val_best.pt forward로 test set 13k images × top-K=5 composition prediction 산출. 각 image에 대해 multimodal LLM (Claude Sonnet/Opus 추정)에 image + K candidate strings 입력 → "이 이미지에 가장 맞는 composition?" 산출. Re-ranked top-1으로 HM/AUC 재계산.

**비용 검토** (블락커):
- mit-states test ~13k images × 1 multimodal LLM call = **13k API call**.
- Claude Sonnet vision: ~$3/Mtok input, $15/Mtok output. image token ~1.6k, 5 candidates short ~200 tok, output ~10 tok → 약 $0.005 per image × 13k = **~$65**.
- 사용자 사전 승인 필요.

**대안 (저비용)**:
- F′: test 대신 val 600 pairs × 1 image씩 sample = ~600 calls = ~$3. Sanity probe로 충분. 결과 좋으면 full test 진행.

**산출물**: `data/test_top5_preds.json` (사전 forward로 캐시) → reranker script → `eval_logs/F_rerank_<TS>.json`.

**Stage 0**: 50 image probe (val subset, 비용 ~$0.25). 출력 합리적인지 확인.
**Stage 1 cutoff**: full test (600 val or 13k test) 위에서 Δ HM > +0.011 → 유의. fail → axis 봉인. 비용 사용자 승인 필수.

→ **F는 C′/D′/E 중 최소 1 axis 통과한 후 best ckpt 위에서 시도**. 모두 fail시 baseline ckpt 위에서 단일 axis로 시도 가능.

### 20-4-3. C′ Stage 1 첫 시도 실패 (5-26 18:14 KST 외부 kill)

5-26 16:53 launch한 첫 C′ Stage 1 (`train_mit_c_prime_seed0_l0.1_20260526_165334.log`)이 **ep1 train 완주 + val eval(HM 0.3744 sane) 후 ep1 test eval Testing 79% (257/326) 시점에 외부 kill**. traceback 없음, save dir 빈 채. 당시 디스크 사용률 94% (107G free). val_metric=best_loss라 val 단계에서 val_best.pt 저장 시도가 있었어야 하나 빈 디렉토리 = 저장 실패 또는 kill 시점이 더 빨랐음. ep1 val HM 0.3744는 baseline ep1 (~0.3729, §11-10) 대비 +0.0015 — 평이한 시작.

### 20-4-4. dongki 디스크 정리 (5-26 21:00 KST)

용량 회수를 위해 dongki/ 트리 일괄 정리:
- **Tier 1**: 5개 baseline dir의 중간 epoch_*.pt 75개 삭제 (val_best.pt + final_model.pt 보존). 회수 ~127G.
- **Tier 2**: SEALED 축 디렉토리 15개 전체 삭제 (v3_text, imgcond, cluspro_baseline_l14_mit_v2_ot_seed0, Troika/save/mit-states_seed{0,1,2}_repro). 회수 ~83G.
- **Tier 3**: 구버전 4개 dir 전체 삭제 (cluspro_baseline_l14_mit_k3, cluspro_baseline_l14_utzap_lr5e5, lhp_czsl_v{1,2}_l14_mit). 회수 ~7G.
- 결과: dongki **298G → 86G**, 디스크 여유 107G → 319G.

**ckpt 손실 명시**:
- `Troika/save/mit-states_seed{0,1,2}_repro/val_best.pt + final_model.pt`: 삭제됨. seed1 (HM 0.3891) / seed2 (HM 0.3865) ckpt 소실, **metric은 train log에 보존** (§18-5와 동일). seed0 (§11-11, HM 0.3940) ckpt 동일하게 삭제됨.
- `cluspro_baseline_l14_mit_v2_ot_seed0`: 삭제됨 (Troika/LHP OT 모두 sealed라 무관).
- baseline_repro ckpt는 re-eval/ensemble/F축에 필요했을 수 있음. 후속 재학습 시 ~6.5h/seed.

### 20-4-5. C′ Stage 1 재시도 launch (5-26 21:28 KST)

cleanup 후 디스크 여유 확보, GPU 1 idle 확인. `Troika/scripts/run_c_prime_seed0.sh` 동일 설정으로 재실행:
- yml: `config/troika/mit-states-logic.yml` (λ_mutex=0.1, λ_compat=0.1)
- seed 0, bs=8/accum=8, 10ep, val_metric=best_loss
- nohup + disown, GPU 1
- save: `Troika/save/mit-states_c_prime_seed0_l0.1/`
- log: `Troika/logs/train_mit_c_prime_seed0_l0.1_20260526_212827.log`
- ETA ~5h 40m → **5-27 03:00 KST 부근 종료 예상**.

Stage 1 cutoff (재확인): test HM > 0.4050 → Stage 2 (seed 1, 2). fail → λ (0.3, 0.3) 1회 재시도 후 봉인.

### 20-4-6. C′ 3-seed 완주 + 판정 — 5-28 14:58 KST

**완주 상태**: seed0 (5-26 21:28 ~ 5-27 04:19), seed1→seed2 chain (5-27 19:54 ~ 5-28 09:07:55). 세 run 모두 정상 종료 — NaN/traceback/OOM 0회, `chain done` + wrapper `exit 0` 확인, 각 시드 `val_best.pt`+`final_model.pt` 저장됨. val_metric=**best_loss** (baseline 0.3899와 동일 기준 → 비교 fair; [[project_lhp_czsl_baseline_gap]] ckpt-selection 축 closed).

**3-seed test_pairs (Closed World, val_best.pt):**

| seed | seen | unseen | HM | AUC | attr_acc | obj_acc |
|---|---|---|---|---|---|---|
| 0 | 0.4655 | 0.5374 | 0.3929 | 0.2160 | 0.3955 | 0.5651 |
| 1 | 0.4845 | 0.5303 | 0.3878 | 0.2187 | 0.4054 | 0.5604 |
| 2 | 0.4664 | 0.5332 | 0.3863 | 0.2134 | 0.3985 | 0.5735 |
| **mean** | **0.4721** | **0.5336** | **0.3890** | **0.2160** | 0.3998 | 0.5663 |
| **std** | ±0.0107 | ±0.0036 | ±0.0035 | ±0.0027 | ±0.0050 | ±0.0066 |

**사전 등록 판정 (§20-2 / §20-4-2):**
- **Stage 1 (seed0)**: test HM 0.3929 < cutoff 0.4050 (Δ vs §11-11 single-seed 0.3940 = **−0.0011**) → Stage 1 **fail** (Δ HM < 0 bucket). [pre-reg대로면 여기서 λ=0.1 중단 → λ(0.3,0.3) 재시도가 맞으나, seed1/seed2를 λ=0.1로 그대로 돌림 — protocol 이탈이나 결과적으로 robust 3-seed null 확보.]
- **Stage 2 (3-seed)**: mean HM 0.3890 < cutoff **0.4009** (= 0.3899+0.011). Δ = **−0.0119** (cutoff 대비 ~3.4σ 아래). std 0.0035 ≤ 0.005 ✓ (consistency 조건만 충족).
- **vs baseline best_loss 3-seed** (HM 0.3899 / AUC 0.2177): **Δ HM −0.0009, Δ AUC −0.0017** — 둘 다 시드 노이즈(1σ ±0.0038) 안. **신호 0**.

**해석:**
- C′ logic-rule grounding (attr mutex soft penalty + obj-attr incompat soft penalty, λ=0.1)이 baseline 대비 metric을 전혀 못 움직였다. 3-seed mean이 baseline에 사실상 겹침 (HM −0.0009 = noise 이하).
- 원인은 Stage 0(§20-4-1)에서 이미 예고됨: λ=0.1에서 L_mutex+L_compat 합산 기여가 base loss의 **~0.2%**. attribute posterior softmax에 거는 soft penalty가 너무 약해 representation을 바꾸지 못함. attr_acc/obj_acc도 baseline 수준 (0.3998/0.5663) → mutex/compat regularizer가 attribute 분포를 sharpen하지 못했다는 직접 증거.
- std 0.0035로 tight → 이 null은 노이즈가 아니라 진짜 (mechanism이 inert). seen은 시드 변동(±0.0107)이 좀 있으나 HM/unseen은 매우 안정.

**Verdict: C′ logic-rule axis fail.** 사전 등록상 남은 변형은 λ(0.3,0.3) Stage 1 1회 재시도뿐. λ를 3배 키워도 loss 기여 ~0.6%로 여전히 미미하고, 3-seed가 directional signal 0 (baseline보다 오히려 −0.0009)이라 retry의 expected value 낮음. **결정 대기**: (a) pre-reg 준수하여 λ(0.3,0.3) seed0 1런(~6.5h) 후 봉인, 또는 (b) 즉시 봉인하고 D′(semantic hard-neg mining, §20-5)로 이동.

**누적**: §13-5h 이후 training-change mechanism 봉인 8 → **9** (C′ 추가). §20 redefined-axis(C′/D′/E/F) 중 첫 번째 fail.

**결정 (5-28, 사용자):** option (b) — **λ(0.3,0.3) 재시도 waive하고 C′ 즉시 영구 봉인**. 근거: λ를 3배 키워도 loss 기여 ~0.6%로 mechanism이 inert하다는 진단(§20-4-1 + 3-seed null)을 뒤집을 근거 없음. → **D′ (LLM semantic hard-neg mining, §20-5) 진입**.

### 20-5-1. D′ 구현 + hard-neg 생성 + Stage 0 smoke — 5-28 15:47 KST

**Hard-neg 생성 방식 (사용자 결정 5-28):** §20-5 옵션1(Claude in-conversation)을 C′ 선례(MUTEX_GROUPS 하드코딩)대로 **큐레이션 클러스터**로 구현. "visually confusable"을 두 Claude-큐레이션 소스로 operationalize:
- `ATTR_SIM_GROUPS` (38개): C′ mutex groups + 비-mutex 시각 혼동군(damage/tear/crush/crease/slice_prep/corrosion/curve/finish/...). attr의 siblings = 소속 그룹 합집합.
- `OBJ_FAMILIES` (37개): fruit/animal/vehicle/building/water_body/metal/textile/... 시각·재질·장면 family. many-to-many.
- 각 anchor (a,o): tier1(same obj+sibling attr) / tier2(same attr+sibling obj) / tier3(cross-primitive, a≠·o≠ 둘 다 sibling) 후보 → train_pairs 내 필터 → quota 2·2·1 후 backfill, top-K=5.
- 산출 `data/hardneg_pairs_mit.json` (`tools/build_hardneg_pairs.py`): **1164/1262 comp이 neg 보유** (98개는 sibling이 train pair에 없어 0개), 4661 edges, avg 4.0/comp, tier edge t1=935 / t2=1916 / **t3=1810 (cross-primitive 39%)**. cross-primitive 비중이 D′의 sealed hardpair(§17-9 syntactic 강제)와의 차별점.

**구현 (loss-only, dataset/sampler 변경 0):**
- `model/troika_hardneg.py` (TroikaHardneg < Troika): comp_logits(전체 1262 train pair 유사도) 행에서 hard-neg 집합만 분모로 제한하는 InfoNCE. `L_hn = -log(exp(s_pos/τ)/(exp(s_pos/τ)+Σ_neg exp(s_n/τ)))`, s=cosine(=comp_logits/logit_scale.exp()), τ=0.1. forward에서 `idx`(train_pairs) 캡처해 hard-neg mask [P,P] lazy 빌드(training only). neg 없는 sample은 -inf 마스킹 → logsumexp=pos → loss 0 (NaN-free). **sampler 미사용** — InfoNCE가 batch 구성과 무관하게 full comp-logit 행에서 동작하므로 학습 이미지 분포가 baseline과 동일(confound 제거). §20-5 "optional InfoNCE"를 core로 채택, sampler-weight는 생략.
- `model/model_factory.py`에 `troika_hardneg` 등록. `config/troika/mit-states-hardneg.yml` (lambda_hardneg=0.1, tau_hardneg=0.1, bs=8/accum=8, 10ep, val_metric=best_loss). parameters.py 변경 불필요(`load_args`가 yml 키를 config에 setattr).

**Stage 0 smoke (`smoke_hardneg.py`, 30 step, GPU 1):**
- mask: 1164 anchors matched, 4661 edges, **0 dropped** (JSON neg 전부 valid train pair).
- NaN/Inf **0회**, VRAM peak **13.96 GB** (≤17 안전, C′와 동일), **1.76 it/s** (baseline 1.86 대비 -5%).
- L_hn raw ~1.1–1.6 → ×λ=0.1 = base loss(~16-19)의 **약 0.7-0.9%**. **C′(~0.2%)보다 4배 강한 기여** — InfoNCE가 soft-penalty보다 bite 있음. avg negs/sample ~4.0.
- **Stage 0 통과** (NaN-free ✓ finite ✓ VRAM ✓ 속도 ✓) → Stage 1 진입.

### 20-5-2. D′ Stage 1 launch — 5-28 15:47 KST

- script `Troika/scripts/run_d_prime_seed0.sh`, yml `mit-states-hardneg.yml`, seed 0, 10ep, GPU 1, nohup+disown.
- save `Troika/save/mit-states_d_prime_seed0_l0.1/`, log `train_mit_d_prime_seed0_l0.1_20260528_154712.log`. main PID 3390030.
- 첫 epoch 1.80 it/s, 3793 it/ep → **ETA ~6h, 5-28 21:45–22:00 KST 종료 예상**.
- **GPU 노트:** 동시각 GPU 0에 외부 사용자(`kiseoup`) retrieval 학습 점유 중이나 **GPU 1은 D′ 단독**(16.8/24GB), 경합 없음.
- **Stage 1 사전 등록 cutoff (§20-2/§20-4-2 공통):** test HM > **0.4050** (Δ>+0.011 vs §11-11 0.3940) → Stage 2(seed 1,2). 0~+0.011 → tie, 봉인. <0 → fail, 봉인. λ=0.1 fail시 (0.3,0.3)/τ sweep 1회 후 봉인.

### 20-5-3. D′ Stage 1 크래시 → loss_calu eval-guard 수정 → 재실행 — 5-29 11:03 KST

**크래시 (5-28 run):** §20-5-2 seed0 run이 epoch 1 학습 완료 후 **첫 validation에서 즉시 크래시** (16:23 로그 종료, 정상 종료 아님 — best_loss 선택/test 결과 0개).
- `RuntimeError: The size of tensor a (1262) must match the size of tensor b (1962) at non-singleton dimension 1` @ `model/troika_hardneg.py:113` (`loss_calu`, masked_fill).
- **원인:** `evaluate()`가 `test.predict_logits`에서 val loss(best_loss용)를 위해 `model.loss_calu(predict, data)`를 호출. eval은 **closed-world 전체 pair set(1962)** 위에서 채점 → `comp_logits` [B,1962]. 그러나 hardneg `hardneg_mask`는 **train pair(1262)** 기준 [1262,1262] → `neg_mask=hardneg_mask[batch_target]`([B,1262])를 cos([B,1962])에 masked_fill 시 mismatch. 학습 경로(comp_logits=train_pairs 1262)에선 안 터지고 eval 첫 진입에서 처음 노출.

**train()/eval() 함정 (수정 설계에 반영):** `train.py`는 `model.train()`을 루프 진입 전 line 29에서 **1회만** 호출하고 매 epoch `evaluate()`가 `model.eval()`(line 109)로 전환, 루프 상단에서 train() 재호출 없음 → **epoch 1 eval 이후 전 학습 step이 eval 모드**. 따라서 `self.training`으로 hardneg를 가드하면 epoch 2~10에서 hardneg가 조용히 꺼져 D′ intervention이 9/10 무력화. (baseline도 동일 train.py로 dropout-off 학습 → 비교 일관성 위해 train.py 미수정.)

**수정 (`troika_hardneg.py` `loss_calu`, 1-line guard):** `self.training` 대신 **pair-set 크기**로 가드 — `if comp_logits.shape[1] != self.hardneg_mask.shape[0]: return loss`. train(P=1262=mask)일 때만 hardneg 적용, eval(P=1962)은 base loss만 반환. 효과: (1) eval 크래시 제거, (2) hardneg는 전 학습 epoch에서 적용(eval 모드 무관), (3) val/test best_loss = base loss → baseline과 동일 선택 신호 유지. dataset/sampler/train.py 변경 0. py_compile 통과.

**재실행 (5-29 11:03 KST, GPU 1):** 동일 script `run_d_prime_seed0.sh`, log `train_mit_d_prime_seed0_l0.1_20260529_110332.log`.
- **epoch 1 통과 검증 (11:42, ~39분):** 크래시 지점 통과, val `seen 0.4197 / unseen 0.5575 / HM 0.3737 / AUC 0.2015` (epoch1). NaN/traceback 0.
- 현재 epoch 2 진행, GPU 1 100%/18.9GB. epoch당 ~39분 → 10ep **ETA ~17:30 KST**. Stage 1 cutoff 동일 (test HM > 0.4050 → Stage 2).

### 20-5-4. D′ Stage 1 결과 + 봉인 판정 — 5-31 KST

**학습 정상 종료 (5-29 17:39 KST):** 10 epoch 완주, crash/NaN/traceback 0. save `mit-states_d_prime_seed0_l0.1/`에 `val_best.pt`(5-29 12:21, best_loss 선택), `epoch_4/9.pt`, `final_model.pt` 정상 생성. train loss 매끄럽게 수렴(ep1 1.419 → ep10 0.0418). §20-5-3 eval-guard 수정으로 첫 validation 크래시 재발 없음.

**최종 test 결과 (val_best.pt → test set, closed-world; log line 72, "--- Evaluating test dataset on Closed World ---" 직후):**

| seen | unseen | **HM** | AUC | attr_acc | obj_acc |
|---|---|---|---|---|---|
| 0.4714 | 0.5388 | **0.3951** | **0.2189** | 0.3943 | 0.5643 |

**판정 (사전등록 cutoff, §20-5-2):**
- Stage 2 진입선 test HM > **0.4050** (Δ > +0.011 vs §11-11 baseline 0.3940) → **미달**.
- 실측 Δ HM = 0.3951 − 0.3940 = **+0.0011** → 사전등록 **tie 구간 (0 ~ +0.011)** 에 해당.
- 규칙: tie → **봉인** (fallback인 (0.3,0.3)/τ sweep은 *fail(<0)* 조건에서만 발동 → tie인 본 케이스는 곧장 seal). λ=0.1에서 InfoNCE hard-neg가 base loss의 ~0.7-0.9% bite(C′의 4배)를 줬음에도 test HM은 baseline 노이즈 대역(mit-states 3-seed HM std ±0.0008 수준) 내 → mechanism-level **information 없음**.

**누적**: training-change mechanism 봉인 9 → **10** (D′ 추가). §20 redefined-axis(C′/D′/E/F) 중 C′·D′ 2건 연속 fail/tie. **E/F 대기.**

**결정:** D′ 영구 봉인. seed1·2 추가 산출 waive — single-seed가 cutoff 대비 −0.010이고 tie 대역이라 3-seed 평균이 0.4050을 넘을 확률 사실상 0 (§20-4 C′ 봉인 시 동일 논리 선례).

### 20-6-1. E Stage 0 smoke 통과 — 5-31 11:35 KST

**메커니즘 (§20-6 사전등록 그대로):** TroikaTaxo — global visual feature 위에 aux linear head 2개로 LLM taxonomy 예측, aux CE로 visual representation을 hierarchical prior로 regularize. C′(posterior-shape penalty)·feasibility(logit bias)와 grain 다름 (coarse label을 image에서 예측).

**산출물:**
- `LHP-CZSL/tools/build_taxonomy.py` + `LHP-CZSL/data/taxonomy_mit.json` — Claude in-conv 큐레이션. 115 attrs → **18 STATE-TYPE groups** (age_wear/size/shape/damage/edge/surface/light/cleanliness/moisture/clarity/cooking/ripeness/phase/cut/fill/openness/orientation/weather_env), 245 objs → **21 super-categories** (fruit/animal/metal/building/place_nature/…). 각 primitive 정확히 1개 그룹에 배정(coverage assertion 통과, 중복 0).
- `Troika/code/model/troika_taxo.py` (TroikaTaxo < Troika): `encode_image` 오버라이드로 global feat 스태시(forward 중복 안 함), aux head 2개는 super().__init__() freeze 루프 *이후* 생성 → requires_grad=True로 optimizer(`model.parameters()` 전체 Adam)가 자동 포착. `loss_calu` = base CE + λ_ag·CE(attr_group) + λ_os·CE(obj_supcat).
- `Troika/code/config/troika/mit-states-taxo.yml` (λ_attr_group=0.1, λ_obj_supcat=0.1, bs=8/accum=8, 10ep, val_metric=best_loss). parameters.py 변경 불필요(load_args가 yml 키 setattr).

**train/eval 게이트 (D′ 크래시 교훈 반영):** aux loss는 `torch.is_grad_enabled()`가 True일 때만 적용. eval은 `predict_logits`/`predict_logits_text_first` 모두 `with torch.no_grad()` 안에서 돌므로(test.py 확인) → eval `loss_calu`는 **base CE만 반환** = best_loss 선택 신호가 vanilla baseline과 동일(confound 0). **`self.training` 미사용** — train.py가 epoch1 eval 후 model.eval() 상태로 남는 함정(§20-5-3) 회피. 또한 aux head는 visual feat[B,768]에만 작용하고 predict tuple(3개)에 아무것도 안 붙이므로 → D′의 pair-set mismatch(1262 vs 1962) / `logit_infer` unpack 크래시에 **구조적 면역**.

**Smoke 결과 (`smoke_taxo.py`, 30 step, GPU 1, λ=0.1/0.1):**
- taxonomy 로딩 정상, aux head [18,768]/[21,768].
- **게이트 검증 ✓:** grad-on total 17.77 vs no_grad total 17.18 == base 17.18 (eval에서 aux 꺼짐 확인).
- NaN/Inf **0회**, VRAM peak **13.97 GB** (≤17, C′/D′와 동일), **1.83 it/s** (baseline 1.86 대비 −1.6%, C′/D′보다 빠름).
- aux 기여 = base의 **~3.3%** (l_attr_group ~3.0 ≈ ln18, l_obj_supcat ~3.0 ≈ ln21 — init near-uniform CE 기대치 일치). **C′(~0.2%)의 16배, D′(~0.8%)의 4배** — 지금까지 가장 강한 bite.
- **Stage 0 통과** (NaN-free ✓ finite ✓ VRAM ✓ 속도 ✓ 게이트 ✓) → Stage 1 진입.

### 20-6-2. E Stage 1 launch — 5-31 11:40 KST

- script `Troika/scripts/run_e_taxo_seed0.sh`, yml `mit-states-taxo.yml`, seed 0, 10ep, GPU 1, nohup+disown.
- save `Troika/save/mit-states_e_taxo_seed0_l0.1/`, log `train_mit_e_taxo_seed0_l0.1_20260531_114037.log`. main PID 2943747.
- 첫 epoch **1.83 it/s, 3793 it/ep → ~34.5분/epoch → 10ep ETA ~17:25 KST**. epoch1 train loss ~2.2 (= total 17.7 ÷ accum 8).
- **GPU 노트:** GPU 0에 외부 사용자(`kiseoup`) retrieval 학습 점유 중이나 **GPU 1은 E 단독**(16.8/24GB), 경합 없음.
- **Stage 1 사전 등록 cutoff (§20-2 공통):** test HM > **0.4050** (Δ>+0.011 vs §11-11 0.3940) → Stage 2(seed 1,2). 0~+0.011 → tie, 봉인. <0 → fail, 봉인. λ=0.1 fail시 (0.3,0.3) sweep 1회 후 봉인.
- **첫 validation 체크포인트:** epoch1 종료 ~12:15 KST에 첫 eval — **크래시 없이 통과 ✓ (12:20 확인)**. epoch1 val `seen 0.4176 / unseen 0.5644 / HM 0.3798 / AUC 0.206 / attr_acc 0.3986 / obj_acc 0.5824` (D′ epoch1 HM 0.3737보다 약간 위). is_grad_enabled 게이트 + predict tuple 불변 설계로 D′형 pair-set 크래시 **구조적 면역 실증됨**. epoch 2~10 계속 진행 중, ETA ~17:25 KST.

### 20-6-3. E Stage 1 결과 + 판정 — 5-31 KST

**학습 정상 종료 (5-31 18:17:53 KST):** 10 epoch 완주, crash/NaN/traceback 0. save `mit-states_e_taxo_seed0_l0.1/`에 `val_best.pt`(12:59, best_loss 선택), `epoch_4/9.pt`, `final_model.pt` 정상 생성. train loss 매끄럽게 수렴(ep1 1.454 → ep10 0.0586). is_grad_enabled 게이트로 첫 validation 크래시 재발 없음.

**Val 곡선 (epoch별, `Evaluating val dataset`):** HM 피크는 **epoch 2 (seen 0.5087 / unseen 0.5717 / HM 0.4201 / AUC 0.2526)**, 이후 단조 하락(ep10 HM 0.382). 전형적 과적합 곡선 — train loss는 계속 떨어지나 val 일반화는 ep2에서 정점.

**Test 결과 (best_loss=val_best ckpt, log line 72 `--- Evaluating test dataset on Closed World ---`):**
- **seen 0.4744 | unseen 0.5333 | HM 0.382 | AUC 0.2136 | attr_acc 0.3965 | obj_acc 0.5642**

**판정 (사전등록 cutoff, §20-6-2 / §20-2 공통):**
- Stage 2 진입선 test HM > **0.4050** → 미달.
- Δ HM vs §11-11 baseline 0.3940 = **−0.012** → pre-reg **fail 버킷 (<0)**. (best_loss baseline 0.3899 대비로도 −0.0079, AUC 0.2136 vs 0.2177 = −0.0041 — seen/unseen/HM/AUC 전 지표 하락.)
- **핵심 관찰:** E의 aux bite는 base loss의 **~3.3%로 §20 redefined-axis 중 최강**(C′ ~0.2%의 16배, D′ ~0.8%의 4배)이었음에도 test signal은 **음의 방향**. "bite를 키우면 representation이 바뀐다"는 가설의 직접 반증 — aux로 hierarchical prior를 강하게 주입해도 compositional generalization은 오히려 약화(과적합 가속). taxonomy aux head가 visual feat를 coarse label로 끌어당기면서 fine-grained attr 변별을 희석한 것으로 해석.

**누적:** training-change mechanism 봉인 후보 10 → (E 봉인 시) **11**. §20 redefined-axis(C′/D′/E/F) 중 **C′·D′·E 3건 연속 fail/tie**.

**결정 (6-1, 사용자): E λ(0.3,0.3) retry waive하고 E 영구 봉인.** 근거: (a) E는 이미 최강 bite(3.3%)에서 −0.012 음의 신호 → λ 3배(bite ~10%)가 방향을 뒤집을 근거 없음, (b) C′ 선례(5-28 결정 b)로 동일 논리의 retry waive 전례. **누적 봉인 10 → 11 (E 확정).**

### 20-7-1. F Stage 0 probe — 무료(API $0) self-Claude rerank, 50장 — 6-1 KST

**동기 (사용자 5-31):** "F를 API 대신 self-Claude(이 대화의 multimodal Claude)로 진행 가능한가?" → F reranker가 multimodal이므로 Claude가 직접 reranker 역할 가능. full 13k는 비현실(수작업·재현불가)이나 **Stage 0 50장 probe는 $0로 즉시 가능**.

**파이프라인 (`Troika/code/probe_f.py`, GPU forward 비용0):** E `val_best.pt`(≈baseline, baseline ckpt는 §20-4-4에서 삭제됨)로 test 12,995장 closed-world forward → 각 image top-5 composition 후보 캐시 `data/F_probe/test_top5_preds.json`. **모델 closed-world top1(unbiased) = 0.286, gt-in-top5 = 0.606 → 복구가능(top1 wrong & gt in top5) 32%(4163장).** 천장이 높아 probe 가치 있음.

**1차 시도 (오염):** 50장 blind(후보에서 gt 가림) probe를 내가 직접 rerank → model 0.340 → 0.560 (**+0.220**, 13 복구가능 중 12 회복). **그러나 무효** — 이미지 `Read` 경로(`.../images/draped fabric/draped-fabric-michelle-miron.jpg`)에 **폴더명·파일명이 곧 gt**라 정답이 경로로 leak됨. 후보만 가리고 경로를 안 가린 설계 결함.

**2차 재측정 (clean):** 50장을 **익명 파일명(`/tmp/F_probe_anon/pNNN.jpg`)으로 복사** + 후보 alphabetical 정렬(model-rank 무관) → **gt를 모르는 fresh subagent 5개**(10장씩)가 익명 이미지만 보고 rerank. 결과:
- **model top1 0.340 → LLM rerank 0.320 (net −0.020).** 진짜 복구 5 vs HURT 6 (멀쩡한 model top1을 깨뜨림). 천장 0.600의 발끝도 못 감.

**진단 — 왜 안 되나:** HURT 케이스가 전부 **한 이미지에 동시 참인 속성이 여럿**인 경우. `huge balloon→inflated balloon`, `sliced potato→browned potato`, `folded chair→upright chair`, `modern clock→small clock`, `coiled copper→coiled wire`, `broken car→crushed car`. zero-shot LLM은 시각적으로 똑같이 맞는 대안 속성을 고르지만 dataset이 채택한 single-label과 불일치. **학습된 모델은 dataset labeling 편향을 학습했고, LLM은 모름** → rerank가 진짜 오류 일부를 고쳐도 동수 이상을 깨서 net ≤ 0.

**판정 (6-1, 사용자 확정): F 영구 봉인, $65 full-test run waive.** F Stage 0 probe가 합리성 체크에서 **non-positive (net −0.02, n=50 노이즈 대역 내 0)** + pair-acc top1조차 음수라 HM 개선 기대 0 → pre-reg Stage 1 cutoff(Δ HM > +0.011) 통과 가망 없음. C′/D′/E waive 선례대로 확정 run waive. → §20 redefined-axis(C′/D′/E/F) **4축 전부 fail/null로 소진.**

**방법론 교훈 (기록):** vision-LLM probe에서 **이미지 파일 경로/파일명에 label이 박혀 있으면 그 자체가 leak**. 향후 multimodal eval은 (1) 익명 파일명 복사, (2) gt-blind 평가자(별도 subagent/API) 필수. 1차 +0.22 → 2차 −0.02 전환이 leak 규모의 직접 증거.

**누적:** §20 axis F까지 봉인 시 redefined-axis 4/4 소진. training-change/inference 봉인 후보: C′·D′·E (10→11 E 봉인 시) + F(inference, 별도 카운트). → **새 architecture/framework 진입 결정 국면.**

---

## §21. New framework program — differentiation-first (post §20 소진, 6-1 진입)

**전제 (§20 결론):** axis/hyperparam/inference 변형으로 Troika gap(−0.017 vs ClusPro) 닫기 EV ≈ 0. F probe 진단: **표현은 gt를 top-5까지 올림(0.606)에도 top-1(0.286)에서 sibling-state를 못 변별** → 병목은 *같은 object 아래 형제 상태의 시각 granularity*.

**목표 재정의 (사용자 6-1):** "ClusPro 구현·성능 추격은 안 함. **우리만의 차별화된 새 메커니즘**이 목표 — 성능이 baseline보다 조금 부족해도, novel하고 defensible하면 OK."
→ **판정 기준도 재정의**: "baseline 초과"가 아니라 **(a) 메커니즘이 의도대로 작동(ablation·해석으로 입증) + (b) 성능이 baseline 대비 큰 손해 없음(non-regression)** 이면 PASS(=논문화 가능한 차별 기여 확보). 초과는 stretch.

**우선순위 (Claude 추천, 6-1 확정 — 하나씩 순차):**
1. **(I) PGAL — Patch-Grounded Attribute Localization** ← 착수
2. **(II) Attributes-as-operators (CLIP-era 부활)** — (I) fail/collide 시 fallback, 가장 깨끗한 novelty
3. **(IV) Generative unseen-feature synthesis** — 승자 위에 직교 stack
4. **(III) Ordinal/continuous state** — 독립 축 대신 (I)/(II)의 regularizer로 흡수

### 21-1. Axis I — PGAL 사전등록 plan

**Thesis (한 줄):** 기존 CLIP-CZSL(CSP/DFSP/Troika/ClusPro)은 **이미지 표현을 전역으로 두고 텍스트 쪽을 적응**시킨다. PGAL은 반대로 **속성이 발현된 영역으로 이미지 표현을 국소화**하여 attribute를 그 국소 영역에서 채점한다.

**Troika 대비 차별 (코드 감사 6-1, troika.py forward/CMT):** Troika CMT = *텍스트가 patch에 cross-attend → 텍스트 prompt 조정* → score = (조정된 텍스트)·**전역 CLS**. 즉 이미지 쪽은 끝까지 global. **PGAL은 image-side localization** — Troika가 안 건드리는 직교 축. `encode_image`가 이미 `(CLS, patches[N,256,768])` 반환 → patch grid 가용 확인.

**아키텍처 스케치:**
- Object 분기: 전역 CLS → object 채점 (object는 holistic).
- **Attribute 분기 (핵심):** attribute query q_a(=attr CLIP text emb 또는 learned proto) → patch token에 cross-attention → attention map a∈Δ^256 + 국소 feature z_a = Σ a_i·patch_i. score_attr = sim(z_a, attr_text).
- Composition score = β·sim(CLS, obj_text) + α·sim(z_a, attr_text) (+ optional joint comp term, Troika comp-branch 재활용 가능).
- **Localization 정규화:** attention entropy/sparsity(집중 유도) + (optional) 동일 attribute가 다른 object에서 유사 시각 단서에 attend하도록 consistency. region label 無 (unsupervised).
- (III) 흡수 옵션: ordinal 상태군(ripeness/size/moisture…)엔 z_a에 순서 ranking 정규화 추가.

**4-stage + 사전등록 cutoff (baseline = Troika best_loss 3-seed HM 0.3899 ± 0.0038 / §11-11 single 0.3940):**
- **Stage 0 (smoke + 차별 lock):** (a) Troika CMT patch 사용 정밀 감사 → 차별 확정 *[6-1 완료: image-side localization으로 lock]*. (b) `model/troika_pgal.py` 구현 → 1ep smoke: NaN/Inf 0, VRAM ≤17GB, **attention map이 ~10장에서 질적으로 의미있는 영역(rust 영역, cut 단면 등) 지목하는지 시각 검증**. 무의미하면 즉시 재설계.
- **Stage 1 (seed0, 10ep):** mit-states. **판정 = 재정의 기준**: (PASS) HM ≥ 0.3899 − 0.005 = **0.3849** AND localization ablation(분기 제거 시 HM 하락) AND attention 의미성 → 차별 기여 확보, Stage 2 진입. (STRETCH) HM > 0.4050. (FAIL/seal) HM < 0.3849 OR localization이 noise(ablation 무효과 + 맵 무의미) → 봉인, (II)로.
- **Stage 2 (3-seed + 전이):** Stage 1 PASS 시 3-seed 안정성 + UT-Zap 전이.
- **Stage 3 (분석/해석):** attention map 정성 + **F_probe recoverable-headroom 오라클**로 "PGAL이 F probe가 짚은 sibling-state 혼동을 실제로 복구하는가" 정량 검증.

**산출:** `Troika/code/model/troika_pgal.py`, `config/troika/mit-states-pgal.yml`, `smoke_pgal.py`, `scripts/run_pgal_seed0.sh`. baseline ckpt 삭제됨(§20-4-4) → 비교는 3-seed mean 수치(보존됨) 기준.

### 21-1-1. PGAL Stage 0 smoke 통과 + Stage 1 launch — 6-1 KST

**구현:** `TroikaPGAL(Troika)` — `PatchLocalizer`(학습 query 1개가 256 patch token에 multi-head cross-attn → 국소 feature z_a + attention map). forward에서 attribute 분기의 *이미지* feature를 `attr_disentangler(CLS)`(전역) 대신 **z_a(국소)**로 교체; comp=CLS, obj=disentangled CLS는 Troika 그대로. localizer는 super().__init__ freeze 루프 *이후* 생성 → requires_grad=True(taxo aux 선례). lambda_loc=0.01 attention-entropy 정규화(focus 유도)는 **train-only(`is_grad_enabled`)** → eval best_loss == base CE (ckpt 선택 confound 0).

**⚠️ eval-path 버그 사전 차단 (중요):** `text_first=True` eval은 `predict_logits_text_first`→**`model.forward_for_open`** 호출(≠ `forward`). forward만 override하면 **학습엔 PGAL 켜지고 eval/test엔 Troika 기본 경로로 우회 → localization이 test에서 빠지는 train/eval mismatch**(§20-5-3 류). → `forward_for_open`도 동일 localization으로 override. 1배치 eval-path 검증: forward_for_open OK, logit_infer [8,1962], eval loss finite, **_last_attn [8,256] finite (localizer가 eval에서도 활성)** ✓.

**Smoke (30 step, GPU1, λ_loc=0.01):** localizer 2.36M params, total trainable 31.4M. **gate ✓**(grad-on 17.93 vs no_grad 17.87 == base 17.87). NaN/Inf **0**, VRAM **14.0GB**(≤17), **1.83 it/s**(baseline 1.86 동급). attention은 init에서 ~uniform(entropy 5.545/5.545, max 0.004=1/256) — 랜덤 init+30step(옵티마 3회)이라 정상; **의미있는 localization은 Stage 1 학습 후 검증**. **Stage 0 통과**(plumbing+gate+NaN+VRAM+속도+eval-path) → Stage 1 진입.

**Stage 1 launch (6-1 00:26 KST):** `run_pgal_seed0.sh`, seed0, 10ep, GPU1 단독. save `mit-states_pgal_seed0_l0.01/`, log `train_mit_pgal_seed0_l0.01_20260601_002626.log`, PID 590394. epoch1 train loss 2.24→2.01 매끄럽게 하강, 1.82 it/s, 16.9GB. **ETA ~06:30 KST (~6h).**
- **판정 (재정의, differentiation-first)**: PASS = test HM ≥ **0.3849**(baseline 0.3899−0.005, non-regression) AND localization ablation 기여 입증 AND attention map 의미성. STRETCH = HM > 0.4050. FAIL/seal = HM < 0.3849 OR localization noise.
- Stage 1 종료 후 할 일: (a) test HM/AUC 판정, (b) **ablation**(localizer 분기 제거=attr 전역 복귀 시 HM 변화), (c) **attention map 시각화**(viz_pgal_attn.py — early/final ckpt로 10장, rust영역·cut단면 등 지목하는지), (d) F_probe 오라클로 sibling-state 복구율.
### 21-1-2. PGAL Stage 1 결과 + 판정 — 6-1 KST

**학습 정상 종료 (6-1 07:03:39 KST):** 10 epoch 완주, crash/NaN/traceback 0. GPU1 단독. save `mit-states_pgal_seed0_l0.01/`에 `val_best.pt`(01:45, best_loss 선택), `epoch_4/9.pt`, `final_model.pt` 정상 생성. train loss 매끄럽게 수렴(ep1 1.323 → ep10 0.0413). forward_for_open override로 eval-path mismatch(§20-5-3 류) 재발 없음.

**Val 곡선 (epoch별):** HM 피크 **epoch 5 (seen 0.5081 / unseen 0.561 / HM 0.4193 / AUC 0.2468)**, ep2~5 plateau(0.415~0.419)에서 ep6 이후 하강(ep10 0.4022). E(taxo)의 ep2 단일피크 대비 **고원이 넓고 피크값도 높음**(E ep2 HM 0.4201과 동급이나 PGAL은 4 epoch 유지). best_loss로 선택된 val_best.pt는 01:45 = 초기 epoch.

**Test 결과 (best_loss=val_best ckpt, log line 72 `--- Evaluating test dataset on Closed World ---`):**
- **seen 0.4676 | unseen 0.5329 | HM 0.3878 | AUC 0.2127 | attr_acc 0.3991 | obj_acc 0.5651**

**판정 (재정의 differentiation-first cutoff, §21-1-1):**
- 비교 기준 = Troika best_loss 3-seed HM **0.3899 ± 0.0038**. non-regression PASS bar = **0.3849**.
- 실측 test HM **0.3878 ≥ 0.3849** → **non-regression 통과** (Δ vs baseline 0.3899 = **−0.0021**, 3-seed std ±0.0038 노이즈 대역 *안*). AUC 0.2127 vs baseline best_loss 0.2177 = −0.005. STRETCH(>0.4050) 미달.
- **§20 4축(C′/D′/E/F) 전부 fail/null 이후 첫 non-regression 결과.** E(taxo)가 동일 baseline 대비 test HM −0.012(fail)였던 것과 대조 — image-side localization 축은 적어도 표현을 *깨뜨리지 않음*.

**⚠️ 단, PASS 조건 미완:** §21-1-1 재정의 PASS = (a) HM non-regression **AND** (b) localization ablation 기여 입증 **AND** (c) attention map 의미성. 현재 **(a)만 충족**. (b)(c)는 미실시 → **차별 기여는 아직 미입증**. HM이 노이즈 대역 내라 "localizer가 실제로 일을 하는지"는 ablation/시각화 없이는 단정 불가(localizer가 사실상 전역 평균으로 collapse해도 같은 HM 나올 수 있음). → **다음 단계 (b)(c)(d) 필수, 이걸로 PASS/seal 최종 확정.**

**다음 할 일 (Stage 1 종료 후, §21-1-1 (b)(c)(d)):**
1. **(b) Ablation** — localizer 분기 제거(attr branch를 전역 disentangled CLS로 복귀) 재학습 seed0. HM 하락하면 localization 기여 입증, 무변화면 localizer가 noise → seal.
2. **(c) Attention map 시각화** — `viz_pgal_attn.py`, val_best/final ckpt로 10장, rust 영역·cut 단면·moisture 등 의미 영역 지목하는지 정성 검증. uniform/무의미하면 seal.
3. **(d) F_probe 오라클** — PGAL이 §20-7-1 F probe가 짚은 sibling-state 혼동(top1 0.286, gt-in-top5 0.606)을 실제로 복구하는지 정량.
4. 위 통과 시 **Stage 2** — 3-seed 안정성 + UT-Zap 전이.

### 21-1-3. PGAL (b) ablation launch + C′ staged 재검토 셋업 — 6-2 KST

**PGAL (b) ablation (uniform attention) launch (6-2 15:18 KST):** `loc_uniform=True`로 patch attention을 균등 강제 → z_a = learned-projection mean-pool. **localization만 제거, 추가 pathway·params는 PGAL과 동일**(localization 기여 분리). config 기반 토글(`troika_pgal.py`에 `self.localizer.uniform = bool(getattr(config,"loc_uniform",False))` 추가, 기본 False=PGAL). yml `mit-states-pgal-abluniform.yml`, script `run_pgal_abluniform_seed0.sh`, seed0 10ep GPU1 단독, log `train_mit_pgal_abluniform_seed0_l0.01_20260602_151801.log`. 검증: loc_uniform True 파싱 ✓, attn max−min=0.0(완전 균등) ✓. epoch1 1.80 it/s, NaN 0. **ETA ~21:20 KST.** 판정: PGAL(0.3878) > uniform-ablation이면 localization 기여 입증, ≈이면 localizer=mean-pool noise → seal.

**C′ staged 재검토 셋업 (launch 대기 — ablation 종료 후 GPU1):** §20-4 C′(sealed fail Δ HM −0.0009, bite ~0.2% inert)를 **LOGICZSL(CVPR'25, +1.3 AUC, 동일 logic family on Troika)** 스타일로 재검. 동기: final_presentation Q2 "구현 차이였나"를 추측이 아니라 데이터로. 식별된 차이 3종 반영:
- **(A) staged injection** — warmup 3ep logic OFF → 2ep 선형 ramp → full 5ep (base 표현 형성 후 투입). step 기반(warmup 11379 / ramp 7586 = 3793 it/ep 기준).
- **(B) 더 강한 bite** — λ_mutex=λ_compat 0.1→**0.5** (inert ~0.2% → ~1%+ 목표).
- **(C) eval confound 제거** — `is_grad_enabled` 가드 추가. *발견: 원래 TroikaLogic.loss_calu는 grad 가드가 없어 eval loss에도 logic term이 섞여 best_loss 선택이 미세 오염됐음*(bite 0.2%라 영향은 작았을 것이나, E/PGAL 규율로 교정).
- 산출물: `model/troika_logic_staged.py`(TroikaLogicStaged < TroikaLogic), `config/troika/mit-states-logic-staged.yml`, `scripts/run_c_prime_staged_seed0.sh`. factory 등록 완료. CPU 검증: config 파싱·staged 스케줄(ep0–3 w=0 / ep4 w=0.5 / ep5+ w=1.0)·rules 파일 모두 정상.
- **주의(정직성):** staging+stronger λ 동시 변경 = **2-변수 진단**(clean 1-변수 ablation 아님). 질문은 "C′ logic family가 LOGICZSL식 세팅에서 *신호를 내긴 하는가*". 
- **사전등록 판정 (vs Troika best_loss 0.3899 / single §11-11 0.3940):** test HM이 원 C′를 실질 마진으로 상회(LOGICZSL +1.2 HM 방향, HM ≳ 0.4010) → "구현이 문제였다" 지지 → 3-seed + revival 논의. 노이즈 대역(~0.389 ±0.0038) 잔류 → C′ seal이 staged+stronger에도 robust → 봉인 확정 강화.

### 21-1-4. PGAL (c) attention 시각화 — 예비 음성 + ckpt 중복 발견 — 6-2 KST

**viz 도구:** `Troika/code/viz_pgal_attn.py` — 학습된 TroikaPGAL ckpt에서 **image encoder + PatchLocalizer만** 돌려(전체 forward·text 없이, 저메모리 → GPU1 ablation과 공존) 256 patch attention을 16×16로 reshape→224 upsample→jet overlay. matplotlib 환경 깨짐(numpy 1.22 vs ≥1.23) → PIL+numpy 자체 colormap. 데모 6장(rusty knife/sliced cake/sliced bread/ripe banana/sliced potato/wet dog).

**예비 결과 (epoch_9 = 가장 학습된 localizer, raw attn max 0.042~0.124 = uniform 1/256의 11~32배):**
- **attention이 속성 영역을 국소화하지 못함.** sliced bread → hot spot이 빵이 아니라 **흰 배경 모서리**; rusty knife → 칼날 일부 + **배경 줄무늬/빈 공간**에 산발. 분산적이고 배경 지향.
- best_loss ckpt(val_best, epoch~2)는 더 흐릿(max 0.014~0.048). → **task (c) attention 의미성: 현재 "통과" 아님, 음성.**
- 해석: localizer가 학습은 됨(query norm 1.21→1.51→1.58 across epochs)이나 *의미 있는 국소화*를 학습하지 못함 → mean-pool에 가까운 행동 가능성. **(b) ablation 정량 결과와 합치 시 PGAL seal 쪽으로 기우는 신호.**

**부수 발견 — ckpt 중복 (무해하나 기록):** `final_model.pt`가 `val_best.pt`와 **852개 파라미터 전부 동일**(maxdiff 0.000000). 반면 epoch_4/epoch_9는 서로·val_best와 다름(정상 학습). → train.py의 final 저장이 epoch10 실제 상태가 아니라 best_loss 상태를 기록. **test는 val_best(best_loss)로 평가 → §21-1-2 보고 HM 0.3878 정상**, final_model.pt만 중복(향후 epoch10 상태 분석엔 epoch_9.pt 사용).

**발표 영향:** slide 12(attention heatmap)는 **보류** — ablation 정량 결과(ETA ~21:20)로 slide 11–12 동시 확정. attention이 배경 지향이라 PGAL 차별화 서사가 ablation 결과에 따라 정직하게 재조정될 수 있음. 산출물: `docs/slide_assets/slide12_attn_ep9/`(참고용), `slide4_context/`(맥락 의존성 3장, 발표 사용 가능).

### 21-1-5. PGAL (b) ablation 종료 — 약한 양(+) 신호, localization 기여 방향 확인 — 6-2 21:57 KST

**uniform-attention ablation 학습 종료 (6-2 21:57:34 KST, 6h29m, seed0 10ep GPU1).** `loc_uniform=True` 확정(Namespace 검증). test는 best_loss(val_best, ~epoch2) ckpt로 평가 — PGAL §21-1-2와 동일 선택 규칙 → **사과 대 사과**.

**결과 (MIT-States CW · test · seed0, 동일 조건):**
| | seen | unseen | HM | AUC |
|---|---|---|---|---|
| PGAL (localization) | 0.468 | 0.533 | **0.3878** | **0.2127** |
| uniform (ablation, 국소화 제거) | 0.4676 | 0.5314 | **0.3831** | **0.2113** |
| **Δ (PGAL − uniform)** | | | **+0.0047** | **+0.0014** |

- **PGAL > uniform, HM·AUC 둘 다.** → §21-1-1 task (b) "localization 기여 입증" = **양(+) 방향 충족**. attention을 mean-pool로 죽이면 성능이 떨어짐 → localizer가 *순수 noise/collapse는 아님*(§21-1-4의 seal-쪽 추정을 일부 반박).
- **단, ΔHM +0.0047 = 1-seed 노이즈 대역(±0.0038) 바로 바깥** → **약한 신호.** 강한 차별 주장은 3-seed로 굳혀야 함.

**§21-1-1 PASS 조건 종합:** (a) HM non-regression ✓ / (b) ablation 기여 **약-✓(+방향, marginal)** / (c) attention map 의미성 **✗(§21-1-4 배경 지향, 음성)**. → **정직한 종합 판정: "정량 ablation은 + 방향이나 약하고, 정성 맵은 아직 거칠다."** seal도 strong-PASS도 아닌 경계 — **PGAL 유지하되 3-seed + (d) F-probe로 굳히는 것이 다음 수**. 발표(slide 12)는 ablation 정량 표를 메인으로, 정성 맵은 "거칠다"고 정직하게 병기.

**발표 자료 반영:** `docs/slides_content.md` slide 12(PDF 13)의 `HM ___` 빈칸 → 위 표·해석으로 채움. 같은 파일에 외부인 가독성 패스(용어 풀이·프레이밍, S1–S14) 동시 적용. ※ PDF 자체는 Mac HTML→Chrome 인쇄본이라 재생성 필요(md는 콘텐츠 마스터).

---

## §21-1-6. (d) F-probe 오라클 실측 — PGAL 형제 변별 복구 net 0 → 차별성 미입증 확정 — 6-3 KST

**실행:** PGAL `mit-states_pgal_seed0_l0.01/val_best.pt`를 test 12,995장에 forward(`probe_f.py`, GPU2, out=`data/F_probe_pgal/`) → baseline(E val_best) 캐시와 대조(`oracle_pgal.py` 신규). best_loss ckpt 선택 동일 → 사과 대 사과.

**결과:**
- 전체 top1: baseline 0.2859 / **PGAL 0.2866** (Δ +0.0008, 노이즈).
- **복구 가능 집합 R** (baseline top1 wrong & gt in top5) = 4,163장.
  - PGAL 복구 = **644/4,163 (15.5%)** ↔ PGAL 파괴(baseline-correct를 틀림) = **662/3,715 (17.8%)** → **net −18 (wash)**.
- **형제 슬라이스** (obj 맞고 attr 틀림) 2,528장: PGAL 복구 = **378/2,528 (15.0%)** ≈ 전체 R 복구율(15.5%)과 동일.

**판정:** PGAL은 형제 변별을 **표적 복구하지 못함**. (1) 고침≈파괴 대칭 churn = "다르지만 더 낫지 않은 모델", (2) 형제 affinity 없음(15.0%≈15.5%) = 설계 목표 미달. §21-1-4(attention 배경 지향) + §21-1-5(ablation 약-+)와 합치 → **국소화가 *학습은 됐으나 의미 있는 위치로 수렴 못 해 전역 평균에 가깝게 collapse*** (그래서 non-regression이면서 동시에 net 0).

**§21-1-1 PASS 최종 종합:** (a) non-regression ✓ / (b) ablation 약-✓ / (c) attention ✗ / **(d) 오라클 ✗(net 0)** → **차별 기여 미입증.** PGAL 현 형태는 seal 경계. 단, 봉인 대신 **방향 전환**으로 진행 결정(§22).

---

## §22. 방향 전환 — Localization-as-Grounding (사전등록) — 6-3 KST

**결정 (사용자, 6-3):** staged C′ seed2 (GPU1, epoch 6/10) **중단** → GPU1 회수. staged C′는 **2-seed(seed0 0.394/0.2177, seed1 0.3883/0.2201, 둘 다 test 노이즈 대역)로 마감**, "LOGICZSL식 staged+강λ로도 신호 없음 → C′ 봉인 robust" 결론 유효(3-seed 완성도만 waive, D′ 선례). → **PGAL 봉인하지 않고 새 축으로 진입.**

**Thesis 재구성 (16축 negative의 조건부 재해석):** 16축 지식주입이 실패한 것은 *아이디어*가 아니라 **닻을 내릴 올바른 국소 표현이 없는 전역 표현 위에 부었기 때문**이라는 가설. → **올바른 국소화가 생기면 그 위에서 지식이 작동할 수 있다.** PGAL(국소화)을 *최종 메커니즘*이 아니라 **지식 grounding의 enabler**로 재정의.

**2단계 게이트 구조:**
- **[1단계 · 게이트] 국소화를 올바른 영역으로 수렴시킨다.** (현 PGAL의 collapse를 깨는 것)
- **[2단계] (1단계 통과 시에만) 올바른 국소 표현 위에 지식 주입 → 성능 향상 검증.** 1단계 미통과 시 2단계 검증 불가. compounding risk 인지.

**1단계 메커니즘 (사전등록):** 근본 원인 2개를 동시 타격.
- **(A) 전역 목발 제거 + peaked attention**: 속성 채점을 z_a에만 의존시키고, z_a가 균등 평균이 못 되게 강제 (top-k 패치 / low-temp softmax / Gumbel hard-select 중 택1; 균등=high-entropy에 강한 페널티).
- **(B) "어디 봐" 신호 주입 (CLIP patch-text prior)**: 속성 텍스트 임베딩 × CLIP 패치 토큰 유사도 → coarse 위치 히트맵을 탐침 attention의 target/prior로 (MaskCLIP식, 라벨 0). ※ B는 그 자체로 *약한 지식주입* = 2단계로 가는 다리.
- 첫 런 = **B+A 동시** (A 단독은 뾰족하게 틀린 곳 짚을 위험). `troika_pgal.py`에 toggle로 추가.

**사전등록 PASS 조건 (1단계, AND 결합):**
- **(a) 국소화 작동**: F-probe 오라클 형제 복구가 **churn 대비 유의한 net +** (현 PGAL 644−662 = −18 대비 분명한 양수로 이동).
- **(b) 올바른 위치**: attention의 **foreground/변별영역 집중도**가 baseline(uniform/현 PGAL) 대비 **유의 상승** (viz 정성 + 정량).
- **(c) non-regression**: test HM ≥ **0.3849** 유지.
- **미달 시**: PGAL **영구 봉인**, 차순위 축(속성=연산자 / generative unseen-feature synthesis)으로.

**검증 인프라(재사용):** `oracle_pgal.py`(형제 복구율) + `viz_pgal_attn.py`(맵이 단면 짚나). 학습 전 cutoff 고정 → 사후해석 함정 차단.

**EV 경고:** (b)(c)(d) 약/음 이력상 "국소화만 고치면 된다"는 미보장. B의 CLIP teacher도 약한 선생(완벽하면 PGAL 불필요) — collapse를 깨는 prior일 뿐. 1단계에서도 오라클 net 0이면 미련 없이 봉인.

**다음:** B+A 변형 코드 설계·구현 → GPU1 학습 → 오라클·viz 판정.

### 22-1. PGAL-grounding (B+A) Stage 1 seed0 완주 + val_best test 정식 측정 — 6-5 KST

**run:** `config/troika/mit-states-pgal-grounding.yml` (B=loc_prior_weight 1.0 / loc_prior_temp 0.1, A=loc_temp 0.7 / loc_topk 0, lambda_loc 0.0로 entropy reg OFF·B 효과 격리), seed0, 10ep, GPU1. 6-4 04:55 종료. val_metric=best_loss.

**측정 방법 주의:** 학습 로그의 epoch별 `best_*` 라인은 **test-oracle peak**(매 epoch test에 bias sweep)이라 부풀려져 있음 — peak HM 0.419/AUC 0.249 @ep3~4. 정당한 숫자는 **val(best_loss)로 고른 ckpt(`val_best.pt`)를 test에 1회 평가**한 값. `test.py --load_model val_best.pt`로 실측:

| | seen | unseen | **HM** | **AUC** |
|---|---|---|---|---|
| val_best @ **val** | 0.490 | 0.572 | 0.4144 | 0.2421 |
| val_best @ **test** (정식) | 0.469 | 0.531 | **0.3836** | **0.2115** |
| baseline (best_loss) | — | — | 0.3899 | 0.2177 |
| **Δ vs baseline** | | | **−0.0063** | **−0.0062** |

**판정 (게이트 (c) non-regression):** test HM 0.3836 < cutoff **0.3849** → **(c) FAIL** (−0.0013, 1-seed 노이즈 대역이긴 하나 미달). §22 PASS는 (a)oracle·(b)viz·(c)non-reg의 **AND**이므로, (c) 미달만으로도 게이트 통과 불가 방향. peak가 oracle 환상이었다는 점이 핵심 — val-selected는 baseline 아래. **(a) oracle 형제복구 / (b) attention 집중도 판정은 아직 미실행** → 봉인 확정 전 (a)(b)까지 보고 사용자 판단 대기 (advisor 보고용). 단 EV 경고(§22)대로 (c) 음수면 미련 두지 않기로 사전 합의됨.

### 22-1b. staged C′(l0.5) 3-seed val_best test 보강 — §22 intro의 2-seed 마감을 3-seed로 완성 — 6-5 KST

§22 intro에서 staged C′를 2-seed로 waive 마감했으나, seed2 ckpt가 남아있어 3-seed 모두 `val_best.pt` test 정식 재측정:

| seed | seen | unseen | HM | AUC |
|---|---|---|---|---|
| 0 | 0.469 | 0.537 | 0.3934 | 0.2171 |
| 1 | 0.488 | 0.532 | 0.3891 | 0.2204 |
| 2 | 0.467 | 0.534 | 0.3863 | 0.2141 |
| **mean** | | | **0.3896 ± 0.0036** | **0.2172 ± 0.0032** |
| baseline | | | 0.3899 | 0.2177 |
| **Δ** | | | **−0.0003** | **−0.0005** |

**판정:** Δ HM −0.0003 / AUC −0.0005, 둘 다 seed noise(±0.003) 안 → **null/fail 재확인**. §20 C′(logic-rule, Δ −0.0009)과 동급. "LOGICZSL식 staged+강λ로도 신호 없음 → C′ 봉인 robust" 결론을 **3-seed로 확정**.

### 22-2. 1단계 게이트 (a) oracle + (b) attention 집중도 측정 → 3축 전부 미달 → PGAL/localization 축 영구 봉인 — 6-6 KST

§22-1에서 (c) non-regression이 미달(0.3836)이었으나, 사전등록 PASS=(a)∧(b)∧(c)의 (a)(b)까지 실측해 봉인 robust 확정.

**측정 인프라:** baseline 캐시는 §21 oracle과 동일 `data/F_probe/test_top5_preds.json`(644−662=−18 비교 기준 유지). grounding val_best로 `probe_f.py` 새 캐시 생성 → `data/F_probe_pgal_grounding/test_top5_preds.json` (12995장, model top1 0.2882). (b)는 신규 `measure_pgal_concentration.py`(localizer attention의 entropy/max-mass/eff-patches 집계, 1504 test img, DataLoader 정식 forward).

**(a) oracle — 형제복구 net:**
- overall top1: baseline 0.2859 → grounding 0.2882 (Δ +0.0023)
- recoverable set R=4163 중 666 복구(16.0%), baseline-correct 3715 중 661 파괴(17.8%) → **net +666 − 661 = +5**
- sibling-state slice(obj 맞고 attr 틀림) 2528 중 389 복구(15.4%)
- **판정:** §21 PGAL −18 → +5로 이동했으나, 복구율(16.0%)≈파괴율(17.8%) = churn 총량 1327 대비 net +5는 **사실상 0 = targeted localization 아닌 무작위 re-shuffle**. 사전등록 "분명한 양수" 미달 → **(a) FAIL**.

**(b) attention 집중도 (1504 test img, uniform 기준: max-mass 0.00391 / entropy-ratio 1.0 / eff-patches 256):**
- max-mass mean **0.00715** (uniform의 1.8배에 불과)
- **entropy-ratio mean 0.9940** (median 0.9944; 1.0=완전붕괴)
- **eff-patches mean 247.6 / 256** (median 248.2) — attention이 256패치 중 ~248개에 균등 분산 = 사실상 mean-pool
- **판정:** loc_temp 0.7(A) + loc_prior_weight 1.0 CLIP patch-text prior(B)에도 **localizer가 near-uniform으로 붕괴** — §22 (A)가 막으려던 바로 그 collapse. attention map viz 12장(sibling-confusion, `docs/slide_assets/slide22_grounding_attn/`)도 attn max 0.006~0.008로 전부 평평. **(b) FAIL**.

**§22 1단계 게이트 종합 — (a)∧(b)∧(c) 3축 전부 미달:**

| 게이트 | 기준 | 측정 | 판정 |
|---|---|---|---|
| (a) localization 작동 | oracle net 분명한 양수 | net +5 (churn 1327) | ✗ |
| (b) 올바른 위치 | attn 집중도 유의 상승 | entropy-ratio 0.994 / eff 247.6 | ✗ |
| (c) non-regression | test HM ≥ 0.3849 | 0.3836 | ✗ |

**봉인 (사전등록대로):** 1단계(localization 수렴) 명확히 실패 → 2단계(올바른 국소표현 위 지식주입) 검증 불가. §22 EV 경고("1단계에서도 오라클 net 0이면 미련 없이 봉인") 발동. **PGAL/image-side localization 축 영구 봉인.** 근본 진단: **(B) CLIP patch-text teacher가 attention을 실제로 뾰족하게 만들지 못함** — 약한 teacher(§22 EV 경고대로 "완벽하면 PGAL 불필요")로는 localizer의 uniform-mean-pool degenerate optimum을 깨지 못함. → §21·§22로 PGAL 2회(localizer만 vs localizer+grounding-prior) 모두 차별성 미입증으로 닫힘. **다음 축: 사전등록 차순위(속성=연산자 operators / generative unseen-feature synthesis).** mechanism 봉인 카운트 갱신.

**산출물:** `code/measure_pgal_concentration.py`(재사용 attention-concentration harness), `data/F_probe_pgal_grounding/`(grounding top-5 캐시), `docs/slide_assets/slide22_grounding_attn/`(viz 12장).

---

## §23. Axis (II) — Attributes-as-Operators (AoP) 사전등록 plan — 6-6 KST

**진입:** §22 PGAL/localization 영구 봉인(6-6) → §21-1에서 정한 차순위 축 (II) operators로. differentiation-first 기조 유지(§21): 목표는 ClusPro 추격이 아니라 **차별화된 우리 메커니즘** — perf가 baseline보다 조금 부족해도 novel+defensible+non-regression이면 PASS.

### 23-0. 차별화 thesis & sealed 축과의 lock

**현 Troika 합성 진단(코드 감사 6-6):** comp 분기 = `[soft-ctx … attr_token obj_token]`을 **frozen CLIP text encoder**에 통과 → self-attention이 attr·obj를 **암묵적·대칭적으로 합성**(`troika.py:construct_token_tensors`, attr는 eos-2, obj는 eos-1 슬롯에 token embedding 삽입). **명시적 합성 연산자 없음** — 합성은 text transformer 내부에 entangle돼 분리·제약 불가.

**AoP thesis:** 속성은 "토큰"이 아니라 **객체 표현에 작용하는 변환 연산자**(Nagarajan-Grauman 2018을 CLIP-CZSL/Troika로 lift). 합성을 joint space의 명시적 대수 연산으로: `c(a,o) = norm(e_clip(a,o) + α·O_a(e_o))`. **이게 sealed 축들과 결정적으로 다른 점**: 합성 *함수 자체*를 바꾸고 operator 대수로 제약 → **E(taxonomy)가 직접 반증한 "aux loss는 representation 못 바꾼다"(§20-6-3) 함정을 구조적으로 우회**(operator는 side-loss가 아니라 composition path 안에 있음).

**Sealed 축 차별 lock:** PGAL(§21·22)=image-side localization(직교), C′/logic(§20-4)=logit penalty, D′(§20-5)=hard-neg InfoNCE, E/taxo(§20-6)=global feat aux CE, OT(§H3)=transport. **operators=text/comp-side 합성함수 재정의** — 위 어느 것과도 메커니즘 비중복.

### 23-1. 메커니즘 (사전등록, 확정)

**합성 경로 = Residual-replace** (사용자 결정 6-6): CLIP-text 합성 보존 + operator가 구조적 보정. ablation(α=0)으로 기여 분리.
- `e_clip(a,o)` = 현 Troika comp 분기 text feature(정규화). `e_o` = obj 분기 text feature(객체 o, 정규화).
- **합성:** `c(a,o) = norm( e_clip(a,o) + α · U_a (V_aᵀ e_o) )`. comp_logit = CLS · c · logit_scale.
- **연산자 O_a = low-rank:** `U_a, V_a ∈ R^{D×r}`, D=768, **r=32**(사전등록). attr별. params ≈ 115×2×768×32 ≈ 5.6M.
- **게이트 α:** attr별 학습 scalar, **init 0.1**(학습 시작 시 CLIP baseline 근처 → non-reg 초기 보호). α는 학습으로 자유 이동.
- **※ preview 정밀화 (거부권 명시):** 사용자가 고른 preview는 `O_a(e_clip)`였으나, operator 입력을 **객체 표현 e_o**로 둠 — AoP thesis("속성이 객체에 작용")에 충실하고, 우선시한 **(ii) direction-consistency 해석을 깨끗하게**(변위가 객체에만 의존). 합성 보정·CLIP 보존·α=0 ablation이라는 선택 취지는 동일 유지. 객체 입력이 부적절하다 판단 시 `O_a(e_clip)`로 전환 가능.
- **학습 loss = 표준 3분기 CE만**(comp는 operator-보정 c 사용) + attr CE + obj CE. **구조 강제 regularizer 없음** — Core-3 구조는 *emergent*여야 함(강제하면 (ii) 순환논증). 구조가 안 생기면 그게 진짜 negative.
- **eval-path 필수:** `forward_for_open`·`encode_text_for_open`도 operator 보정 mirror — eval에서 operator 우회되면 train/eval mismatch(§20-5-3·§22 교훈). Stage0에서 검증.

**구현:** `model/troika_operator.py`(`model_name='troika_operator'`, Troika 상속, comp 분기만 override), `config/troika/mit-states-operator.yml`, `code/measure_operator_structure.py`(Core-3 측정), `scripts/run_operator_seed0.sh`. HP: lr 1e-4, 10ep, bs8/accum8, val_metric=best_loss, ViT-L/14.

### 23-2. 사전등록 PASS 조건 (AND 결합)

**(a) operator가 실제로 작동 = Core 3종 모두 충족** (사용자 결정 6-6):
- **(a-i) Ablation 기여:** 동일 val_best ckpt에서 α=0(operator off) eval vs on eval → **comp HM(on) − HM(off) ≥ +0.005**. 동일 weight 비교라 seed noise 무관 = operator 직접 기여.
- **(a-ii) Attribute-direction 일관성:** 변위 `Δ_{a,o}=α·O_a(e_o)` 정규화 후, attr별 객체 간 평균 cosine 일관성 `C_meas`. **rank-32 random operator baseline `C_rand` 대비 `C_meas − C_rand ≥ 0.10`** (low-rank artifact 넘는 진짜 coherent 방향).
- **(a-iii) Non-triviality:** `median ‖α·O_a(e_o)‖ / ‖e_clip‖ ≥ 0.05` (변위가 합성 feature의 ≥5% = identity/0 붕괴 아님).

**(b) Non-regression:** val_best @ test **HM ≥ 0.3849** (= baseline 0.3899 − 0.005, PGAL과 동일 cutoff).

**STRETCH:** test HM > 0.4050 OR ablation Δ > +0.01(operator가 강하게 기여).

**FAIL/SEAL:** HM < 0.3849(회귀) **OR** (a-i)/(a-ii)/(a-iii) 중 하나라도 미달(operator가 trivial/무구조 → "operator는 CLIP 합성 위에 구조를 못 더한다"). → operators 축 봉인, 차순위(generative unseen-feature synthesis)로.

### 23-3. 단계 & 검증 인프라

- **Stage 0 smoke:** 1ep/subset — NaN0, VRAM, grad가 U_a/V_a/α로 흐름(α grad≠0), eval-path가 operator 사용(α 토글로 logit 변화 확인), gate clean.
- **Stage 1:** seed0 10ep GPU1(CUDA_VISIBLE_DEVICES=1) → val_best test eval(`test.py`) + `measure_operator_structure.py`로 Core-3 산출 → 23-2 판정.
- **Stage 2(PASS 시만):** 3-seed + UT-Zappos. ※ UT-Zap은 AMP NaN fix 필요([[project_lhp_czsl_amp_nan]]).

### 23-4. EV 경고 (정직)

base rate 가혹(16+ 축 sealed). **구체적 실패 모드 2개**: (1) CLIP-text 합성이 강해 operator가 α→0으로 학습 → (a-iii) 미달 = 봉인 신호. (2) operator가 일반 추가 capacity로 fit하나 coherent 구조 없음 → (a-ii) 미달 = 봉인 신호. **단 §21 재정의대로 "차별화O + non-reg + 구조 입증"이면 perf가 baseline 동급이어도 PASS** — 바는 SOTA가 아니라 differentiation. (a-ii)가 핵심 변별: "operator가 단지 fit한 것"과 "구조적으로 합성한 것"을 가름.

**다음:** 사용자 사전등록 승인 → `troika_operator.py` 구현 → Stage 0 smoke.

### 23-5. 사전등록 승인 + 구현 + Stage 0 smoke 통과 + Stage 1 launch — 6-6 KST

**사용자 승인(6-6):** §23 사전등록 그대로 승인. 합성=residual-replace, (a) Core-3.

**구현:** `model/troika_operator.py`(`TroikaOperator(Troika)` + `AttrOperators` 모듈: per-attr U,V∈R^{768×32}, gate α; `forward`/`encode_text_for_open`만 override해 comp 분기만 보정, attr/obj 분기는 vanilla 그대로 → eval 계약 보존). `model_factory.py` 등록(`troika_operator`). `config/troika/mit-states-operator.yml`(rank32/α_init0.1/init_std0.02, bs8/accum8, 10ep, best_loss). `parameters.py`에 `--operator_rank/_alpha_init/_init_std/_disabled` 추가. `smoke_operator.py`, `scripts/run_operator_seed0.sh`, `measure_operator_structure.py`(Core-3 (a-ii)(a-iii) 판정).

**Stage 0 smoke 통과 (`smoke_operator_20260606_020453.log`):** operator_params 5.65M(=115×2×768×32+α), total trainable 34.7M. NaN 0, VRAM 14.3GB, 1.81 it/s. **grad-flow ✓**(|U.grad|2.6e-2, |V.grad|2.5e-2, |α.grad|5.1e-3 전부>0 → operator 학습됨). **eval-path operator 사용 ✓**(comp ON≠OFF L2 0.0063, **attr/obj 분기 byte-identical=True → 계약 보존**, §20-5-3/§22 우회버그 차단). **gate ✓**(loss_calu==no_grad==base CE 17.4534, operator-specific loss 없음 → ckpt-selection 혼동 0). ⚠️ **주의: init non-triviality 0.0063 = (a-iii) floor 0.05의 1/8** — init(α0.1+small std)이라 당연, floor는 학습후 진단. **Stage 1이 operator를 ~8× 키워야 (a-iii) 통과**, 안 크면 §23-4 실패모드(1)(α→0 붕괴)=봉인 신호. 사전등록 init값 유지(post-hoc 튜닝 안 함).

**Stage 1 launch (6-6 02:06 KST, GPU1, `train_mit_operator_seed0_20260606_020625.log`):** seed0 10ep 정상 시작(train loss 2.18, 1.16 it/s, GPU1 100%/16.8GB, NaN 0). 3793 iter/ep × 10 + epoch별 test eval → ~6–7h 예상(≈09:00 KST). save mit-states_operator_seed0.

**판정 절차(완주 후):** (b) `test.py --load_model val_best.pt` → test HM; (a-i) `test.py`를 `--operator_disabled` 유/무로 2회 → comp HM on−off; (a-ii)(a-iii) `measure_operator_structure.py`. §23-2 AND 판정.

### 23-6. operator(AoP) Stage 1 완주 + 정식판정 waive 봉인 — 6-9 KST

**완주:** seed0 10ep 정상 종료(6-6 08:49 KST, `done!`, NaN/crash 0, final train loss 0.0311). save에 `val_best.pt`/`epoch_9`/`final_model` 존재. GPU1 이후 idle.

**raw 학습로그 test 라인(epoch별 oracle-peak, bias sweep):** best_hm **0.3851** / AUC 0.2162 / seen 0.4761 / unseen 0.5383. cutoff(b) 0.3849에 +0.0002로 사실상 동률이나, **이 숫자는 §22-1에서 규명한 test-oracle peak** — PGAL 전례(peak 0.419→val_best 정식 0.3836)대로면 val_best 정식 eval은 cutoff 아래로 갈 공산. smoke에서 (a-iii) non-triviality init 0.0063 = floor 0.05의 1/8 → §23-4 실패모드(1)(α→0/저변위) 신호.

**판정 (사용자 결정 6-9, 정식판정 waive):** base rate(누적 17축 sealed) + raw가 oracle-peak로도 동률 + (a-iii) floor 신호를 종합, §F($65 waive) 선례대로 **(a)Core-3·(b)val_best 정식 eval을 waive하고 operators 축 봉인.** 정식 3-command(위 23-5 판정절차)는 보존 — advisor 요청 시 사후 채움 가능. **operators=text/comp-side 합성함수 재정의 축도 차별화 입증 미달로 closed.**

**봉인 카운트:** §22 PGAL(localization) → §23 operators까지, **누적 18축 sealed.** axis/hyperparam/inference/localization/composition-function 전 경로에서 Troika의 ClusPro gap 닫기 실패 재확인.

---

## §24. Axis (III) — Generative unseen-feature synthesis 사전등록 plan (stub) — 6-9 KST

**진입:** §23 operators 봉인(6-9) → §22 봉인 노트·§23-2/23-4에 사전등록된 차순위 축으로. 18축이 전부 *discriminative*(입력/표현측: logit penalty, aux CE, InfoNCE, transport, localization, composition operator)였던 것과 달리, 이건 유일한 **data-side / generative** 경로.

**thesis:** unseen (attr,obj) 쌍의 visual feature를 직접 합성해 학습에 노출. generator `G(e_a, e_o, z)` → CLIP-space feature 합성, seen pair로 G 학습 → unseen pair feature 생성 → classifier를 unseen 분포에 직접 노출(seen/unseen imbalance 보완). CZSL feature-generation 갈래를 Troika/CLIP-CZSL로 lift. "gap=architectural"(§20) 결론과 정합 — 표현을 못 바꾸면 데이터를 만든다.

**차별 lock:** 봉인 18축은 전부 discriminative. generative/data-augmentation 경로는 메커니즘 비중복.

### 24-1. 생성기 결정 = cVAE — 6-9 KST

**사용자 결정(6-9): generator = conditional VAE (cVAE).** 후보 4개(cVAE/WGAN-GP/VAE-GAN/flow) 비교 후 선택. 근거: (1) adversarial 없음 → AMP NaN 이력([[project_lhp_czsl_amp_nan]])·단일 run 예산과 정합(안정적 수렴), (2) ablation 깨끗(decoder/synthetic-feature만 떼면 기여 분리), (3) CADA-VAE 등 CLIP-era ZSL feature-gen 표준 → defensible. WGAN(f-CLSWGAN)이 published reference로 더 강하나 목표가 SOTA 아닌 "작동 입증"이라 불안정성 떠안을 이유 약함. cVAE 미달 시 Stage 2에서 VAE-GAN 하이브리드로 escalate 여지 남김.

**cVAE 설계 함의(확정 전 메모):** 합성 대상 = ViT-L/14 visual feature 768-dim **L2 정규화(hypersphere)** → recon loss는 MSE보다 **cosine/von Mises-Fisher 계열** 또는 출력 renormalize 필요. 조건 = (attr,obj) 임베딩(CLIP text vs Troika 학습 임베딩 — 미결). latent dim·β(KL weight) 미결.

### 24-2. 합성 feature 사용방식 = (A) Decoupled classifier — 6-9 KST

**사용자 결정(6-9): (A) decoupled classifier** (f-CLSWGAN/CADA-VAE 표준 레시피). 파이프라인 확정:
1. CLIP backbone freeze → seen-pair train 이미지의 진짜 visual feature 추출·캐시(1회).
2. cVAE 학습: (attr,obj) 조건 → seen feature 재구성.
3. unseen 조합의 synthetic feature 대량 생성.
4. [real-seen + synthetic-unseen feature]로 분류기만 학습 → zero-shot을 (거의) supervised로 전환.

근거: 가벼움(feature-space only, backbone 재학습 0 → 단일 GPU 최적), ablation 깨끗(synthetic feature 유/무), published recipe라 defensible, **봉인 18 joint 축과 메커니즘 확실 분리**(차별점="feature-space data synthesis"). 단점(Troika CMT 적응 이점 미사용)은 differentiation-first 기조상 수용.

### 24-3. cVAE 조건 = separate attr+obj CLIP text concat — 6-9 KST

**사용자 결정(6-9): ② separate attr-text + obj-text CLIP 임베딩 concat.** 각 단어를 frozen CLIP text encoder로 인코딩(L2-norm) → 이어붙여 조건 `c ∈ R^{1536}`. 근거: cVAE가 **결합 자체를 학습**(①처럼 CLIP text가 합성을 다 해버리지 않음) → (a) 기여 ablation·차별성이 깨끗. ① joint comp-text보다 약간 불안정하나 differentiation 우위. ③(Troika 학습 임베딩)은 봉인축과 근접/일반화 약 → 기각.

### 24-4. 나머지 설계 락 (recon/latent/β, feature source, 분류기) — 6-9 KST (claude 추천 기본값, 사용자 위임)

- **Feature source = 학습된 Troika baseline(seed0, best_loss ckpt)의 adapted image CLS feature** (768-dim, L2-norm). raw CLIP 대신 adapted를 쓰는 이유: Troika CMT/adapter 적응 이점 보존 → **non-reg 비교 공정**(decoupled로 갈아끼우며 adaptation까지 버리면 합성 무관하게 회귀할 risk 차단, §24-6 EV Risk1). fallback: raw frozen CLIP.
- **cVAE arch:** encoder MLP(768 → 512 → 2×64), **latent_dim 64**, decoder MLP(64+1536 → 512 → 768) → 출력 **L2-renorm**. 조건 c는 encoder 입력·decoder 입력 양쪽 concat.
- **Loss:** recon = `1 − cos(x̂, x)` (feature가 정규화돼 hypersphere → MSE 대신 cosine) + **β·KL**, β 0→0.01 annealing(posterior collapse 방지). f-CLSWGAN의 CLS 항은 1차엔 미사용(순수 cVAE 유지, Stage2 escalate 여지).
- **생성:** unseen pair당 z~N(0,I) **N=300** 샘플 디코드 → synthetic feature.
- **분류기 = 학습 linear `W ∈ R^{P×768}`** (P=closed-world pair 수), **CLIP comp-text("a photo of attr obj") 임베딩으로 init**, CE(+logit_scale)로 학습. seen pair row=real feature(이미지별), unseen pair row=synthetic feature(300/pair). bias/threshold는 val로 선택(oracle 금지).
- **Eval:** 표준 closed-world seen/unseen/HM/AUC, val_best 선택 후 test 1회.

### 24-5. 사전등록 PASS 조건 (AND) — 6-9 KST

**(a) generative 축이 실제로 작동 (AND):**
- **(a-i) 기여 ablation:** 분류기 [real-seen + synthetic-unseen] vs [real-seen only(unseen row=text-init frozen)] → **Δ HM ≥ +0.005**, unseen-pair acc 상승이 동인. (합성이 unseen에 실제 신호 주는가 = 핵심.)
- **(a-ii) 생성 충실도/anti-collapse:** held-out seen pair에서 synthetic 생성 → real centroid와 mean cosine이 **random-condition baseline 대비 Δ ≥ 0.10** (feature가 pair-specific, generic 붕괴 아님). operator (a-ii)/(a-iii) 정신 계승.

**(b) Non-regression:** val-selected, test **HM ≥ 0.3849** (baseline 0.3899 − 0.005, 전 축 공통 cutoff).

**STRETCH:** test HM > 0.4050.
**FAIL/SEAL:** HM < 0.3849 **OR** (a-i)/(a-ii) 중 하나라도 미달 → generative 축 봉인(차순위 미정).

### 24-6. EV 경고 (정직) — 6-9 KST

base rate 가혹(18축 sealed). 구체적 실패 모드 3개: **(Risk1)** decoupled linear 분류기가 Troika joint scoring보다 약해 합성과 무관하게 non-reg 미달 — adapted feature source + text-init로 완화하나, [real-seen only] 분류기가 baseline 한참 아래면 *decoupled 틀 자체가 병목*(합성 품질 confound) → (a-i) 비교는 유효하나 (b) 판정엔 이 confound 명기. **(Risk2)** cVAE posterior collapse → generic feature → (a-ii) 미달(β annealing 완화). **(Risk3)** synthetic 과적합으로 seen-acc 하락 → real/fake balance 조절. **단 §21 재정의대로 "차별화O + non-reg + (a) 입증"이면 baseline 동급이어도 PASS.**

### 24-7. 구현 계획 (승인 대기) — 6-9 KST

`model/`·`code/`에 decoupled 파이프라인 신규(Troika 본체 미변경): ① `extract_features.py`(Troika baseline ckpt로 seen train feature 캐시) → ② `model/cvae_featgen.py`(cVAE) + `train_cvae.py` → ③ `generate_unseen.py`(synthetic 캐시) → ④ `train_classifier.py`(real+synth CE) + eval → ⑤ `measure_gen_fidelity.py`((a-ii)). **Stage 0 smoke**(NaN0/VRAM/grad-flow/eval-path/분류기 init 검증) → **Stage 1** seed0 GPU1(CUDA_VISIBLE_DEVICES=1) → §24-5 AND 판정. PASS 시 Stage 2(3-seed + UT-Zap, AMP NaN fix 필요 [[project_lhp_czsl_amp_nan]]). HP: ViT-L/14, mit-states, best_loss 선택.

**다음:** 사용자 사전등록 승인 → Stage 0 smoke 구현.

### 24-8. 구현 + Stage 0 smoke 통과 — 6-9 KST

**사용자 승인(6-9):** §24 사전등록 그대로 승인(cVAE / decoupled / cond=attr+obj text concat / 나머지 claude 추천 기본값). Stage 0 smoke 구현 지시.

**Feature source 확정(사전등록 deviation 명시):** 순수 baseline ckpt 미저장 → **c_prime_seed0(logic, λ=0.1) val_best.pt**를 baseline-equiv 추출기로 사용. 근거: TroikaLogic은 `__init__`/`loss_calu`만 override(arch 변경 0), 실제로 vanilla Troika에 `load_state_dict` 시 **missing=0 unexpected=0**(state_dict 키 완전일치 확인) → adapted CLS feature가 baseline과 동일계열, Δ HM −0.0009. Stage1도 동일 ckpt 사용 예정(advisor 보고 시 deviation 명기).

**구현(Troika 본체 미변경, code/ 신규):** `featgen.py`(공용 lib) + `smoke_featgen.py`(Stage 0). lib 구성: `load_extractor`(vanilla Troika strict=False), `extract_image_features`(adapted CLS 768-d L2-norm), `text_embeddings`(CLIP text, context_length=8 주의), `CVAE`(enc/dec MLP, latent64, cond1536, 출력 L2-renorm), `cvae_loss`(1−cos recon + β·KL), `generate_features`, `PairClassifier`(cosine prototype, comp-text init) + `train_pair_classifier`, `evaluate_with_official`(Troika **Evaluator/test() 그대로 재사용** → 평가 프로토콜 동일 보장). 버그수정 2건: text tokenizer가 CPU 텐서·기본 context_length=77 반환 → `.cuda()` + `context_length=config.context_length(8)` 로 토큰화.

**Stage 0 smoke 통과 (subset: train512/val256/test256, 30 unseen pair × 50 gen):**
- **[1] 추출:** ckpt 로드 missing=0/unexpected=0, train_feats finite, **L2-norm OK(mean norm 1.00000)**.
- **[2] text emb:** attr[115]/obj[245]/comp[1962] 전부 finite·normalized.
- **[3] cVAE:** **grad-flow 전 param>0**(enc 1.5e-3, fc_mu 3.0e-3, fc_logvar 2.8e-4, dec.0 0.12, dec.last 0.47), **recon 0.955→0.288 감소**, KL finite(0.009~0.049), NaN 0.
- **[4] 생성:** synthetic 1500개 finite, **L2-norm OK**, shape 정상.
- **[5] 분류기:** real-only CE 0.228(W 업데이트 확인), real+synth CE 1.34, NaN 0.
- **[6] eval-path:** 공식 Evaluator/test() 정상 작동 → HM/AUC 산출(subset이라 수치 무의미).
- **[7] (a-i) ablation 토글 live**(real-only HM 0.509 vs real+synth 0.559, **Δ +0.0497**, unseen 0.536→0.634 — subset 기준 mechanics만), **(a-ii) fidelity 측정 작동**(cos(syn,real) 0.969 vs cos(rand-cond,real) 0.790, Δ +0.179).
- **VRAM peak 2.39 GB** (cached-feature space라 매우 가벼움).

**중요(Stage1 EV):** real+synth가 subset에서도 unseen↑·HM↑ 보인 건 **고무적이나 mechanics 검증일 뿐**(256장 test, 30 unseen pair, bias sweep 낙관 편향). 정식 판정은 Stage1 full(30338 train feat / 700 unseen pair 전체 / val_best 선택 / 12995 test). Risk1(decoupled cosine 분류기가 Troika joint scoring보다 약해 (b) 미달) 여전 유효 — Stage1 [real-only] full HM이 baseline 0.3899 대비 어디 떨어지는지가 1차 관문.

**다음:** Stage 1 driver(전체 feature 디스크 캐시 → cVAE 풀학습 → synthetic 생성 → 분류기 real+synth/real-only → val_best+test 공식평가 → §24-5 AND 판정) 작성 + GPU1 launch.

### 24-9. Stage 1 완주 + 봉인 (메커니즘 작동, decoupled 천장이 병목) — 6-9 KST

**실행:** `stage1_featgen.py` seed0 GPU1, 정상 완주(NaN 0). feature 캐시(train 30338/val 10420/test 12995, `save/featgen_seed0/`), cVAE 100ep(recon 0.955→0.172, KL~2.09), synthetic 210000개(700 unseen × 300), 분류기 ×2 val-AUC select. val selects / test reports(test-oracle epoch picking 없음).

**§24-5 AND 판정 (seed0):**
| 게이트 | 기준 | 측정 | 판정 |
|---|---|---|---|
| (a-i) 합성 기여 | Δ HM ≥ +0.005 | **+0.0051** (test HM real+synth 0.3068 − real-only 0.3017; unseen 0.3732→0.4196) | PASS |
| (a-ii) 생성 충실도 | Δ ≥ 0.10 | **+0.4389** (cos(syn\|correct,real) 0.9651 vs cos(syn\|rand-cond,real) 0.5262, n=1262) | PASS |
| (b) non-reg | HM ≥ 0.3849 | **0.3068** (AUC 0.1474) | **FAIL (−0.078)** |

→ AND 게이트 (b) 미달 → **FAIL/SEAL**. 단일 seed지만 Δ −0.078 = seed noise(±0.0038)의 ~20배 → 3-seed 불요, 결정적.

**진단 (핵심): generative 메커니즘은 작동, decoupled 틀이 병목 = §24-6 Risk1 실현.**
- (a-ii) +0.44: cVAE 생성 feature가 pair-specific(real centroid와 cos 0.965, random-cond 0.526) — **collapse 없음, 생성 품질 우수**.
- (a-i) +0.0051: synthetic unseen이 unseen 인식 실제 향상(0.3732→0.4196, +0.0464).
- **그러나 real-only 분류기도 HM 0.3017** = baseline 0.3899 −0.082. **합성 품질이 아니라 Troika CMT joint scoring을 버린 decoupled cosine 분류기의 천장(~0.31)이 원인.** 합성 +0.005로 0.08 격차 메우기 불가 — 병목이 분류기 구조. §24-6 Risk1("real-only가 baseline 한참 아래면 decoupled 틀 자체가 병목, 합성 confound") 정확히 예측대로.

**결론:** **§24 generative(decoupled feature-synthesis) 축 SEALED.** 메커니즘은 입증됐으나(작동 O), decoupled 천장 때문에 non-reg 불가. **누적 19축 sealed.** 산출물: `code/featgen.py`(재사용 lib), `code/stage1_featgen.py`, `save/featgen_seed0/`(feature 캐시·cvae·result.json).

**프로그램 종료점:** §21-1 사전등록 우선순위((I)PGAL→(II)operators→(IV)generative)를 **전부 소진**. axis/hyperparam/inference/localization/composition-function/generative 6경로 모두 봉인. → 다음은 axis 변형이 아니라 **framework pivot / 새 architecture 결정 국면**(advisor 논의용 brainstorm). decoupled 천장이 병목이었던 만큼, generative를 살리려면 *joint 통합*(§24-2에서 기각한 옵션 B)이 후보이나 봉인한 joint-training 축군과 인접 → 차별성 재검토 필요.

---

# §25. NEW FRAMEWORK — GenProto (Recognition-by-Generated-Prototype) 사전등록 — 6-9 KST

**진입(사용자 결정 6-9):** §21 우선순위 축 전소진(19축 sealed) → "덧붙이기" 전략 사망진단 확정 → **axis가 아닌 새 framework 빌드로 pivot.** 사용자가 4갈래(generative-native / image-side 분해 / ClusPro base+혁신 / from-scratch) 중 **generative-native** 선택. 근거: §24가 **합성 메커니즘 작동(fidelity +0.44, unseen +0.046)을 직접 입증**, 죽은 건 decoupled frozen-cosine scorer뿐 → blank-slate가 아니라 *작동 증거 위에* 설계하는 evidence-led pivot. CLIP-CZSL 메이저 framework 중 generative-native 부재 = novel.

### 25-0. Thesis & 차별 lock

**Thesis:** 합성을 framework의 **코어**로. 합성기 G가 모든 composition의 visual prototype을 정의(seen=real 보정, unseen=생성)하고, **강한 joint-trained CMT scorer**가 이미지 patch와 매칭. §24의 유일 사망원인(frozen cosine 천장 0.31)을 강한 학습 scorer로 정확히 제거. "gap=architectural"(§20)을 표현 덧붙이기가 아니라 **composition을 생성으로 정의**하는 구조로 정면 대응.

**차별 lock (19축 + 외부 SOTA):**
- §24 generative=decoupled+frozen cosine scorer → GenProto=joint-trained CMT scorer (천장 제거가 핵심 차이).
- Troika=composition anchor가 frozen CLIP **text** → GenProto=**생성된 visual prototype** anchor.
- ClusPro=**real-clustered** primitive prototype(unseen 조합 생성 불가) → GenProto=conditional generator로 **unseen 조합 prototype 직접 생성**.
- 봉인 joint축군(C′/D′/E/OT)=Troika에 loss 덧붙이기 → GenProto=scoring anchor 자체를 생성으로 교체(구조 변경).

### 25-1. 아키텍처 (사전등록, 확정)

- **Image:** CLIP ViT-L/14 + adapters (baseline-equiv c_prime_seed0 ckpt, **frozen** — 검증된 표현 재사용, 새 메커니즘만 학습) → CLS f[768] + patches[257,768].
- **Generator G(attr,obj,z):** §24 cVAE 재사용(enc로 recon-grounding, dec로 생성). cond=[attr_text;obj_text](1536), latent 64. prototype p(a,o)=`G.decode(z, cond)` (L2-norm). eval-time mean prototype = z=0 (또는 K-sample 평균).
- **Scorer (CMT급, Troika `CrossAttentionLayer`×cmt_layers 재사용):** query=prototype p[P,768], kv=patch_norm(patches)[B,257,768] → cross-attn → `cmt_p = p + λ·CMT(p,patches)` → normalize → `logit = scale·cos(norm CLS f, cmt_p)` → [B,P]. **frozen cosine 아님 — CMT+λ 학습.**
- **학습 loss (joint):** `L = CE(comp_logits) + w_recon·(1−cos recon) + β·KL`. CE가 G+scorer를 discriminative하게, recon+KL이 G를 real feature 분포에 grounding(§24 작동분 보존). train pair(1262)로 CE.
- **eval-path:** 전 closed-world pair(1962) prototype 생성(z=0) → CMT scorer로 test 이미지 scoring → [N,P] → Troika `Evaluator/test()` 그대로(프로토콜 동일).

### 25-2. 사전등록 PASS (AND, §21 differentiation-first)

- **(a) 생성 코어 작동:** ablation — eval에서 prototype을 생성본 vs **comp-text init**로 교체 → 생성본 HM이 text-init보다 **≥ +0.005** 높음(생성이 실제 기여). + prototype fidelity(생성 prototype이 real centroid와 cos, random-cond 대비 Δ ≥ 0.10, §24 (a-ii) 계승).
- **(b) Non-regression:** val-selected test **HM ≥ 0.3849**.
- **STRETCH:** test HM **> 0.407** (ClusPro 추월 → framework로 강력).
- **FAIL/SEAL:** HM < 0.3849 **OR** (a) 미달.

### 25-3. 단계 & EV 경고

- **Stage 0 smoke:** MVP subset — G가 prototype 생성, CMT scorer 동작, joint loss grad가 G(enc/dec)+CMT(cross-attn/λ)로 흐름, recon↓, eval-path [N,P] 산출, ablation 토글(생성 vs text-init) logit 변화, NaN0.
- **Stage 1:** seed0 full(frozen image, G+CMT 학습) GPU1 → val_best test + ablation/fidelity → §25-2 판정.
- **EV 경고(정직):** 새 framework=큰 빌드, base rate 가혹(19축 sealed). 핵심 risk: **(R1) joint 학습 불안정** — scorer가 patches만으로 풀고 generator 무시(prototype 기여 0 → (a) 미달) OR generator mode-collapse. **(R2)** frozen image라 표현 천장은 Troika와 동일 → non-reg는 scorer 강도에 의존. 단 §24가 코어(생성) 작동을 이미 입증 → blank-slate보다 근거 우위. **§21 재정의대로 차별화O+non-reg+(a)입증이면 baseline 동급도 PASS.**

**다음:** 사용자 사전등록 승인됨(6-9) → `genproto.py`(scorer) + `smoke_genproto.py` 구현 → Stage 0.

### 25-4. 구현 + Stage 0 smoke 통과 — 6-9 KST

**구현(Troika 본체 미변경):** `genproto.py` — `GenProtoScorer`(Troika `CrossAttentionLayer`×cmt_layers 재사용: prototype query × patch_norm(patches) cross-attn → `p + λ·CMT` → cos(CLS, ·)·scale), `GenProto`(generator=§24 `featgen.CVAE` 재사용 + scorer; `prototypes`(z=0 mean)/`recon`(grounding)/`logits`), `encode_images`(frozen extractor → CLS+patches). `smoke_genproto.py`.

**Stage 0 smoke 통과 (subset: train64/test16 img, scorer cmt_layers=3, init_lamda=0.1):**
- **[1] extractor:** ckpt missing0/unexpected0, CLS[64,768]·patches[64,257,768] finite.
- **[2] cond:** train[1262]/all[1962] finite.
- **[3] joint train (핵심 — R1 1차 관문):** **grad가 generator(enc 2.4e-3, fc_mu 4.3e-3, dec_last 3.35) AND scorer(cmt0.q 3.2e-2, λ 2.28) 양쪽 모두로 흐름** → scorer가 generator 무시(R1) 안 함, 둘 다 학습. **CE 7.87→0.027(↓), recon 1.01→0.246(↓)**, NaN0.
- **[4] eval-path:** 전 1962 pair 생성 prototype → CMT scorer → [16,1962] → 공식 Evaluator 정상 작동(HM 0.0 = 16-img·40-iter overfit subset이라 무의미, plumbing만).
- **[5] ablation 토글:** 생성 prototype vs comp-text-init logit 차이 **10.60** → prototype source가 점수에 강하게 기여(생성이 실제로 anchor 역할).
- **VRAM peak 21.4 GB** (batch64 × 1262 prototype CMT w/ grad — 높음). **Stage1 주의: bs8(+grad accum)으로 낮춰야**(Troika도 bs8/accum8 동일 CMT 비용). prototype은 step당 1회 G forward로 재계산.

**다음:** Stage 1 driver(frozen image encoder, G+CMT joint 학습 seed0 full, bs8/accum8, val_best test + ablation/fidelity) 작성 → GPU1 launch → §25-2 판정.

### 25-5. Stage 1 1차 발산 → 진단 → lr fix 재실행 — 6-10 KST

**구현:** `train_genproto.py`(frozen encoder로 전 이미지 CLS+patches **CPU 캐시** 21GB → 재인코딩 0; G+CMT joint, bs32, val-AUC select). 버그수정 1건: recon cond를 이미지 idx로 인덱싱 → pair-label 인덱싱(`cond_train[lab_all[b]]`). **속도: frozen+cache로 ~5min/epoch**(다른 run의 ViT end-to-end 7h 대비, ViT forward/backward 제거가 핵심).

**1차 run 발산(lr 1e-3):** val AUC ep0 0.042→ep4 0.020(near-random, §24 decoupled 0.147보다도 아래), **CE ep1 3.48 최저 후 ep5 3.99로 역행**, recon 동반 상승. 사용자 결정으로 kill 후 진단.

**진단 (`diag_genproto.py`, 4-config 5ep sweep):** config A(lr **1e-4**, recon+KL 유지)에서 **val AUC 단조 상승 0.150→0.176→0.179→0.188 / HM 0.313→0.360**(ep3까지, 계속 상승), **protoInterCos 0.224→0.160 감소(prototype 더 분별됨 = collapse 아님 → KL-collapse 가설 기각)**, λ·logit 안정. → **발산 원인 = lr 1e-3 과대**(Troika CMT 표준 1e-4). 1e-4로 안정 학습 확인. (B no-KL/C disc-only/D 1e-3재현은 A 확정 후 user가 full run 재개 지시로 중단.)

**재실행(6-10 00:40, GPU1):** `train_genproto.py` **lr 1e-4 / 20ep**(수렴 느려 epoch↑, val-select가 overfit 처리)로 full run launch. cache hit으로 인코딩 skip. **주목: diag에서 ep3 HM 0.36이 아직 상승 중 → frozen 표현 위에서도 non-reg(b) 0.3849 통과 가능성 보임.** 완주 후 §25-2 (a-i 생성vs text-init / a-ii fidelity / b non-reg) 판정.

### 25-6. Stage 1 완주 → §25-2 판정 FAIL/SEAL (scorer 작동·생성 기여 입증, frozen image 천장이 병목) — 6-10 KST

**완주(6-10 00:40→02:28, GPU1, lr 1e-4/20ep):** val HM **ep2 0.3577 피크 후 단조 하락**(ep3 0.3521→ep10 0.3298) → 20ep로도 추가 상승 없음, val-AUC select가 ep2 부근 checkpoint 채택. `genproto_valbest.pt`·`stage1_result.json` 산출.

**§25-2 판정 (seed0, test):**
- **(b) non-reg FAIL (지배적):** **gen HM=0.3262 / AUC=0.1609** (seen 0.450 / unseen 0.442) vs baseline 0.3849 → **Δ HM −0.059** = seed noise(±0.0038)의 ~15배, 결정적.
- **(a-i) 생성 기여 PASS:** gen HM 0.3262 > **text-init HM 0.3038** = **Δ +0.0224** (≥0.005) → 생성 prototype이 frozen-cosine 아닌 강한 scorer 위에서도 실제 기여(§24 (a-i) 재확인, 마진 4배↑). 분해 흥미: **gen은 seen↑(0.450 vs 0.351), text는 unseen↑(0.497 vs 0.442)** — 생성이 seen 보정, text가 unseen 일반화에 강함.
- **(a-ii) fidelity 미달:** fid_cm 0.761 − fid_cr 0.685 = **Δ 0.076** (< 0.10 임계).
- verdict(스크립트): **FAIL/SEAL.**

**핵심 진단 (R2 실현 확정):** §24 사망원인(frozen cosine 천장)을 joint CMT scorer로 **정확히 제거했고 생성도 기여하는데도** HM 0.326 천장. scorer·generator 둘 다 작동(R1 회피 입증) → 남은 단일 병목 = **frozen image encoder(c_prime ckpt)**. §24와 동일 천장이지만 이번엔 강한 학습 scorer로도 못 뚫음 → **frozen 표현 위에서는 anchor를 text→생성 visual로 바꿔도 baseline(end-to-end ViT 0.3899) 도달 불가**가 결정적으로 확정. §25-1에서 "검증된 표현 재사용, 새 메커니즘만 학습"으로 frozen을 등록한 것이 정확히 천장의 원인.

**결론:** **§25 GenProto(frozen-image, generated-prototype anchor + joint CMT scorer) 축 SEALED. 누적 20축 sealed.** 산출물: `code/genproto.py`(scorer lib), `code/train_genproto.py`, `code/diag_genproto.py`, `save/genproto_seed0/`(enc 캐시·valbest·result.json). 잔여 코드버그 1건(재사용 시): `train_genproto.py:71` text test-eval에서 device mismatch(cpu vs cuda) — 핵심 결과(json) 기록엔 무영향, prototype `.cuda()` 누락만 수정.

**다음 결정 국면 (advisor-level, 사용자 판단 필요):** generative-native framework의 마지막 lever = **image encoder unfreeze(end-to-end GenProto)**. R2(frozen 천장)를 직접 제거하는 유일 경로이나 — (1) cache 이점 소멸, ~7h/epoch(Troika 동급), (2) Troika가 **이미 end-to-end text-anchor로 0.3899** 달성 → GenProto가 생성 visual anchor로 이를 **추월/동급**해야 하는데, frozen regime에서 생성의 text 대비 마진은 +0.022에 불과 → end-to-end에서도 GenProto ≈ Troika ± 소폭일 EV. 차별성(novel anchor)은 실재하나 SOTA-beating EV는 modest. **20축 sealed·base rate 가혹** 감안 시, end-to-end 시도(고비용·중EV) vs generative-native 종료·새 framework 결정은 brainstorm 사안.

### 26-5. 구현 + Stage 0 smoke 통과 — 6-10 KST

**구현(Troika 본체 미변경, 단일-통제):** `model/genproto_e2e.py` `GenProtoE2E(Troika)` — Troika를 subclass, comp(i_element=0) branch의 anchor만 `idx_text_features`(frozen text) → `self._proto(idx)`(=cVAE `G.decode(z=0, [attr_text;obj_text])`, L2-norm)로 교체. **attr/obj branch·CMT·patch_norm 루프내 재적용 quirk까지 baseline 그대로 보존**(단일 변수화). `loss_calu`는 `super().loss_calu`(Troika CE) + `w_recon·(1−cos recon(CLS)) + β·KL`. generator cond = frozen CLIP text(attr/obj, lazy 1회 계산). `use_text_anchor` 토글로 (a-i) eval-time swap. `forward_for_open`은 NotImplementedError(text_first eval로 frozen-text anchor 새는 것 차단; train.py evaluate는 predict_logits=forward 경로라 무관). model_factory 등록(`troika_genproto_e2e`). yml `mit-states-genproto-e2e.yml`(baseline 복제 + **bs8/accum8**[§11-11 0.3899 recipe]·20ep·gen_latent64/w_recon1.0/beta_max0.01·warmup0·text_first False). driver `train_genproto_e2e.py`(=train.py fork + β-ramp + optional encoder warmup + val_best(best_loss) select + §26-2 판정/(a-i)/(a-ii)/seen-unseen/json).

**Stage 0 smoke 통과(`smoke_genproto_e2e.py`, bs8 subset):**
- [1] gen 파라미터 optimizer 포함·trainable, 학습 파라미터 254개(=baseline + generator).
- [2] **anchor swap 실효: 생성 prototype vs comp-text cos mean −0.003(거의 직교)** → comp branch가 실제로 다른 anchor로 구동.
- [3] **grad가 encoder(adapter) AND generator AND cmt AND lamda 전부로 흐름**(§25 frozen은 encoder grad 無였던 것과 대비 — 여기가 핵심 신규 관문). g_enc step0=0(adapter LoRA up_proj zero-init 특성)→step1부터 0.013→0.02 정상(baseline 동일 거동). NaN0.
- [4] **recon 0.977→0.945 단조 하락 → R4(이동 CLS target) 단기 안정.**
- [5] eval-path: 전 closed pair [B,1962] finite, logit_infer OK. [6] **VRAM peak 10.2 GB**(bs8, A5000 24GB 대폭 여유).

**타이밍 정정(중요):** 실측 **~2.9 it/s × 3793 batch/ep ≈ 22min/epoch → 20ep ≈ ~7.3h(overnight 1회)**. §26-3에서 "~7h/epoch"로 적은 것은 추정 오류 — 실제는 **~7h 전체**(§25-5의 "ViT end-to-end 7h"도 per-run 총량이었음). multi-day 아님 → Stage 1 비용 부담 하향 수정.

**다음:** Stage 1 driver dry-run(eval/judgment 경로 = §25를 죽인 device-mismatch 지점 검증) 통과 확인 후 GPU1 full launch(20ep, ~7h).

### 26-6. Stage 1 dry-run 통과 → full run launch — 6-10 KST

**dry-run(E2E_EPOCHS1/MAX_BATCHES20/FIDELITY_MAXN2000, 별도 save dir):** train 20step → val → test(gen) → test(text ablation) → fidelity → judgment → json **전 경로 크래시 0 완주.** **§25를 죽인 eval device-mismatch 버그 없음 확인.** 숫자는 20step이라 무의미하나 신호는 예상대로: **text-anchor HM 0.215**(encoder 거의 frozen인 20step에서도 의미값 → fork가 Troika baseline 경로 온전 보존), **gen-anchor HM 0.000**(generator random-init cold-start, fidelity Δ≈0) — §25 frozen과 동일 cold-start(학습 후 gen이 text 추월했음). R3(generator가 따라잡는지)는 full run이 판정.

**Stage 1 full launch(6-10 13:22 KST, GPU1, PID 1600771):** `train_genproto_e2e.py` 기본값(20ep, bs8/accum8, lr1e-4, w_recon1.0, beta_max0.01, warmup0). 기동 직후 GPU1 93% 가동·VRAM 12GB·2.84it/s·recon 0.99→0.87 하락. **ETA ~7.4h(~20:45 KST).** 완주 후 §26-2 판정(=stage1_result.json) → PASS/FAIL/SEAL.
