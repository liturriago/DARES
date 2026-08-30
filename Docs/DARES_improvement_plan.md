# DARES Improvement Plan — Diagnosis and Action Items

**Status:** Hand-off document for an implementing agent.
**Basis:** Analysis of `Docs/KimiReport.txt`, `Docs/CREDA.md`, `Docs/SegCREDA.py`,
`reports/LIME_stress/medium/*` and `configs/LIME_stress/medium/*` only.
**Task setting:** Binary UDA segmentation (forest / non_forest), ConvNeXt-Tiny +
ResUNet, 4-channel input, source → Target_Medium, 25 epochs, batch 8, patch 224.

---

## 1. Current results (LIME_stress / medium, target TEST mIoU)

| Method | Target mIoU | forest IoU | forest recall | Source mIoU |
|---|---|---|---|---|
| DACS | **0.7861** | 0.7963 | 0.9599 | 0.8259 |
| FDA | 0.5416 | 0.4352 | 0.4386 | 0.7804 |
| ADVENT | 0.5347 | 0.4737 | 0.5483 | 0.8675 |
| **DARES** | **0.4949** | 0.4104 | 0.4655 | **0.8323** |
| source_only | 0.3647 | 0.1740 | 0.1761 | **0.8885** |

DARES gains only +0.13 over source_only, trails DACS by −0.29, and is the only
adapted method that **loses source-domain performance** (0.8323 vs 0.8885).

## 2. Evidence from the DARES training log (`reports/LIME_stress/medium/dares.txt`)

1. **Best checkpoint predates adaptation.** Best target val mIoU = 0.4991 at
   epoch 2, where `loss_total == loss_seg` exactly (λ_eff = 0 in warmup, frozen
   backbone). From epoch 3 (backbone unfrozen, λ > 0) target val drops to ~0.44
   and fluctuates 0.41–0.48 for 23 epochs without recovering.
2. **Estimator exploited in the over-dispersion direction.** `loss_align` drifts
   monotonically −0.25 → **−2.77 bits** (a true match is Δ ≈ 0). With the source
   stop-graduated, Δ = H2(mix) − ½H2_s − ½H2_t can only be pushed negative by
   **inflating target dispersion** — the −½H2_t term pays ~0.5 bit per bit of
   target entropy growth. The anti-collapse floor is only a *lower* bound, so it
   stays silent (`loss_anti_collapse` → ~0.001) while Δ is gamed.
3. **Safeguards inactive from epoch 4.** `loss_repulsion` → ~0.0002,
   `loss_anti_collapse` → ~0.001. No collapse occurred — the floors work — but
   they do not bound the opposite failure mode.
4. **Trust region strangles the method.** Backing λ_eff out of
   `loss_total − loss_seg = λ_eff·loss_aux`: λ_eff ≈ 0.0026 (epoch 3) →
   ≈ 0.011 (epoch 25), i.e. ~1% of λ_max = 1.0. With grad_ratio ρ = 0.8 this
   implies aux gradients are ~50–80× seg gradients — gradient domination is
   real, and the response (suppress λ) leaves the method contributing almost no
   useful signal while applying a flawed direction for 23 epochs.
5. **Source corruption despite anchoring.** Source val mIoU 0.82 (epoch 2) →
   ~0.76–0.78 late; source test 0.8323 vs 0.8885 for source_only. sg(Φ_s)
   freezes the anchor only *within* a step: target-side gradients still update
   the shared encoder, moving the source manifold at the next step.
6. **Failure concentrated in forest recall.** Target forest: precision 0.776,
   recall 0.466 — the classifier stays conservative under shift. DACS solves
   exactly this (recall 0.96) via dense per-pixel pseudo-supervision. DARES's
   only target signal is a second-order statistic over 256 sampled bottleneck
   vectors per class.

## 3. Action items

### P0 — Close the estimator's free-lunch direction
File: `src/losses/dares_loss.py` (class `DARESLoss`), see `_delta_bits`, `forward`.

- **Make the target entropy constraint two-sided.** Replace
  `floor_t = ReLU(H_s.detach() − entropy_gap − H_t)` with a symmetric band:
  penalize both `H_s_sg − gap − H_t` and `H_t − H_s_sg − gap`. This directly
  caps the over-dispersion reward observed in the log.
- **Alternative (or additionally):** remove the marginal target-entropy bonus —
  use a conditional form `H2(mix) − H2_s` (no −½H2_t term to inflate), or
  replace the per-class term with MMD² on the same Gram blocks. (The Dirac
  objection in KimiReport Part II §1.1 no longer applies once floors +
  anchoring exist; the run shows the *opposite* basin is the live threat.)
- **Observability (non-negotiable):** log per epoch the diagnostics the module
  already returns: `h2_source_mean`, `h2_target_mean`, `delta_align_mean`,
  `delta_repulsion_mean`, `lambda_eff`, `n_valid_classes`. The run's central
  pathology had to be reverse-engineered from loss arithmetic.

### P0 — Add dense target-domain supervision
The gap to DACS is supervision density, not alignment quality.

- Add confidence-thresholded pseudo-label self-training on the target (pixel CE
  and/or soft Dice), with an EMA teacher producing the pseudo-labels (see P1).
- Add ClassMix-style source/target mixing, or weak–strong augmentation
  consistency on target images.
- Keep the Rényi term as a complement, not the sole target signal.
- Reference: DACS config uses `dacs_threshold: 0.968`, `dacs_mix_ratio: 0.5`
  plus color jitter + blur — a proven recipe on this exact dataset/split.

### P1 — Attack the photometric gap at the input level
- FDA (input-space Fourier amplitude swap) alone beats DARES (0.5416 vs
  0.4949). Add FDA-style amplitude swap or target-matched color jitter as
  preprocessing, complementary to the deep-feature alignment.
- Implement KimiReport Part I §4.4 multi-scale alignment: apply the alignment
  term at an additional shallow decoder level (λ_ℓ weights), since the
  spectral shift lives in early features.

### P1 — Fix the moving anchor and the unfreeze shock
- Use an **EMA teacher** as the alignment anchor and pseudo-label source:
  stops pseudo-label drift and makes the anchor stable across steps (current
  hard argmax + soft weights cannot).
- Soften the epoch-3 unfreeze shock (all methods dip at epoch 3; DARES never
  recovers): partial encoder unfreeze, or a lower LR for the encoder than the
  decoder, or per-domain normalization statistics.

### P2 — Pseudo-label quality control (binary task)
- Add a confidence **threshold** for membership in `T_c` (currently pure
  argmax: confidently-wrong pixels get w ≈ 1 and are aligned onto the wrong
  source class — confirmation bias). Suggested starting point: 0.968 (DACS).
- In `_quota_indices`, sample **without replacement** when `n_c < M` (current
  `torch.randint` duplicates points; κ(x,x)=1 blocks inflate purity and bias
  Δ). Either lower M for small classes or skip replacement duplicates.

### P2 — Cheap ablations
- Align the **foreground class only** (ω_bg = 0, KimiReport Part I §3.3):
  non_forest is what differs most across domains; aligning it may be
  net-harmful.
- Re-tune the schedule given measured λ_eff ≈ 0.01: config has
  `grad_ratio: 0.8`, `ramp_steps: 4000`; the reference defaults are ρ = 1.0,
  ramp 9000. Consider raising ρ and λ_max only *after* the two-sided floor is
  in place (otherwise more λ just feeds the exploited direction).
- **Investigate a comparison confound:** DARES and source_only run nominally
  identical objectives during the 2 warmup epochs (λ = 0, frozen backbone, same
  seed 42), yet epoch-2 target val is 0.4991 (DARES) vs a run-best of 0.3488
  (source_only). Check whether data order / augmentation RNG differs by method,
  or whether an extra forward pass changes BN statistics. If systematic, all
  method comparisons on this benchmark are confounded and must be re-run.

## 4. Suggested implementation order

1. Logging of existing diagnostics (no behavior change) + rerun baseline to
   confirm Δ → negative over-dispersion and record H2 traces.
2. Two-sided target entropy band (P0) — single-run ablation.
3. EMA teacher + pseudo-label self-training + ClassMix (P0/P1) — expect the
   bulk of the forest-recall recovery (target: recall → ≥ 0.8).
4. FDA-style input adaptation (P1) — cheap, orthogonal.
5. Confidence threshold + quota sampling fix (P2), foreground-only alignment
   ablation (P2), schedule re-tune (P2).
6. Re-run the full 5-method comparison on `LIME_stress/medium` with identical
   seeds/augmentation after resolving the confound in P2.

## 5. Constraints and notes for the implementing agent

- `src/losses/dares_loss.py` is the hardened loss module (class `DARESLoss`); the
  trainer/criterion wiring lives elsewhere in the repo — locate it first
  (search for `DARESLoss`, `update_lambda`, `lambda_eff`).
- The training loop must call `crit.update_lambda(parts["loss_seg"],
  parts["loss_aux"], ref_params)` after forward and before backward, with
  `ref_params` = deepest shared encoder block (see module docstring §6 usage).
- Keep the CREDA core in fp32 with autocast disabled (AMP-safe); the repo runs
  `use_amp: true`.
- Config keys for DARES live in `configs/LIME_stress/medium/dares.yaml`; add
  new hyperparameters there (teacher EMA decay, pseudo-label threshold,
  two-sided gap, FDA on/off, etc.).
- Do not regress the three verified safeguards: entropy floors (anti-collapse),
  margin-hinged repulsion, GradNorm-lite trust region — they work (no collapse
  in the log); the problem is what they don't cover.
- Success criterion: target test mIoU ≥ 0.70 on LIME_stress/medium without
  dropping source test mIoU below source_only's 0.8885; forest recall is the
  primary per-class metric to watch.
