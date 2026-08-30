# DARES Improvement Plan v2 — Mathematically Defensible Route

**Status:** Supersedes v1 (the "mix of methods" version, reverted in git).
**Goal:** Close the gap to DACS *without* importing foreign mechanisms
(EMA teacher, ClassMix, FDA). Every improvement must stay inside the Rényi-2 /
Gram-matrix framework so the contribution remains causally attributable to the
headline term. Optional input/self-training ingredients are deferred to an
explicit ablation tier, never part of the core method.

**Basis:** `Docs/KimiReport.txt` (CREDA→segmentation adaptation, theory),
`Docs/CREDA.md` (original paper), `Docs/SegCREDA.py` (reference hardened
module), Run evidence from `reports/LIME_stress/medium/*`.

---

## 1. Diagnosis (unchanged from v1 — still valid)

- Best checkpoint (0.4991) is at epoch 2, before adaptation was active.
- `loss_align` exploited via target over-dispersion: Δ → −2.77 bits. The root
  cause is the **−½H2_t reward inside the Rényi mutual-information surrogate**
  `Ĩ2 = ½H2_s + ½H2_t − H2(mix)` (KimiReport Part II §0/§1.1). Because the
  source is stop-graduated, maximizing Ĩ2 over the target = minimizing
  `H2(mix) − ½H2_t`, and the +½H2_t term pays for dispersion.
- Safeguards constrained the feasible set but did not bound the exploit:
  anti-collapse is only a *lower* bound on target entropy.
- Trust region responded by suppressing λ_eff to ~1% of λ_max — the headline
  term became a near-no-op, unsatisfiable in a methods paper.
- Failure concentrated in forest recall (0.47 vs DACS 0.96): a density-of-
  supervision problem.

## 2. The defensible repair (replaces the old P0)

### 2.1 Root fix — conditional-entropy alignment, not a band patch

In `Docs/SegCREDA.py` / `src/dares/losses/dares_loss.py`, replace the per-class
MI surrogate with the **conditional-entropy form**:

    L_align,c = H2(K_c^mix) − H2(K_c^s)        (source side stop-graduated)

Removing the −½H2_t term removes the dispersion reward *by construction*; no
two-sided band patch needed. This stays entirely in the matrix-based Rényi-2
family and fits the paper's "conditional adaptation" branding. Justification
is already in `Docs/KimiReport.txt` Part II §1.1: the block-matrix MI
construction fails subadditivity in the dispersed (segmentation) regime, hence
the conditional form is the *correct* estimator there. Keep the entropy floors
and repulsion hinge — they carve the feasible set independently of which
intra-class divergence is plugged in.

**Implementation:** add `align_form: "ce" | "mi" = "ce"` (CE = conditional
entropy; MI retained for ablation). When `ce`, compute
`d_c = _h2_joint(s2_st-part) − _h2_bits(s2_s, tr_s)` where the joint is the
source-only-plus-cross block; wire `align_form` through `TrainConfig` and the
dares engine. Do not change the source stop-grad anchoring.

### 2.2 Dense target supervision from within the framework — Rényi-EM

CREDA already computes per-pixel Rényi-2 prediction entropy for the confidence
weight `w = 1 − Ĥ2/log C`. Turn that diagnostic into an objective:

    L_em = mean_t( (1 − w_agg) · Ĥ2(P_t) )     over target pixels

i.e. entropy minimization on target predictions, weighted by the complement of
the (optionally spatially pooled) confidence weight so genuinely ambiguous
pixels are not forced. Same mathematical family as CDAN+E's entropy
conditioning (the CREDA paper itself compares against it), zero borrowed
machinery. This supplies the dense per-pixel gradient that forest recall needs.

**Implementation:** new term inside `DARESLoss.forward`, weight `lambda_em`
(expose in config; start at 0.01–0.1, schedule-free), applied to the soft
predictions `P_t` before argmax detach. Ablation flag `use_renyi_em`.

### 2.3 Let λ actually matter — ramp-only scheduling

With the conditional form bounded by the floors, the trust region is no longer
fighting a runaway objective. Default `trust_region: false` (config path
already exists) so λ follows the sigmoid ramp to O(1); keep the trust region
as an ablation safety valve. Monitor gradient norms via the (already returned)
diagnostics instead of clamping them.

### 2.4 Principled integrity fixes (keep unconditionally)

- `_quota_indices`: sample **without replacement** (no κ(x,x)=1 duplicate
  blocks inflating purity). Engine handles variable-cardinality class sets.
- Per-epoch logging of existing diagnostics: `h2_source_mean`,
  `h2_target_mean`, `delta_align_mean`, `delta_repulsion_mean`, `lambda_eff`,
  `n_valid_classes` — observability is non-negotiable.
- `ce.py` NaN guard (zero loss when all target pixels ignored) only if needed
  by a new term.

## 3. Stratified evaluation (makes the claim defensible)

DACS belongs to a different signal class (data mixing + dense pseudo-labels).
Do **not** fold it in. Evaluate in tiers:

- **Tier 1 (headline claim):** DARES [conditional form + Rényi-EM] vs
  source_only, ADVENT, FDA — won by the alignment/EM term alone.
- **Tier 2 (context):** DARES vs DACS, reported with an **ablation table**
  isolating: align-only / EM-only / align+EM. The table is the defense.
- If pure alignment saturates below DACS, report it as a finding: second-order
  feature alignment saturates; prediction-space (Rényi) supervision is
  necessary. That is publishable, not a failure.
- **Resolve the warmup confound first:** old DARES vs source_only differed at
  epoch 2 (0.4991 vs 0.3488) under identical λ=0 objectives. Audit data order /
  augmentation RNG / BN statistics before any comparison run. Until resolved,
  all method comparisons on this benchmark are suspect.

## 4. Deferred (explicitly NOT in core DARES)

EMA teacher, ClassMix, FDA input swap, multi-scale alignment, pseudo-label
thresholds, schedule re-tunes. If ever wanted, they go in a clearly-labeled
"extended" ablation (Tier 3), never in the method definition.

## 5. Implementation order for the agent

1. Audit & resolve the warmup confound (§3) — blocks all comparisons.
2. `align_form = "ce"` conditional-entropy alignment (§2.1) + keep floors/
   repulsion; run LIME_stress/medium vs Tier 1.
3. Rényi-EM term (§2.2), `use_renyi_em` flag; ablate align/EM/both.
4. `trust_region: false` default (§2.3); confirm λ ramps to O(1) and
   `delta_align_mean` stays bounded (no over-dispersion exploit).
5. No-replacement quotas + diagnostics logging (§2.4).
6. Full Tier-1/Tier-2 comparison on LIME_stress/medium with resolved seeds;
   success = beat ADVENT/FDA without foreign mechanisms, and a clean ablation
   table for the DACS comparison.

## 6. Constraints for the implementing agent

- Loss lives in `src/dares/losses/dares_loss.py` (`DARESLoss`); trainer wiring
  in `src/dares/engines/dares.py` (`update_lambda` called after forward, before
  backward, `ref_params` = deepest shared encoder block).
- Keep the CREDA core fp32 with autocast disabled (AMP-safe; repo uses
  `use_amp: true`).
- Do not regress the three safeguards (entropy floors, repulsion hinge, trust
  region) — they constrain the feasible set regardless of `align_form`.
- All changes must be ablation-flagged so each term's contribution is
  attributable; no foreign mechanisms in the default path.
