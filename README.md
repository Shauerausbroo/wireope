# Off-Policy Evaluation of CT Angiography-Conditioned Robotic Guidewire Strategies for Estimating Vessel Injury Risk during Peripheral Arterial Recanalization and Lesion Crossing

## Overview

This repository is the code release for the manuscript of the same title. Guidewire crossing is
treated as a finite-horizon decision problem whose state combines the luminal envelope obtained
from pre-procedural CT angiography with anatomical identifiers and a partially observed device
state. The delivered action is not logged by the robotic systems, so it is reconstructed as the
excursion of the wire body beyond the envelope boundary, and the rare injury endpoint is
decomposed into a nested ladder of graded excursions. A strategy is certified by maximising a
lower confidence limit on the release's utility over the strategies that clear a per-stratum,
per-rung overlap floor and stay inside a pre-specified risk margin, with abstention when none
does.

The package `wireope` implements Equations (1)-(7) of the manuscript, the eight table harnesses
it reports, the component ablation and ladder-depth sensitivity of its Table 2 panel A, and the
estimator bake-off of its Table 2 panel B.

The primary evidence layer is a private anonymized multicentre cohort held under data-sharing
agreements, and the manuscript states that its record-level source data are not shared. Every
cohort-level value is therefore absent from this release. The executable science runs on a cohort
the release assembles for itself from `configs/cohort/release.yaml` and a seed, whose generating
probabilities are known in closed form; its records are constructed rather than collected. No
value in this repository is a copy of a value in the manuscript, and the manuscript's printed
values are transcribed once, with their table and row, so the release can report the paper's own
arithmetic beside its own.

## Project context

Resolved from the manuscript before any code was written.

| Field | Value | Source | Confidence |
| --- | --- | --- | --- |
| project name | `wireope` (package); the directory carries the paper title | title page | HIGH |
| domain | clinical off-policy evaluation for endovascular procedural safety | Abstract, Sec. 3 | HIGH |
| framework | PyTorch 2.x with `torch.nn`, NumPy, SciPy, scikit-learn | Sec. 3.2 encoder, Sec. 3.4 state-space hazard head | MEDIUM |
| venue | npj Digital Medicine | filename prefix, reference style, structured abstract | MEDIUM |
| primary datasets | 1 private clinical cohort plus 5 open auxiliary corpora and 5 public logged corpora | Sec. 3.2 footnotes 1-5, Sec. 3.6, Table 4 panel B | HIGH |
| compute target | a single processor graphics node; accelerator, memory and wall-clock are not reported | Sec. 3.7, final paragraph | HIGH |
| hyperparameter reference | not reported; the manuscript states only that the seeds are hard-wired | Sec. 3.7, final paragraph | HIGH |
| supplementary path | none; the PDF carries no appendix beyond a reference to Appendix A | - | HIGH |

## Installation

pip:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

conda:

```bash
conda env create -f environment.yml
conda activate wireope
```

Docker:

```bash
docker build -t wireope .
docker run --rm wireope --repo-root /opt/wireope
```

The container base is `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`, which ships the Python
3.11 interpreter the project targets. The release runs on CPU; a GPU is only needed for the
volume encoder at the scale the manuscript used.

## Data

The primary cohort is private. Its records are held under site data-sharing agreements and are
not shared at record level, so this repository ships no path, no mirror and no derived table of
it. Every cohort-level value the release would otherwise report is absent, and the mechanism is
exercised on that cohort instead.

| Corpus | Access | Licence | Role |
| --- | --- | --- | --- |
| private anonymized multicentre cohort | data-sharing agreement, not distributed | - | primary evidence layer (Sec. 3.6) |
| CT-FM 3D CT foundation model | <https://github.com/project-lighter/CT-FM> | MIT | frozen volumetric encoder (Sec. 3.2) |
| Medical Segmentation Decathlon Task 09 | <https://registry.opendata.aws/msd/> | CC BY-SA 4.0 | open auxiliary corpus |
| MedMNIST v2 VesselMNIST3D | <https://medmnist.com/> | CC BY 4.0 | vessel-geometry probe |
| ARCADE | <https://zenodo.org/records/10390295> | CC0 | lesion-labelling auxiliary set |
| MONAI Model Zoo Swin UNETR | <https://github.com/Project-MONAI/model-zoo> | Apache-2.0 | segmentation baseline |
| Open Bandit Dataset | <https://research.zozo.com/data.html> | CC BY 4.0 (ZOZO RESEARCH) | logged corpus with known propensities |
| NeoRL / NeoRL-2 | <https://github.com/polixir/NeoRL> | Apache-2.0 | confounded sequential logs |
| COBS | <https://github.com/google-deepmind/offpolicy_selection_eslb> | Apache-2.0 | auxiliary reference |
| Deep-OPE | <https://github.com/google-research/deep_ope> | Apache-2.0 | auxiliary reference |
| D4RL | <https://github.com/Farama-Foundation/D4RL> | Apache-2.0 | idealised tier, not clinical |

`dataset_urls.txt` carries each link with its licence, its role and its access conditions, and
lists only links that answered when probed from the verification host. `THIRD_PARTY_NOTICES.txt`
carries the same provenance in prose. The five logged corpora are transcription targets: the
release does not download them, so their rows carry no value.

Assemble the release cohort and inspect its partitions:

```bash
python -m wireope.cli.reconstruct --repo-root . --experiment main
```

The assembly writes to `artefacts/cohort/cohort_summary.json`. It is reassembled from
`configs/cohort/release.yaml` and a seed, needs no download, and reproduces byte for byte.

## Training

One command per reported experiment. Each command writes its record under `artefacts/`.

```bash
# full framework: the main comparison of Table 1
python -m wireope.cli.fit --repo-root . --experiment main

# the six single and paired removals plus the depth-sensitivity row of Table 2 panel A
python -m wireope.cli.fit --repo-root . --experiment ablation_without_envelope_excursion_reconstruction
python -m wireope.cli.fit --repo-root . --experiment ablation_without_nested_excursion_ladder
python -m wireope.cli.fit --repo-root . --experiment ablation_without_support_constraint
python -m wireope.cli.fit --repo-root . --experiment ablation_reconstruction_and_ladder_removed
python -m wireope.cli.fit --repo-root . --experiment ablation_ladder_and_support_removed
python -m wireope.cli.fit --repo-root . --experiment ablation_reconstruction_and_support_removed
python -m wireope.cli.fit --repo-root . --experiment ablation_ladder_depth_six

# the supplementary studies: Table 3, Table 4, Table 5, Table 6, Sec. 4.6
for experiment in supplementary_weight_control_sweep supplementary_event_budget \
                  supplementary_public_corpora supplementary_sitewise \
                  supplementary_scaling supplementary_clinician_arms; do
  python -m wireope.cli.fit --repo-root . --experiment "${experiment}"
done
```

The release reports no training hyperparameters for the cohort-level fits because the manuscript
reports none; the values in `configs/fit/` and `configs/estimators/hazard.yaml` are engineering
defaults, and every one of them is listed in the `deviations` array of `claim_to_code.json`.
The defaults are: AdamW at learning rate 3e-3 with weight decay 1e-4, batch 256, 40 epochs, a
linear warmup over 5% of the steps followed by cosine decay, gradient clipping at norm 1.0, an
EMA copy at decay 0.999, and the seed set `{0, 1, 2, 3, 4}`. The effective batch is
256 x 1 x 1 = 256. A two-epoch smoke run on `configs/experiment/_smoke.yaml` is the end-to-end
test and is not a reported configuration.

## Evaluation

One command per reported table. The numbers below are the release's own, computed on the cohort
the release assembles; the manuscript's printed values are transcribed in
`wireope/report/transcribe.py` with their table and row, and are never mixed into the release's
output.

```bash
python -m wireope.cli.estimate --repo-root . --experiment main       # per-strategy estimates
python -m wireope.cli.certify  --repo-root . --experiment main       # the certified decision
python -m wireope.cli.studies  --repo-root . --experiment main       # every table, as plain text
python -m wireope.cli.verify   --repo-root .                         # both verification layers
```

`wireope.cli.studies` writes `artefacts/report.txt`, which carries:

| Block | Counterpart | What it reports |
| --- | --- | --- |
| main comparison | Table 1 | AUROC with a DeLong interval, net benefit at two thresholds, selection accuracy, violation upper bound, abstention |
| component ablation | Table 2 panel A | change in selection accuracy and violation rate per variant, interaction ratio, effective sample size, overlap |
| estimator bake-off | Table 2 panel B | relative error, effective sample size, overlap coefficient, selection accuracy, near-top frequency, diagnostic |
| event budget | Table 4 panel A | relative error, effective sample size, lower-bound width as the rarest-rung count moves at fixed records |
| scaling | Table 6 | one scaling axis per row with both the record and rarest-rung counts |
| sitewise | Table 5 | per site, per stratum and per subgroup replication with coverage |
| weight control | Table 3 | every stratum crossed with the clipping-by-pessimism grid |
| reader arms | Sec. 4.6 | clinician alone, model alone, and the clinician with the model |

Expected magnitudes on the release cohort, so a reader can tell a working run from a broken one:
the relative error of the nested estimator on the held-out share is of order 0.1; the overlap
coefficient of the most aggressive strategy is of order 0.1 and rises as the ladder is
shallowed; the certified rule selects a gentler strategy than the unconstrained argmax on
every configuration. These are the release's own numbers and no tolerance is claimed against
the manuscript's tables, which were produced on a cohort that is not distributed.

## Compute budget

The manuscript reports that encoder training uses a single processor graphics node, that every
estimator and hazard head is fitted with hard-wired seeds, and that no online inference or
real-time latency requirement applies. It does not report the accelerator model, its memory,
the wall-clock or the storage. Those four are recorded as `COMPUTE_NOT_REPORTED` in
`configs/fit/runtime.yaml` rather than guessed.

What this release actually uses:

| Stage | Hardware | Memory | Wall-clock |
| --- | --- | --- | --- |
| cohort assembly, main configuration (12,000 attempts) | CPU, 1 process | under 1 GB | under a minute |
| per-strategy estimation and certification, `estimate` and `certify` | CPU, 1 process | under 1 GB | under a minute |
| every table harness, `studies`, on the main configuration | CPU, 1 process | under 2 GB | about twelve minutes, dominated by the clustered resamples behind each arm |
| hazard head, 40 epochs on the full cohort | CPU, 1 process | under 2 GB | minutes per seed |
| verification suite, `verify` | CPU, 1 process | under 2 GB | a few minutes, against the smoke configuration |

Disk: under 100 MB for the repository, the cohort is assembled in memory, and checkpoints
are pruned to the last two by default. Distributed execution is wired through `torchrun` and is
not needed at this scale; `scripts/launch_fit.sh` selects it when `WORLD_SIZE` exceeds one.

## Repository layout

```
.
├── configs/
│   ├── cohort/            the private cohort description, the release cohort, the partitions, the public corpora
│   ├── geometry/          the envelope, its boundary and the anatomical descriptors
│   ├── traces/            device events, tip reconstruction, registration and the propagating band
│   ├── laddering/         rung thresholds, ladder weights
│   ├── behavior/          the two-part behaviour reconstruction
│   ├── estimators/        the candidate library, the hazard head, the rung outcome model
│   ├── bounds/            the deviation bound and the effective-sample-size definitions
│   ├── certification/     the overlap floor, the risk margin and the sweep grids
│   ├── fit/               optimiser, schedule, runtime shape
│   └── experiment/        main, the seven ablations, six supplementary studies and _smoke
├── src/wireope/
│   ├── geometry/          LumenEnvelope, the analytic vessel, the descriptors, the encoder
│   ├── traces/            event integration, similarity registration, the excursion observable
│   ├── laddering/         the nested severity ladder, its weights and its resolution criterion
│   ├── behavior/          branch and action propensity models, fidelity and its cost
│   ├── estimators/        IS, direct, kernel, doubly-robust, switching, nested DR, the comparator
│   ├── bounds/            Hoeffding, empirical Bernstein, effective sample size, intervals
│   ├── certification/     the support floor, the selection rule, the weight-control sweep
│   ├── cohort/            the data dictionary, the release cohort, the partitions, the adjudication
│   ├── metrics/           discrimination, calibration, decision-analytic, safety, bootstrap
│   ├── studies/           the analysis pipeline, the arms, the ablation and the table harnesses
│   ├── fit/               the encoder adapter, the hazard head, the trainer, checkpointing, DDP
│   ├── report/            the manuscript transcription and the plain-text renderer
│   ├── verification/      the claim map, the execution layer, the manuscript layer, the writers
│   ├── cli/               reconstruct, fit, estimate, certify, studies, verify
│   └── utils/             seeding, logging, atomic writes, digests, numerical primitives
├── scripts/               shell entry points for the four stages
├── tests/                 unit, shape, regression, overfit and end-to-end smoke tests
├── claim_to_code.json     every paper item, the code that carries it, and the probe result
├── verification_report.json
├── verification_summary.txt
├── integrity_manifest.json
└── dataset_urls.txt
```

## Engineering decisions the manuscript leaves open

The manuscript fixes several quantities without printing them, and prints two constraints that
disagree with each other. The release resolves each one in the open; the full list, with the
paper location and the reason, is the `deviations` array of `claim_to_code.json`. The four that
change the most:

1. The risk constraint follows Eq. (2), a lower confidence limit compared against a margin,
   rather than Eq. (5), an upper limit against a threshold. The two disagree in direction, and
   the release reports that as a finding about the manuscript.
2. The utility the rule maximises is the graded crossing progress on the ladder, the complement
   of the graded excursion severity of Eq. (3), because Eq. (2) maximises net benefit while
   Eq. (3) writes a severity functional whose sign runs the other way.
3. The rung indicators nest downward with the top rung equal to the adjudicated endpoint, as
   Sec. 3.4 requires, rather than upward as Sec. 3.1 prints.
4. The envelope segmentation tolerance, the ladder weights, the default depth, the overlap floor,
   the risk margin, the level delta and the bounded rung range are engineering defaults, because
   the manuscript states the tip tolerance and nothing else in that chain.

## Verification

Every result in this release is marked PASS, FAIL, NOT_RUN or BLOCKED from what actually ran.
The suite has two layers. The execution layer reads data, runs forward passes, computes losses,
takes gradients, updates parameters, writes and reloads checkpoints, overfits a single batch and
runs a minimal training loop, and checks the estimators and bounds against closed forms, brute
force enumeration and a second library. The manuscript layer recomputes the paper's own
tabulated arithmetic and reports every disagreement under its own family, so a discrepancy
inside the manuscript is never confused with a defect in the release.

```bash
scripts/run_verification.sh                  # deterministic: the live link probe stays undecided
scripts/run_verification.sh --probe-links    # also reach the network to re-probe every dataset link
```

The driver writes `claim_to_code.json`, `verification_report.json`,
`verification_summary.txt` and `integrity_manifest.json` in that order, and the manifest digests
the finished tree while excluding only itself. Run it twice with no edits in between and the
artefacts are byte-identical.
