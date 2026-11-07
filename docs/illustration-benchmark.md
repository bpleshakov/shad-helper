# Illustration quality benchmark

The benchmark binds human labels to the exact saved notes that were reviewed.
It performs no model calls and never modifies lecture outputs. It does not infer
that a changed or removed bad image has been fixed.

Freeze the labels before regenerating notes:

```sh
.venv/bin/python scripts/illustration_benchmark.py freeze \
  --labels illustrations.md --roots out_algo out_vision_improved \
  --out /tmp/illustrations-baseline.json
```

The JSON records each lecture, PDF hash, rendered figure number, block ID,
illustration and coverage source IDs, asset hash, caption/context hash, human
label, source-crop verification route, and open review IDs. All numbered figures
must have labels. Missing-illustration lecture annotations are separate cases.
The tool checks assets and PDF against `source-map.json`, and figure IDs against
`main.aux` and `notes.json`. The initial label numbers must describe this current
render; no automated check can establish which older PDF a person reviewed.

Each lecture receives a deterministic development/holdout assignment from its
name hash (approximately 80/20). Keep all figures from the same lecture together.
Tune on development lectures, then inspect holdout results without tuning to
them. Both the assignment and source hashes are retained in the baseline.
The current labels already informed this implementation, so this split is not an
unseen evaluation of its design. Newly labeled lectures are needed to measure
generalization independently; use the split to separate subsequent tuning work.

After generating outputs in the same or separate roots:

```sh
.venv/bin/python scripts/illustration_benchmark.py compare \
  --baseline /tmp/illustrations-baseline.json \
  --roots out_algo out_vision_improved \
  --out /tmp/illustrations-comparison.json
```

Matching uses source identity and coverage, then identical asset bytes when source
IDs change. It does not attach old labels to the same figure number after
renumbering. A label is retained only when both the asset and its caption/context
remain unchanged. Changed images, changed claims, ambiguous matches and newly
added figures require human review. Missing good figures are counted explicitly;
an exclusion reason does not make them successful. An unavailable or inconsistent
lecture is reported rather than scored as unchanged.

The missing-illustration case remains `still_missing` when no figures appear.
Once figures appear, it becomes `needs_recall_review`: a nonzero figure count is
not evidence that all meaningful board drawings or stages were recovered. Video
jobs whose current evidence lacks the visual stage are reported separately.

The frozen summary separates bad labels on crops that passed verification from
original source crops with open review issues. These expose different failure
paths; the route alone is not a quality judgment. For diagrams, the asset hash
identifies the accepted TeX source. A changed PDF is flagged separately, because
unchanged assets do not establish final layout, scale or readability.

Both commands create their report exclusively and refuse to overwrite an existing
report. Reports must be outside the specified output roots. Keep the original
baseline and review changed figures independently before assigning new labels.
