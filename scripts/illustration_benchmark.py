"""Freeze human illustration labels and compare later notes without inventing new labels."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from shad_helper.jobs import rendered_pdf


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_labels(path):
    lectures = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("## "):
            name = line[3:].strip()
            if not name or name in lectures or Path(name).name != name:
                raise ValueError(f"Duplicate or invalid lecture: {name}")
            current = lectures[name] = {"labels": {}, "missing_figures": ""}
        elif current is None:
            raise ValueError("Labels must start with a lecture heading")
        elif match := re.fullmatch(r"(\d+) (good|bad)(?:,\s*(.+))?", line):
            number = int(match[1])
            if number < 1 or number in current["labels"]:
                raise ValueError(f"Duplicate or invalid figure number: {line}")
            current["labels"][number] = {"label": match[2], "reason": match[3] or ""}
        elif line.startswith("Иллюстраций нет") or line.lower().startswith("no illustrations"):
            current["missing_figures"] = line
        else:
            raise ValueError(f"Unrecognized label: {line}")
    if not lectures:
        raise ValueError("No labeled lectures")
    return lectures


def locate(roots, name):
    matches = [root / name for root in roots if (root / name).is_dir()]
    if len(matches) != 1:
        raise ValueError(f"Expected one output for {name}, found {len(matches)}")
    return matches[0]


def snapshot(job):
    notes = read_json(job / "notes.json")
    source_map = read_json(job / "notes/source-map.json")
    pdf_hash = file_digest(rendered_pdf(job))
    if pdf_hash != source_map["pdf_sha256"]:
        raise ValueError(f"PDF/source-map mismatch: {job}")
    pairs = re.findall(r"\\newlabel\{fig:([^}]+)\}\{\{(\d+)\}",
                       (job / "notes/main.aux").read_text(encoding="utf-8"))
    numbers = {identifier: int(number) for identifier, number in pairs}
    if len(numbers) != len(pairs) or len(set(numbers.values())) != len(numbers):
        raise ValueError(f"Duplicate figure numbering: {job}")
    blocks = [block for chapter in notes["chapters"] for block in chapter["blocks"]]
    figures = [block for block in blocks if block["kind"] in {"diagram", "screenshot"}]
    if set(numbers) != {block["id"] for block in figures}:
        raise ValueError(f"Figure numbering/notes mismatch: {job}")
    coverage = [item for chapter in notes["chapters"]
                for item in chapter.get("illustration_coverage", [])]
    records = []
    for block in figures:
        identifier = block["id"]
        source = source_map["blocks"][identifier]
        if any(source.get(key, "") != block.get(key, "")
               for key in ("kind", "content", "illustration_id", "illustration_crop_id")):
            raise ValueError(f"Notes/source-map mismatch: {job}/{identifier}")
        asset = source["figure_asset"]
        path = (job / "notes" / asset["path"]).resolve()
        if not path.is_relative_to((job / "notes").resolve()):
            raise ValueError(f"Figure asset outside notes: {path}")
        asset_hash = file_digest(path)
        if asset_hash != asset["sha256"]:
            raise ValueError(f"Figure asset/source-map mismatch: {path}")
        identities = {item["illustration_id"] for item in coverage
                      if item["block_id"] == identifier}
        if block.get("illustration_id"):
            identities.add(block["illustration_id"])
        issues = [issue["id"] for issue in notes.get("issues", [])
                  if issue.get("status") == "open"
                  and identities.intersection(issue.get("evidence_ids", []))]
        crop_id = block.get("illustration_crop_id", "")
        crop_verdict = job / "illustration-crops" / crop_id / "verification/verdict.json"
        verdict = read_json(crop_verdict).get("verdict") if crop_id and crop_verdict.exists() else None
        route = ("verified_crop" if verdict == "pass" else "unverified_crop") if crop_id else (
            "original_crop_with_review" if issues else "original_crop")
        context = {key: block.get(key, "") for key in ("kind", "caption", "content", "topic")}
        context["referring_prose"] = sorted(item["content"] for item in blocks
                                           if identifier in item.get("figure_refs", []))
        records.append({"number": numbers[identifier], "block_id": identifier,
                        "illustration_id": block.get("illustration_id", ""),
                        "source_ids": sorted(identities), "kind": block["kind"],
                        "asset_sha256": asset_hash, "asset_path": asset["path"],
                        "context_sha256": digest(json.dumps(context, ensure_ascii=False,
                                                            sort_keys=True).encode()),
                        "crop_id": crop_id, "route": route, "open_review_ids": issues})
    evidence = read_json(job / "evidence.json")
    config_path = job / "extraction-config.json"
    config = read_json(config_path) if config_path.exists() else {}
    expects_visuals = config.get("pipeline") == "audio_video" and config.get("vision_model") != "off"
    return {"job": str(job.resolve()), "pdf_sha256": pdf_hash,
            "notes_sha256": file_digest(job / "notes.json"),
            "figures": sorted(records, key=lambda item: item["number"]),
            "excluded_sources": {item["illustration_id"]: item["exclusion_reason"]
                                 for item in coverage if not item["block_id"]},
            "visual_evidence": {"expected": expects_visuals,
                                "frames": len(evidence.get("visuals", [])),
                                "missing_stage": expects_visuals and "visuals" not in evidence}}


def split(name):
    """Keep every observation from a lecture in the same stable 80/20 split."""
    return "holdout" if int(digest(name.encode())[:8], 16) % 5 == 0 else "development"


def freeze(labels_path, roots):
    lectures = []
    for name, labels in parse_labels(labels_path).items():
        state = snapshot(locate(roots, name))
        available = {item["number"] for item in state["figures"]}
        if available != set(labels["labels"]):
            raise ValueError(f"Label/figure numbering mismatch in {name}: "
                             f"unlabeled={sorted(available - labels['labels'].keys())}, "
                             f"absent={sorted(labels['labels'].keys() - available)}")
        if labels["missing_figures"] and available:
            raise ValueError(f"Missing-figure label conflicts with rendered figures: {name}")
        for record in state["figures"]:
            record.update(labels["labels"][record["number"]])
        lectures.append({"id": name, "split": split(name),
                         "missing_figures": labels["missing_figures"], **state})
    cases = [case for lecture in lectures for case in lecture["figures"]]
    summary = {"figures": len(cases), "labels": dict(Counter(case["label"] for case in cases)),
               "bad_reasons": dict(Counter(case["reason"] for case in cases if case["label"] == "bad")),
               "routes": {label: dict(Counter(case["route"] for case in cases if case["label"] == label))
                          for label in ("good", "bad")},
               "missing_figure_lectures": sum(bool(item["missing_figures"]) for item in lectures),
               "missing_visual_stages": sum(item["visual_evidence"]["missing_stage"] for item in lectures)}
    return {"version": 1, "labels_sha256": file_digest(labels_path),
            "summary": summary, "lectures": lectures}


def match_figure(case, current):
    exact = [item for item in current if case["illustration_id"]
             and case["illustration_id"] == item["illustration_id"]]
    covered = [item for item in current if set(case["source_ids"]).intersection(item["source_ids"])]
    identical = [item for item in current if case["asset_sha256"] == item["asset_sha256"]]
    candidates = exact or covered or identical
    same_asset = [item for item in candidates if case["asset_sha256"] == item["asset_sha256"]]
    return same_asset or candidates


def compare(baseline, roots):
    if baseline.get("version") != 1:
        raise ValueError("Unsupported benchmark version")
    results, lectures = [], []
    for old in baseline["lectures"]:
        try:
            current = snapshot(locate(roots, old["id"]))
        except (OSError, ValueError, KeyError) as error:
            lectures.append({"id": old["id"], "split": old["split"], "error": str(error),
                             "missing_figure_check": "unavailable" if old["missing_figures"]
                             else "not_labeled"})
            results.extend({"lecture": old["id"], "split": old["split"],
                            "baseline_number": case["number"],
                            "baseline_label": case["label"], "status": "unavailable"}
                           for case in old["figures"])
            continue
        matched = set()
        for case in old["figures"]:
            candidates = match_figure(case, current["figures"])
            row = {"lecture": old["id"], "split": old["split"],
                   "baseline_number": case["number"], "baseline_block_id": case["block_id"],
                   "baseline_label": case["label"], "baseline_reason": case["reason"]}
            if len(candidates) == 1:
                candidate = candidates[0]
                matched.add(candidate["block_id"])
                unchanged = all(case[key] == candidate[key]
                                for key in ("asset_sha256", "context_sha256"))
                row.update(status="unchanged" if unchanged else "needs_review",
                           current_block_id=candidate["block_id"], current_number=candidate["number"],
                           renumbered=case["number"] != candidate["number"],
                           current_route=candidate["route"], open_review_ids=candidate["open_review_ids"])
                if unchanged:
                    row["retained_label"] = case["label"]
            else:
                row.update(status="ambiguous" if candidates else "missing",
                           candidates=[item["block_id"] for item in candidates],
                           exclusion_reasons={key: value for key, value in current["excluded_sources"].items()
                                              if key in case["source_ids"]})
            results.append(row)
        lectures.append({"id": old["id"], "split": old["split"],
                         "pdf_changed": old["pdf_sha256"] != current["pdf_sha256"],
                         "visual_evidence": current["visual_evidence"],
                         "missing_figure_check": ("needs_recall_review" if current["figures"] else "still_missing")
                         if old["missing_figures"] else "not_labeled",
                         "new_figures_needing_review": [item["block_id"] for item in current["figures"]
                                                        if item["block_id"] not in matched]})
    return {"version": 1, "summary": {
        "statuses": dict(Counter(item["status"] for item in results)),
        "by_split": {name: dict(Counter(item["status"] for item in results if item["split"] == name))
                     for name in ("development", "holdout")},
        "retained_labels": dict(Counter(item["retained_label"] for item in results if "retained_label" in item)),
        "good_figures_to_review": sum(item["baseline_label"] == "good" and item["status"] != "unchanged"
                                      for item in results),
        "missing_good_figures": sum(item["baseline_label"] == "good" and item["status"] == "missing"
                                    for item in results)}, "lectures": lectures, "cases": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze", help="Bind current human labels to saved figures")
    freeze_parser.add_argument("--labels", type=Path, default=Path("illustrations.md"))
    compare_parser = commands.add_parser("compare", help="Compare outputs with a frozen baseline")
    compare_parser.add_argument("--baseline", type=Path, required=True)
    for command in (freeze_parser, compare_parser):
        command.add_argument("--roots", type=Path, nargs="+", required=True)
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if any(args.out.resolve().is_relative_to(root.resolve()) for root in args.roots):
        parser.error("Write benchmark reports outside the lecture output roots")
    try:
        result = (freeze(args.labels, args.roots) if args.command == "freeze"
                  else compare(read_json(args.baseline), args.roots))
        # Baselines and reports are immutable snapshots; choose another path to repeat.
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
