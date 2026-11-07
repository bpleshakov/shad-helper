import json
import tempfile
import unittest
from pathlib import Path

from scripts.illustration_benchmark import compare, digest, freeze, parse_labels, snapshot, split


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_job(root, name="lecture", *, figures=None, coverage=None, excluded=None, visuals=True):
    job = root / name
    (job / "notes/figures").mkdir(parents=True, exist_ok=True)
    if figures is None:
        figures = [{"id": "fig1", "illustration_id": "source1", "number": 1, "pixels": b"image"}]
    blocks, sources, aux = [], {}, []
    for figure in figures:
        block = {"id": figure["id"], "kind": "screenshot", "content": "Plot",
                 "caption": figure.get("caption", "Plot"), "topic": "Topic",
                 "illustration_id": figure["illustration_id"], "illustration_crop_id": ""}
        path = f"figures/{figure['id']}.png"
        (job / "notes" / path).write_bytes(figure["pixels"])
        blocks.append(block)
        sources[block["id"]] = {**block, "figure_asset": {
            "path": path, "sha256": digest(figure["pixels"])}}
        aux.append(f"\\newlabel{{fig:{block['id']}}}{{{{{figure['number']}}}{{1}}{{caption}}}}")
    coverage = coverage if coverage is not None else [{
        "illustration_id": block["illustration_id"], "block_id": block["id"], "exclusion_reason": ""}
        for block in blocks]
    coverage += [{"illustration_id": source, "block_id": "", "exclusion_reason": reason}
                 for source, reason in (excluded or {}).items()]
    write_json(job / "notes.json", {"chapters": [{"blocks": blocks, "illustration_coverage": coverage}],
                                    "issues": []})
    (job / "notes/main.pdf").write_bytes(b"pdf")
    (job / "notes/main.aux").write_text("\n".join(aux), encoding="utf-8")
    write_json(job / "notes/source-map.json", {"pdf_sha256": digest(b"pdf"), "blocks": sources})
    write_json(job / "evidence.json", {"segments": [], **({"visuals": []} if visuals else {})})
    write_json(job / "extraction-config.json", {"pipeline": "audio_video", "vision_model": "35b"})
    return job


class IllustrationBenchmarkTests(unittest.TestCase):
    def test_freeze_binds_labels_and_detects_missing_visual_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_job(root)
            write_job(root, "missing", figures=[], visuals=False)
            labels = root / "labels.md"
            labels.write_text("## lecture\n1 bad, wrong crop\n## missing\n"
                              "Иллюстраций нет, хотя лектор рисует.\n", encoding="utf-8")
            baseline = freeze(labels, [root])
            self.assertEqual(baseline["summary"]["labels"], {"bad": 1})
            self.assertEqual(baseline["summary"]["missing_visual_stages"], 1)
            figure = baseline["lectures"][0]["figures"][0]
            self.assertEqual(figure["source_ids"], ["source1"])
            self.assertEqual(figure["asset_sha256"], digest(b"image"))
            self.assertEqual(baseline, freeze(labels, [root]))

    def test_renumbering_and_new_block_id_keep_unchanged_label_via_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_job(root)
            labels = root / "labels.md"
            labels.write_text("## lecture\n1 good\n")
            baseline = freeze(labels, [root])
            write_job(root, figures=[{"id": "new_id", "illustration_id": "new_source",
                                     "number": 3, "pixels": b"image"}], coverage=[{
                                         "illustration_id": "source1", "block_id": "new_id",
                                         "exclusion_reason": ""}])
            result = compare(baseline, [root])
            row = result["cases"][0]
            self.assertEqual(row["status"], "unchanged")
            self.assertTrue(row["renumbered"])
            self.assertEqual(row["retained_label"], "good")
            self.assertFalse(result["lectures"][0]["new_figures_needing_review"])

    def test_changed_pixels_or_context_never_become_automatically_good(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_job(root)
            labels = root / "labels.md"
            labels.write_text("## lecture\n1 bad, meaningless\n")
            baseline = freeze(labels, [root])
            for change in ({"pixels": b"better"}, {"caption": "Different claim"}):
                with self.subTest(change=change):
                    write_job(root, figures=[{"id": "fig1", "illustration_id": "source1",
                                             "number": 1, "pixels": b"image", **change}])
                    row = compare(baseline, [root])["cases"][0]
                    self.assertEqual(row["status"], "needs_review")
                    self.assertNotIn("retained_label", row)

    def test_excluding_good_figure_remains_a_missing_good_case(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_job(root)
            labels = root / "labels.md"
            labels.write_text("## lecture\n1 good\n")
            baseline = freeze(labels, [root])
            write_job(root, figures=[], excluded={"source1": "unresolved_source"})
            result = compare(baseline, [root])
            self.assertEqual(result["summary"]["missing_good_figures"], 1)
            self.assertEqual(result["cases"][0]["exclusion_reasons"], {"source1": "unresolved_source"})

    def test_multiple_representatives_are_ambiguous_and_new_figures_need_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_job(root)
            labels = root / "labels.md"
            labels.write_text("## lecture\n1 good\n")
            baseline = freeze(labels, [root])
            write_job(root, figures=[{"id": f"fig{i}", "illustration_id": "source1",
                                     "number": i, "pixels": bytes([i])} for i in (1, 2)])
            result = compare(baseline, [root])
            self.assertEqual(result["cases"][0]["status"], "ambiguous")
            self.assertEqual(result["lectures"][0]["new_figures_needing_review"], ["fig1", "fig2"])

    def test_missing_figures_require_recall_review_even_after_figures_appear(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_job(root, figures=[], visuals=False)
            labels = root / "labels.md"
            labels.write_text("## lecture\nNo illustrations despite board drawings\n")
            baseline = freeze(labels, [root])
            self.assertEqual(compare(baseline, [root])["lectures"][0]["missing_figure_check"], "still_missing")
            write_job(root)
            self.assertEqual(compare(baseline, [root])["lectures"][0]["missing_figure_check"], "needs_recall_review")

    def test_freeze_rejects_unlabeled_figures_and_stale_assets_or_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = write_job(root)
            labels = root / "labels.md"
            labels.write_text("## lecture\n2 good\n")
            with self.assertRaisesRegex(ValueError, "Label/figure numbering mismatch"):
                freeze(labels, [root])
            (job / "notes/figures/fig1.png").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "asset/source-map mismatch"):
                snapshot(job)
            write_job(root)
            (job / "notes/main.pdf").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "PDF/source-map mismatch"):
                snapshot(job)

    def test_unavailable_lecture_does_not_appear_to_improve_quality(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = write_job(root)
            labels = root / "labels.md"
            labels.write_text("## lecture\n1 bad, wrong crop\n")
            baseline = freeze(labels, [root])
            (job / "notes/main.pdf").unlink()
            result = compare(baseline, [root])
            self.assertEqual(result["summary"]["statuses"], {"unavailable": 1})
            self.assertEqual(result["summary"]["retained_labels"], {})
            self.assertIn("error", result["lectures"][0])

    def test_duplicate_labels_and_lecture_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.md"
            for text in ("## lecture\n1 good\n1 bad\n", "## ../lecture\n1 good\n"):
                with self.subTest(text=text):
                    path.write_text(text)
                    with self.assertRaises(ValueError):
                        parse_labels(path)

    def test_split_depends_only_on_lecture_not_figure_order(self):
        names = ["lecture1", "lecture2", "lecture3", "lecture4", "lecture5"]
        assignments = {name: split(name) for name in names}
        self.assertEqual(assignments, {name: split(name) for name in reversed(names)})
        self.assertEqual(set(assignments.values()), {"development", "holdout"})
