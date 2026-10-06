import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from shad_helper import local_worker


@pytest.mark.parametrize("text", [
    "'/'konde" * 40,
    '{"summary":"' + "'/'konde" * 40 + "'/'k",
    "повторение/" * 40,
])
def test_long_consecutive_word_repetition_is_detected(text):
    assert local_worker.has_repetition(text)


@pytest.mark.parametrize("text", [
    "Определим отображение T и докажем его линейность. Затем рассмотрим базис и матрицу T.",
    r"\(f(x)=x^2\), \(f'(x)=2x\), \(\nabla f=(2x,2y)\).",
    "'/'konde" * 7,
    "'/'konde" * 31,
    " " * 2000,
    "[" + ",".join("0" for _ in range(1500)) + "]",
    r"\begin{bmatrix}" + r"0 & 0 \\" * 200 + r"\end{bmatrix}",
    json.dumps({"regions": [
        {"kind": "formula", "bbox": [i / 100, 0.2, 0.3, 0.4], "text": f"a_{i}+b_{i}"}
        for i in range(20)
    ]}),
])
def test_normal_lecture_content_and_structural_repetition_are_preserved(text):
    assert not local_worker.has_repetition(text)


def test_a_long_block_requires_at_least_eight_consecutive_copies():
    block = "konde" + "-" * 59
    assert not local_worker.has_repetition(block * 7)
    assert local_worker.has_repetition(block * 8)


class FakeEOSCriteria:
    def __init__(self):
        self.resets = []
        self.eos_ids = [999]

    def reset(self, eos_ids):
        self.resets.append(eos_ids)
        self.eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]

    def __call__(self, token):
        return token in self.eos_ids


class FakeTokenizer:
    def __init__(self):
        self.stopping_criteria = FakeEOSCriteria()
        self.decode_lengths = []

    def decode(self, tokens, **kwargs):
        self.decode_lengths.append(len(tokens))
        return "".join("konde/" if token == 1 else f"term_{token};" for token in tokens)


def install_fake_generation(monkeypatch, generate):
    package = ModuleType("mlx_vlm")
    package.generate = generate
    prompts = ModuleType("mlx_vlm.prompt_utils")
    prompts.apply_chat_template = Mock(return_value="formatted lecture prompt")
    monkeypatch.setitem(sys.modules, "mlx_vlm", package)
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", prompts)
    return prompts.apply_chat_template


def test_generation_stops_repetition_preserves_raw_and_restores_eos_criteria(monkeypatch):
    tokenizer = FakeTokenizer()
    original = tokenizer.stopping_criteria
    processor = SimpleNamespace(tokenizer=tokenizer)
    model = SimpleNamespace(config=SimpleNamespace(eos_token_id=[999, 998]))
    attempts = []

    def generate(model, processor, prompt, **kwargs):
        # Match MLX-VLM: generate installs the supplied callable and checks it
        # before adding each token to the printable output.
        processor.tokenizer.stopping_criteria = kwargs["stopping_criteria"]
        emitted = []
        for _ in range(kwargs["max_tokens"]):
            attempts.append(1)
            if processor.tokenizer.stopping_criteria(1):
                break
            emitted.append(1)
        return SimpleNamespace(text=tokenizer.decode(emitted))

    template = install_fake_generation(monkeypatch, generate)
    with pytest.raises(local_worker.RepetitionError) as raised:
        local_worker._generate(model, processor, "image.png", "lecture", 4096,
                               detect_repetition=True)
    assert "konde/" in raised.value.raw
    assert 0 < len(raised.value.raw) < 1000
    assert len(attempts) < 128
    assert raised.value.raw == "konde/" * (len(attempts) - 1)
    assert tokenizer.stopping_criteria is original
    assert original.resets[-1] == [999, 998]
    template.assert_called_once_with(processor, model.config, "lecture", num_images=1,
                                     enable_thinking=False)


def test_repetition_guard_decodes_only_a_bounded_recent_token_window(monkeypatch):
    tokenizer = FakeTokenizer()
    processor = SimpleNamespace(tokenizer=tokenizer)
    model = SimpleNamespace(config=SimpleNamespace(eos_token_id=[9999]))

    def generate(model, processor, prompt, **kwargs):
        processor.tokenizer.stopping_criteria = kwargs["stopping_criteria"]
        emitted = []
        for token in [*range(2, 1102), *([1] * 200)]:
            if processor.tokenizer.stopping_criteria(token):
                break
            emitted.append(token)
        return SimpleNamespace(text=tokenizer.decode(emitted))

    install_fake_generation(monkeypatch, generate)
    with pytest.raises(local_worker.RepetitionError):
        local_worker._generate(model, processor, "image.png", "lecture", 4096,
                               detect_repetition=True)
    # The final decode belongs to our fake backend's full raw-text construction.
    guard_windows = tokenizer.decode_lengths[:-1]
    assert max(guard_windows) == 1024
    assert all(length <= 1024 for length in guard_windows)


def test_live_callback_detects_a_long_russian_repetition_block(monkeypatch):
    class CharacterTokenizer(FakeTokenizer):
        def decode(self, tokens, **kwargs):
            return "".join(map(chr, tokens))

    tokenizer = CharacterTokenizer()
    original = tokenizer.stopping_criteria
    processor = SimpleNamespace(tokenizer=tokenizer)
    model = SimpleNamespace(config=SimpleNamespace(eos_token_id=[0]))
    unit = " ".join(f"пункт{i:02d}" for i in range(16)) + "|"
    text = unit * 16
    attempts = []

    def generate(model, processor, prompt, **kwargs):
        processor.tokenizer.stopping_criteria = kwargs["stopping_criteria"]
        emitted = []
        for token in map(ord, text):
            attempts.append(token)
            if processor.tokenizer.stopping_criteria(token):
                break
            emitted.append(token)
        return SimpleNamespace(text=tokenizer.decode(emitted))

    install_fake_generation(monkeypatch, generate)
    with pytest.raises(local_worker.RepetitionError) as raised:
        local_worker._generate(model, processor, "image.png", "lecture", 4096,
                               detect_repetition=True)
    assert len(unit) == 128
    assert len(attempts) == 1024
    assert len(attempts) < 4096
    assert raised.value.raw == text[:1023]
    assert tokenizer.stopping_criteria is original


def test_generation_preserves_normal_eos_and_restores_criteria(monkeypatch):
    tokenizer = FakeTokenizer()
    original = tokenizer.stopping_criteria
    processor = SimpleNamespace(tokenizer=tokenizer)
    model = SimpleNamespace(config=SimpleNamespace(eos_token_id=[999]))
    emitted = []

    def generate(model, processor, prompt, **kwargs):
        processor.tokenizer.stopping_criteria = kwargs["stopping_criteria"]
        for token in [2, 3, 999, 4]:
            if processor.tokenizer.stopping_criteria(token):
                break
            emitted.append(token)
        return SimpleNamespace(text=tokenizer.decode(emitted))

    install_fake_generation(monkeypatch, generate)
    assert local_worker._generate(model, processor, "image.png", "lecture", 4096,
                                  detect_repetition=True) == "term_2;term_3;"
    assert emitted == [2, 3]
    assert tokenizer.stopping_criteria is original


def test_generation_restores_eos_criteria_when_the_backend_fails(monkeypatch):
    tokenizer = FakeTokenizer()
    original = tokenizer.stopping_criteria
    processor = SimpleNamespace(tokenizer=tokenizer)
    model = SimpleNamespace(config=SimpleNamespace(eos_token_id=[999]))

    def generate(model, processor, prompt, **kwargs):
        processor.tokenizer.stopping_criteria = kwargs["stopping_criteria"]
        raise RuntimeError("simulated backend failure")

    install_fake_generation(monkeypatch, generate)
    with pytest.raises(RuntimeError, match="simulated backend failure"):
        local_worker._generate(model, processor, "image.png", "lecture", 4096,
                               detect_repetition=True)
    assert tokenizer.stopping_criteria is original


def test_generation_without_guard_keeps_existing_arguments_and_text(monkeypatch):
    generate = Mock(return_value=SimpleNamespace(text="original OCR text"))
    install_fake_generation(monkeypatch, generate)
    processor = SimpleNamespace()
    model = SimpleNamespace(config=SimpleNamespace())
    assert local_worker._generate(model, processor, "image.png", "OCR:", 4096) == "original OCR text"
    generate.assert_called_once_with(model, processor, "formatted lecture prompt", image=["image.png"],
                                     max_tokens=4096, temperature=0.0, verbose=False)
