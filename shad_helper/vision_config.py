"""Supported local vision models pinned to the lecture-frame pilot revisions."""

DEFAULT_VISION_MODEL = "35b"
VISION_MODELS = {
    "4b": {"model": "mlx-community/Qwen3.5-4B-4bit",
           "revision": "0e7ffd5c629ef7719d4cbc04069232580bfa9d9c"},
    "9b": {"model": "mlx-community/Qwen3.5-9B-4bit",
           "revision": "8b2b98c00a6b4d291155e4890773ca8f769aee53"},
    "35b": {"model": "mlx-community/Qwen3.5-35B-A3B-4bit",
            "revision": "1e20fd8d42056f870933bf98ca6211024744f7ec"},
}
