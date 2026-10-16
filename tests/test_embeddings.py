from unittest.mock import Mock

import numpy as np

from semfs.embeddings import Embedder


def test_query_document_and_image_conventions(tmp_path):
    model = Mock()
    model.encode.return_value = np.array([[1, 0, 0]], dtype=np.float32)
    embedder = Embedder()
    embedder._model = model
    assert embedder.query("CPU scheduling").shape == (3,)
    assert model.encode.call_args.kwargs["prompt_name"] == "SearchQuery"
    assert model.encode.call_args.kwargs["normalize_embeddings"] is True

    embedder.documents(["Round robin uses a time quantum."], "notes.md")
    assert model.encode.call_args.args[0] == [
        "title: notes.md | text: Round robin uses a time quantum."
    ]
    assert model.encode.call_args.kwargs["prompt"] == ""

    image = tmp_path / "photo.png"
    embedder.image(image)
    assert model.encode.call_args.args[0] == {"image": str(image)}
    assert model.encode.call_args.kwargs["prompt"] == ""


def test_missing_cuda_uses_cpu(monkeypatch):
    import sys
    from types import SimpleNamespace

    torch = SimpleNamespace(
        float32="float32",
        bfloat16="bfloat16",
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    constructor = Mock(return_value=Mock())
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=constructor)
    )
    Embedder(device="cuda")._load()
    assert constructor.call_args.kwargs["device"] == "cpu"
    assert constructor.call_args.kwargs["config_kwargs"] == {"audio_config": None}
    assert constructor.call_args.kwargs["model_kwargs"]["torch_dtype"] == "float32"


def test_cuda_load_out_of_memory_falls_back(monkeypatch):
    import sys
    from types import SimpleNamespace

    class OutOfMemoryError(RuntimeError):
        pass

    cuda = SimpleNamespace(
        is_available=lambda: True,
        is_bf16_supported=lambda: True,
        OutOfMemoryError=OutOfMemoryError,
        empty_cache=Mock(),
    )
    torch = SimpleNamespace(float32="float32", bfloat16="bfloat16", cuda=cuda)
    constructor = Mock(side_effect=[OutOfMemoryError("full"), Mock()])
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=constructor)
    )
    Embedder()._load()
    assert constructor.call_args_list[0].kwargs["device"] == "cuda"
    assert constructor.call_args_list[1].kwargs["device"] == "cpu"
    cuda.empty_cache.assert_called_once()
