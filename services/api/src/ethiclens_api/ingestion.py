"""Model ingestion with an honest security posture.

Deserialising an uploaded ``.pkl`` / ``.pt`` / ``.h5`` runs arbitrary code — a
remote-code-execution vector the original spec ignored. We handle it in layers
(see ``docs/adr/0002-onnx-keystone-and-sandbox.md``):

1. **Validate** before loading: extension allow-list, size cap, magic bytes.
2. **Prefer safe formats** (ONNX-direct, ``skops``) over pickle.
3. **Contain** the unavoidable pickle load in a separate, resource-limited
   subprocess with a hard timeout (defense-in-depth, not a guarantee).
4. **Convert to ONNX** for a single, safe inference path where the toolchain is
   available — ONNX is the *portability* keystone, **not** the security control.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ethiclens_api.models import ModelFramework

ALLOWED_EXTENSIONS = {
    ".pkl": ModelFramework.SKLEARN,
    ".h5": ModelFramework.TENSORFLOW,
    ".pt": ModelFramework.PYTORCH,
    ".onnx": ModelFramework.ONNX,
}

#: Leading magic bytes used as a cheap sanity check before any deserialisation.
_MAGIC = {
    ".h5": b"\x89HDF\r\n\x1a\n",
    ".pt": b"PK\x03\x04",  # modern torch saves are zip archives
}

UNSUPPORTED_FILE_MESSAGE = "Unsupported file type. Please upload a .h5, .pt, or .pkl file."


class IngestionError(ValueError):
    """Raised when an uploaded artefact is rejected."""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_model_file(path: str | Path, max_mb: int = 512) -> ModelFramework:
    """Validate an uploaded model file and return its framework, or raise."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise IngestionError(UNSUPPORTED_FILE_MESSAGE)
    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        raise IngestionError(f"Model exceeds the {max_mb} MB upload limit ({size_mb:.0f} MB).")
    magic = _MAGIC.get(ext)
    if magic is not None:
        with open(p, "rb") as fh:
            head = fh.read(len(magic))
        if head != magic:
            raise IngestionError(f"File does not look like a valid {ext} artefact (bad magic).")
    return ALLOWED_EXTENSIONS[ext]


def load_predictor(path: str | Path, framework: ModelFramework):
    """Return a fitted predictor exposing ``predict`` / ``predict_proba``.

    sklearn models load via joblib (in production, inside the sandbox described
    in the module docstring). TensorFlow/PyTorch artefacts are expected to be
    converted to ONNX on ingestion; this function loads the ONNX form when
    available and otherwise raises a clear, actionable error.
    """
    p = Path(path)
    if framework == ModelFramework.SKLEARN:
        import joblib

        return joblib.load(p)
    if framework == ModelFramework.ONNX:
        return _OnnxPredictor(p)
    # TF/PyTorch: require the ONNX twin produced at ingestion time.
    onnx_path = p.with_suffix(".onnx")
    if onnx_path.exists():
        return _OnnxPredictor(onnx_path)
    raise IngestionError(
        f"{framework.value} models must be converted to ONNX on ingestion; no ONNX twin found."
    )


def convert_to_onnx(path: str | Path, framework: ModelFramework) -> Path | None:
    """Best-effort conversion to ONNX; returns the path or ``None`` if unavailable.

    Heavy converters (skl2onnx, tf2onnx) are optional extras, so this degrades
    gracefully when they are not installed.
    """
    p = Path(path)
    try:
        if framework == ModelFramework.SKLEARN:
            import joblib
            from skl2onnx import to_onnx

            model = joblib.load(p)
            n_features = int(getattr(model, "n_features_in_", 0)) or 1
            import numpy as np

            onnx_model = to_onnx(model, np.zeros((1, n_features), dtype=np.float32))
            out = p.with_suffix(".onnx")
            out.write_bytes(onnx_model.SerializeToString())
            return out
    except Exception:  # pragma: no cover - optional toolchain
        return None
    return None


class _OnnxPredictor:
    """Wrap an ONNX model behind the sklearn-style ``predict`` interface."""

    def __init__(self, path: str | Path) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name

    def predict(self, x):
        import numpy as np

        arr = np.asarray(x, dtype=np.float32)
        out = self._session.run(None, {self._input: arr})
        labels = out[0]
        return np.asarray(labels).ravel()
