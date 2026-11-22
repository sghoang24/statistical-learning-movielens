"""Persist fitted Phase 3 component models so they are fitted once and reused.

Covers the selected Item-CF and MF models and the P1.6 bias fallback. Settings
are read from the existing project code (``load_selected_config`` and
``FALLBACK_SETTINGS``); this module does not repeat their values.

Format
    One ``<role>-before-<YYYY-MM-DD>.npz`` file per fitted model under
    ``artifacts/models/`` (Git-ignored). The file holds every fitted attribute
    as plain NumPy arrays (ID mappings, biases, factors, sparse-matrix parts)
    plus a JSON metadata record. Nothing is pickled: files are read with
    ``allow_pickle=False``, so loading cannot execute code.

Trust
    Even so, load only artifacts this project created on your own machine.
    The checks below detect stale or damaged files, not deliberate tampering.

Compatibility
    An artifact is used only if its metadata matches what would be fitted now:
    artifact schema, model role and class, complete constructor settings,
    training cutoff, training-history fingerprint (row count and SHA-256 of the
    exact userId/movieId/rating rows, in order), and SHA-256 of the model's
    source module and ``src/models.py``. Library versions, the git commit and
    the raw-ratings SHA-256 from ``data/processed/dataset_manifest.json`` are
    recorded for information only.

Integrity
    The content hash covers every stored array and the attribute manifest
    (attribute name -> kind), so neither can change undetected. Each supported
    model class has a fixed set of required fitted attributes
    (``REQUIRED_ATTRIBUTES``); the manifest must match it exactly, and the
    stored arrays must be exactly those the manifest describes. Any failure
    while restoring the model is reported as ``ArtifactError``. A mismatch or
    damaged file raises ``ArtifactError`` unless the caller explicitly asks to
    refit.

Atomic writes
    The file is written to a temporary name in the same directory and moved
    into place with ``os.replace``, so an interrupted save leaves either the
    previous artifact or none, never a partial file under the final name.

Usage (from the project root)
    from src.model_artifacts import load_or_fit_components
    models = load_or_fit_components(history)   # history: pre-2017 ratings
    models["item_cf"], models["matrix_factorization"], models["fallback"]
"""

import dataclasses
import hashlib
import inspect
import json
import os
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp

from src.bias import BiasModel
from src.item_cf import ItemCFModel
from src.matrix_factorization import MatrixFactorizationModel
from src.splits import VALIDATION_START

ARTIFACT_DIR = Path("artifacts/models")
# 2: content hash also covers the attribute manifest; required attributes enforced.
SCHEMA_VERSION = 2
METADATA_KEY = "__metadata__"
MANIFEST_PATH = Path("data/processed/dataset_manifest.json")
MODELS_SOURCE = Path(__file__).resolve().parent / "models.py"

# Metadata fields that must match exactly for an artifact to be reused.
CHECKED_FIELDS = (
    "schema_version",
    "role",
    "model_class",
    "settings",
    "training_cutoff_utc",
    "training_history",
    "code",
)


class ArtifactError(RuntimeError):
    """A stored model is missing, damaged or incompatible with the request."""


# Exact fitted state (attribute -> kind) that score_candidates() and predict()
# rely on for each supported model. Update this table when a model's fitted
# attributes change; artifacts for other classes are refused.
REQUIRED_ATTRIBUTES: dict[type, dict[str, str]] = {
    BiasModel: {
        "mu_": "float",
        "user_ids_": "ndarray",
        "movie_ids_": "ndarray",
        "user_index_": "series",
        "movie_index_": "series",
        "bias_user_": "ndarray",
        "bias_movie_": "ndarray",
        "passes_": "int",
    },
    ItemCFModel: {
        "user_index_": "series",
        "movie_index_": "series",
        "ratings_": "csr_matrix",
        "binary_": "csr_matrix",
        "item_mean_": "ndarray",
        "similarity_": "csr_matrix",
        "residuals_": "csr_matrix",
    },
    MatrixFactorizationModel: {
        "user_index_": "series",
        "movie_index_": "series",
        "mu_": "float",
        "bias_user_": "ndarray",
        "bias_movie_": "ndarray",
        "factors_user_": "ndarray",
        "factors_movie_": "ndarray",
        "iterations_": "int",
        "training_rmse_": "float",
    },
}

# Array keys stored for each attribute kind ("" means the attribute name itself).
KIND_PARTS = {
    "int": ("",),
    "float": ("",),
    "ndarray": ("",),
    "series": ("/index", "/values"),
    "csr_matrix": ("/data", "/indices", "/indptr", "/shape"),
}


def required_attributes(model_class: type) -> dict[str, str]:
    try:
        return REQUIRED_ATTRIBUTES[model_class]
    except KeyError:
        raise ArtifactError(
            f"{model_class.__module__}.{model_class.__qualname__} is not a supported "
            "artifact model; add it to REQUIRED_ATTRIBUTES first."
        ) from None


def validate_manifest(kinds: dict, model_class: type, array_keys) -> None:
    """Manifest must equal the class's required attributes and cover all arrays."""
    if not isinstance(kinds, dict):
        raise ArtifactError("Attribute manifest is missing or not a mapping.")
    required = required_attributes(model_class)
    problems = []
    missing = sorted(set(required) - set(kinds))
    extra = sorted(set(kinds) - set(required))
    if missing:
        problems.append(f"missing attributes {missing}")
    if extra:
        problems.append(f"unexpected attributes {extra}")
    for name in sorted(set(required) & set(kinds)):
        if kinds[name] not in KIND_PARTS:
            problems.append(f"{name} has invalid kind {kinds[name]!r}")
        elif kinds[name] != required[name]:
            problems.append(f"{name} has kind {kinds[name]!r}, expected {required[name]!r}")
    if not problems:
        expected_keys = {
            f"{name}{part}" for name, kind in kinds.items() for part in KIND_PARTS[kind]
        }
        actual_keys = set(array_keys)
        if expected_keys - actual_keys:
            problems.append(f"missing arrays {sorted(expected_keys - actual_keys)}")
        if actual_keys - expected_keys:
            problems.append(f"arrays not in the manifest {sorted(actual_keys - expected_keys)}")
    if problems:
        raise ArtifactError(
            f"Attribute manifest is invalid for {model_class.__qualname__}: " + "; ".join(problems)
        )


# --------------------------------------------------------------------------- #
# Component specifications (settings come from existing project code)
# --------------------------------------------------------------------------- #


def component_specs(selection: dict[str, dict] | None = None) -> dict[str, tuple[type, dict]]:
    """Role -> (model class, constructor params) for the Phase 3 components.

    Item-CF and MF use the P2.5 selection read by the Phase 2 runner; the
    fallback uses the shared ``FALLBACK_SETTINGS``. Missing constructor
    arguments keep the class defaults, exactly as in tuning and validation.
    """
    from src.run_phase2_validation import load_selected_config
    from src.tune_models import FALLBACK_SETTINGS, MODEL_FACTORIES

    if selection is None:
        selection = load_selected_config()
    return {
        "item_cf": (MODEL_FACTORIES["item_cf"], dict(selection["item_cf"])),
        "matrix_factorization": (
            MODEL_FACTORIES["matrix_factorization"],
            dict(selection["matrix_factorization"]),
        ),
        "fallback": (BiasModel, dict(FALLBACK_SETTINGS)),
    }


# --------------------------------------------------------------------------- #
# Identity helpers
# --------------------------------------------------------------------------- #


def history_fingerprint(history: pd.DataFrame) -> dict:
    """Row count and SHA-256 of the exact training rows (order-sensitive)."""
    frame = history[["userId", "movieId", "rating"]]
    hashed = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    digest = hashlib.sha256(hashed.tobytes())
    digest.update(str([str(t) for t in frame.dtypes]).encode())
    return {"rows": int(len(frame)), "sha256": digest.hexdigest()}


def _source_sha256(path: Path) -> str:
    # Normalise line endings so a CRLF/LF checkout difference is not a change.
    text = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(text).hexdigest()


def code_fingerprint(model_class: type) -> dict:
    """Hashes of the code that defines fitting, scoring and prediction."""
    return {
        "model_module": f"{model_class.__module__}.{model_class.__qualname__}",
        "model_source_sha256": _source_sha256(inspect.getsourcefile(model_class)),
        "models_py_sha256": _source_sha256(MODELS_SOURCE),
    }


def model_settings(model) -> dict:
    """Complete constructor settings of a model (defaults included)."""
    return {
        field.name: getattr(model, field.name)
        for field in dataclasses.fields(model)
        if field.init
    }


def _cutoff_label(cutoff: int) -> str:
    return datetime.fromtimestamp(cutoff, tz=timezone.utc).strftime("%Y-%m-%d")


def artifact_path(role: str, cutoff: int = VALIDATION_START, directory: Path = ARTIFACT_DIR) -> Path:
    return Path(directory) / f"{role}-before-{_cutoff_label(cutoff)}.npz"


def expected_metadata(role: str, model, cutoff: int, fingerprint: dict) -> dict:
    """The checked metadata an artifact must carry to be reused."""
    return {
        "schema_version": SCHEMA_VERSION,
        "role": role,
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "settings": model_settings(model),
        "training_cutoff_utc": int(cutoff),
        "training_history": fingerprint,
        "code": code_fingerprint(type(model)),
    }


def _informational() -> dict:
    info = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }
    try:
        info["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        info["git_uncommitted_changes"] = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        info["git_commit"] = None
    if MANIFEST_PATH.is_file():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        info["raw_ratings_sha256"] = manifest.get("source", {}).get("sha256")
    return info


# --------------------------------------------------------------------------- #
# Fitted-state <-> arrays
# --------------------------------------------------------------------------- #


def _fitted_attributes(model) -> list[str]:
    names = sorted(k for k in vars(model) if k.endswith("_") and not k.startswith("_"))
    if not names:
        raise ArtifactError(f"{type(model).__name__} has no fitted attributes; fit it first.")
    return names


def _encode(model) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Fitted attributes -> flat arrays, plus the kind of each attribute."""
    arrays: dict[str, np.ndarray] = {}
    kinds: dict[str, str] = {}
    for name in _fitted_attributes(model):
        value = getattr(model, name)
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating, np.ndarray, pd.Series, sp.csr_matrix)
        ):
            raise ArtifactError(
                f"Cannot store {type(model).__name__}.{name} of type {type(value).__name__}; "
                "extend src/model_artifacts.py before persisting this model."
            )
        if isinstance(value, (int, np.integer)):
            kinds[name], arrays[name] = "int", np.asarray(int(value), dtype=np.int64)
        elif isinstance(value, (float, np.floating)):
            kinds[name], arrays[name] = "float", np.asarray(float(value), dtype=np.float64)
        elif isinstance(value, np.ndarray):
            if value.dtype == object:
                raise ArtifactError(f"{name} has dtype object, which cannot be stored safely.")
            kinds[name], arrays[name] = "ndarray", value
        elif isinstance(value, pd.Series):
            kinds[name] = "series"
            arrays[f"{name}/index"] = value.index.to_numpy()
            arrays[f"{name}/values"] = value.to_numpy()
        else:  # csr_matrix
            kinds[name] = "csr_matrix"
            arrays[f"{name}/data"] = value.data
            arrays[f"{name}/indices"] = value.indices
            arrays[f"{name}/indptr"] = value.indptr
            arrays[f"{name}/shape"] = np.asarray(value.shape, dtype=np.int64)
    return arrays, kinds


def _decode(arrays, kinds: dict[str, str]) -> dict[str, object]:
    restored: dict[str, object] = {}
    for name, kind in kinds.items():
        if kind == "int":
            restored[name] = int(arrays[name])
        elif kind == "float":
            restored[name] = float(arrays[name])
        elif kind == "ndarray":
            restored[name] = arrays[name]
        elif kind == "series":
            restored[name] = pd.Series(arrays[f"{name}/values"], index=arrays[f"{name}/index"])
        elif kind == "csr_matrix":
            shape = tuple(int(n) for n in arrays[f"{name}/shape"])
            restored[name] = sp.csr_matrix(
                (arrays[f"{name}/data"], arrays[f"{name}/indices"], arrays[f"{name}/indptr"]),
                shape=shape,
            )
        else:
            raise ArtifactError(f"Unknown stored attribute kind {kind!r} for {name}.")
    return restored


def _content_sha256(arrays: dict[str, np.ndarray], kinds: dict[str, str]) -> str:
    """SHA-256 over the attribute manifest and every stored array."""
    digest = hashlib.sha256()
    digest.update(json.dumps(kinds, sort_keys=True).encode())
    for key in sorted(arrays):
        array = np.ascontiguousarray(arrays[key])
        digest.update(f"{key}|{array.dtype.str}|{array.shape}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Save / load
# --------------------------------------------------------------------------- #


def save_model(model, path: Path, metadata: dict) -> Path:
    """Write a fitted model atomically. ``metadata`` is from ``expected_metadata``."""
    path = Path(path)
    arrays, kinds = _encode(model)
    # Refuse to write an artifact that could not be loaded back.
    validate_manifest(kinds, type(model), arrays)
    record = {
        **metadata,
        "attributes": kinds,
        "content_sha256": _content_sha256(arrays, kinds),
        "info": _informational(),
    }
    payload = dict(arrays)
    payload[METADATA_KEY] = np.asarray(json.dumps(record, sort_keys=True))
    return atomic_savez(path, payload)


def atomic_savez(path: Path, payload: dict[str, np.ndarray]) -> Path:
    """Write an uncompressed .npz so that ``path`` is either old, new or absent.

    Data goes to a hidden ``.partial`` file in the same directory, is flushed
    to disk, then moved into place with ``os.replace``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def read_metadata(path: Path) -> dict:
    """Metadata of a stored artifact (raises ArtifactError if unreadable)."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            return json.loads(str(archive[METADATA_KEY]))
    except FileNotFoundError:
        raise
    except Exception as error:  # zip, JSON or key errors all mean "damaged"
        raise ArtifactError(f"Artifact {path} is damaged or unreadable: {error}") from error


def _differences(stored: dict, expected: dict) -> list[str]:
    return [
        f"{field}: stored {stored.get(field)!r} != expected {expected[field]!r}"
        for field in CHECKED_FIELDS
        if stored.get(field) != expected[field]
    ]


def load_model(path: Path, expected: dict, model_class: type):
    """Load an artifact after checking compatibility and integrity."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No model artifact at {path}.")
    try:
        with np.load(path, allow_pickle=False) as archive:
            record = json.loads(str(archive[METADATA_KEY]))
            differences = _differences(record, expected)
            if differences:
                raise ArtifactError(
                    f"Artifact {path} does not match the requested model:\n  - "
                    + "\n  - ".join(differences)
                )
            arrays = {key: archive[key] for key in archive.files if key != METADATA_KEY}
    except ArtifactError:
        raise
    except Exception as error:
        raise ArtifactError(f"Artifact {path} is damaged or unreadable: {error}") from error

    try:
        kinds = record.get("attributes")
        if not isinstance(kinds, dict):
            raise ArtifactError("attribute manifest is missing or not a mapping")
        if _content_sha256(arrays, kinds) != record.get("content_sha256"):
            raise ArtifactError("stored arrays or attribute manifest fail the content hash")
        validate_manifest(kinds, model_class, arrays)
        model = model_class(**record["settings"])
        for name, value in _decode(arrays, kinds).items():
            setattr(model, name, value)
    except Exception as error:  # including ArtifactError raised above
        raise ArtifactError(f"Artifact {path} is damaged or invalid: {error}") from error
    return model


def load_or_fit(
    role: str,
    history: pd.DataFrame,
    *,
    cutoff: int = VALIDATION_START,
    directory: Path = ARTIFACT_DIR,
    refit_on_mismatch: bool = False,
    fingerprint: dict | None = None,
    specs: dict[str, tuple[type, dict]] | None = None,
):
    """Return a fitted component, loading a compatible artifact or fitting once.

    - No artifact: fit on ``history``, save, return.
    - Compatible artifact: load and return without fitting.
    - Damaged or incompatible artifact: raise ``ArtifactError`` listing the
      reasons, or, only if ``refit_on_mismatch=True``, say so, refit and
      overwrite it.

    ``history`` must be the training ratings before ``cutoff``; it is
    fingerprinted to confirm the artifact was fitted on the same rows.
    """
    specs = component_specs() if specs is None else specs
    if role not in specs:
        raise KeyError(f"Unknown component role {role!r}; expected one of {sorted(specs)}.")
    model_class, params = specs[role]
    template = model_class(**params)
    fingerprint = history_fingerprint(history) if fingerprint is None else fingerprint
    expected = expected_metadata(role, template, cutoff, fingerprint)
    path = artifact_path(role, cutoff, directory)

    if path.is_file():
        try:
            model = load_model(path, expected, model_class)
            created = read_metadata(path).get("info", {}).get("created_utc")
            print(f"[artifacts] loaded {role} from {path} (fitted {created})")
            return model
        except ArtifactError as error:
            if not refit_on_mismatch:
                raise
            print(f"[artifacts] refitting {role}: existing artifact rejected.\n{error}")
    else:
        print(f"[artifacts] no artifact for {role} at {path}; fitting.")

    model = template.fit(history)
    save_model(model, path, expected)
    print(f"[artifacts] saved {role} to {path}")
    return model


def load_or_fit_components(
    history: pd.DataFrame,
    *,
    cutoff: int = VALIDATION_START,
    directory: Path = ARTIFACT_DIR,
    refit_on_mismatch: bool = False,
    roles: tuple[str, ...] = ("item_cf", "matrix_factorization", "fallback"),
) -> dict:
    """Load or fit the selected Item-CF, MF and fallback models for ``history``."""
    specs = component_specs()
    fingerprint = history_fingerprint(history)
    return {
        role: load_or_fit(
            role,
            history,
            cutoff=cutoff,
            directory=directory,
            refit_on_mismatch=refit_on_mismatch,
            fingerprint=fingerprint,
            specs=specs,
        )
        for role in roles
    }
