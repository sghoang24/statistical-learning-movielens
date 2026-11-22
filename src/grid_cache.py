"""Cache of the fallback-filled Item-CF and MF score grids for the P3.2 sweep.

The endpoint runner scores the 2017 cohort once and saves the two float64
component grids, the Item-CF direct-score mask, the fallback counts and the
ordered user and movie IDs that index them. A later weight sweep loads them
without fitting or scoring.

File: ``artifacts/grids/hybrid-components-before-<YYYY-MM-DD>.npz``
(Git-ignored). On the full validation grid (4,778 x 17,965) it is about
1.5 GB. It is written with ``model_artifacts.atomic_savez`` and read with
``allow_pickle=False``; as with model artifacts, load only caches this
project created on your machine.

A cache is reused only if all of these match the current request:

- cache schema, training cutoff;
- for each component (Item-CF, MF, fallback): the checked model-artifact
  metadata it was scored with (role, class, complete settings, cutoff,
  training-history fingerprint, model code hashes), and, when that model
  artifact file is present, its content hash (a refit makes the cache stale);
- the cohort: SHA-256 of the ordered user IDs and movie IDs, with their dtypes;
- SHA-256 of ``src/hybrid.py`` and ``src/pipeline.py`` (scoring and filling).

A content hash over every stored array and the fallback counts detects
damage. Mismatches raise ``GridCacheError`` unless the caller asks to rebuild.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from src import model_artifacts as ma
from src.hybrid import ComponentScores
from src.splits import VALIDATION_START

GRID_DIR = Path("artifacts/grids")
SCHEMA_VERSION = 1
METADATA_KEY = "__metadata__"
ROLES = ("item_cf", "matrix_factorization", "fallback")
ARRAY_KEYS = ("users", "movie_ids", "item_cf", "mf", "item_cf_direct")
SOURCE_DIR = Path(__file__).resolve().parent

CHECKED_FIELDS = ("schema_version", "training_cutoff_utc", "components", "cohort", "code")


class GridCacheError(RuntimeError):
    """A cached grid file is missing, damaged or stale."""


def grid_path(cutoff: int = VALIDATION_START, directory: Path = GRID_DIR) -> Path:
    return Path(directory) / f"hybrid-components-before-{ma._cutoff_label(cutoff)}.npz"


def _array_sha256(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(f"{array.dtype.str}|{array.shape}".encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def cohort_identity(users: np.ndarray, movie_ids: np.ndarray) -> dict:
    return {
        "n_users": int(len(users)),
        "n_movies": int(len(movie_ids)),
        "users_sha256": _array_sha256(np.asarray(users)),
        "movie_ids_sha256": _array_sha256(np.asarray(movie_ids)),
    }


def component_identity(
    fingerprint: dict,
    *,
    cutoff: int = VALIDATION_START,
    specs: dict | None = None,
) -> dict:
    """Checked metadata of each component model, without loading the models."""
    specs = ma.component_specs() if specs is None else specs
    identity = {}
    for role in ROLES:
        model_class, params = specs[role]
        identity[role] = ma.expected_metadata(role, model_class(**params), cutoff, fingerprint)
    return identity


def _artifact_content_sha256(role: str, cutoff: int, model_dir: Path) -> str | None:
    path = ma.artifact_path(role, cutoff, model_dir)
    if not path.is_file():
        return None
    try:
        return ma.read_metadata(path).get("content_sha256")
    except ma.ArtifactError as error:
        raise GridCacheError(f"Model artifact {path} is unreadable: {error}") from error


def expected_metadata(
    fingerprint: dict,
    users: np.ndarray,
    movie_ids: np.ndarray,
    *,
    cutoff: int = VALIDATION_START,
    specs: dict | None = None,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "training_cutoff_utc": int(cutoff),
        "components": component_identity(fingerprint, cutoff=cutoff, specs=specs),
        "cohort": cohort_identity(users, movie_ids),
        "code": {
            "hybrid_py_sha256": ma._source_sha256(SOURCE_DIR / "hybrid.py"),
            "pipeline_py_sha256": ma._source_sha256(SOURCE_DIR / "pipeline.py"),
        },
    }


def _content_sha256(arrays: dict[str, np.ndarray], fallback_counts: dict) -> str:
    digest = hashlib.sha256(json.dumps(fallback_counts, sort_keys=True).encode())
    for key in sorted(arrays):
        digest.update(f"{key}|".encode())
        digest.update(_array_sha256(arrays[key]).encode())
    return digest.hexdigest()


def save_component_grids(
    components: ComponentScores,
    path: Path,
    metadata: dict,
    *,
    cutoff: int = VALIDATION_START,
    model_dir: Path = ma.ARTIFACT_DIR,
) -> Path:
    """Save grids, mask, counts and ordered IDs atomically with ``metadata``."""
    if components.item_cf_direct is None:
        raise GridCacheError("Cannot cache component grids without the Item-CF direct mask.")
    if metadata["cohort"] != cohort_identity(components.users, components.movie_ids):
        raise GridCacheError("Metadata cohort does not match the component grids' IDs.")
    arrays = {
        "users": components.users,
        "movie_ids": components.movie_ids,
        "item_cf": components.item_cf,
        "mf": components.mf,
        "item_cf_direct": components.item_cf_direct,
    }
    record = {
        **metadata,
        "fallback_counts": dict(components.fallback_counts),
        "model_artifact_sha256": {
            role: _artifact_content_sha256(role, cutoff, model_dir) for role in ROLES
        },
        "content_sha256": _content_sha256(arrays, components.fallback_counts),
        "info": ma._informational(),
    }
    payload = dict(arrays)
    payload[METADATA_KEY] = np.asarray(json.dumps(record, sort_keys=True))
    return ma.atomic_savez(path, payload)


def read_grid_metadata(path: Path) -> dict:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return json.loads(str(archive[METADATA_KEY]))
    except FileNotFoundError:
        raise
    except Exception as error:
        raise GridCacheError(f"Grid cache {path} is damaged or unreadable: {error}") from error


def load_component_grids(
    path: Path,
    expected: dict,
    *,
    cutoff: int = VALIDATION_START,
    model_dir: Path = ma.ARTIFACT_DIR,
) -> ComponentScores:
    """Load cached grids after compatibility and integrity checks."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No grid cache at {path}.")
    record = read_grid_metadata(path)
    problems = [
        f"{field}: stored != expected"
        + (f" ({_short_difference(record.get(field), expected[field])})")
        for field in CHECKED_FIELDS
        if record.get(field) != expected[field]
    ]
    for role in ROLES:
        current = _artifact_content_sha256(role, cutoff, model_dir)
        stored = record.get("model_artifact_sha256", {}).get(role)
        if current is None:
            print(f"[grid cache] note: model artifact for {role} is absent; "
                  "relying on its recorded settings, history and code hashes.")
        elif current != stored:
            problems.append(f"model artifact for {role} has changed since the grids were scored")
    if problems:
        raise GridCacheError(f"Grid cache {path} is stale:\n  - " + "\n  - ".join(problems))

    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {*ARRAY_KEYS, METADATA_KEY}:
                raise GridCacheError(f"unexpected arrays {sorted(archive.files)}")
            arrays = {key: archive[key] for key in ARRAY_KEYS}
        if _content_sha256(arrays, record["fallback_counts"]) != record["content_sha256"]:
            raise GridCacheError("stored arrays or fallback counts fail the content hash")
        components = ComponentScores(
            users=arrays["users"],
            movie_ids=arrays["movie_ids"],
            item_cf=arrays["item_cf"],
            mf=arrays["mf"],
            fallback_counts={k: int(v) for k, v in record["fallback_counts"].items()},
            item_cf_direct=arrays["item_cf_direct"],
        )
    except Exception as error:
        raise GridCacheError(f"Grid cache {path} is damaged or invalid: {error}") from error
    if cohort_identity(components.users, components.movie_ids) != expected["cohort"]:
        raise GridCacheError(f"Grid cache {path} IDs do not match the expected cohort.")
    return components


def _short_difference(stored, expected) -> str:
    if isinstance(stored, dict) and isinstance(expected, dict):
        keys = sorted(k for k in set(stored) | set(expected) if stored.get(k) != expected.get(k))
        return "differs in " + ", ".join(keys)
    return f"{stored!r} vs {expected!r}"


def load_existing_component_grids(
    history,
    cohort,
    *,
    cutoff: int = VALIDATION_START,
    model_dir: Path = ma.ARTIFACT_DIR,
    grid_dir: Path = GRID_DIR,
) -> ComponentScores:
    """Load a compatible cache or raise; never fits, loads models or scores.

    For consumers such as the P3.2 sweep that must reuse the P3.1 grids.
    """
    path = grid_path(cutoff, grid_dir)
    if not path.is_file():
        raise GridCacheError(
            f"No grid cache at {path}. Run `python -m src.run_hybrid_endpoints` first."
        )
    expected = expected_metadata(ma.history_fingerprint(history), cohort.users, cohort.movie_ids,
                                 cutoff=cutoff)
    components = load_component_grids(path, expected, cutoff=cutoff, model_dir=model_dir)
    print(f"[grid cache] loaded component grids from {path}")
    return components


def load_or_build_component_grids(
    history,
    cohort,
    *,
    cutoff: int = VALIDATION_START,
    model_dir: Path = ma.ARTIFACT_DIR,
    grid_dir: Path = GRID_DIR,
    rebuild_on_mismatch: bool = False,
    refit_on_mismatch: bool = False,
) -> tuple[ComponentScores, dict | None, str]:
    """Return (component grids, fitted models or None, status).

    status is "loaded" (cache reused; models not loaded), "built" (no cache)
    or "rebuilt" (stale cache replaced because ``rebuild_on_mismatch``).
    """
    from src.hybrid import component_scores_from_models

    fingerprint = ma.history_fingerprint(history)
    specs = ma.component_specs()
    path = grid_path(cutoff, grid_dir)
    status = "built"
    if path.is_file():
        expected = expected_metadata(fingerprint, cohort.users, cohort.movie_ids,
                                     cutoff=cutoff, specs=specs)
        try:
            components = load_component_grids(path, expected, cutoff=cutoff, model_dir=model_dir)
            print(f"[grid cache] loaded component grids from {path}")
            return components, None, "loaded"
        except GridCacheError as error:
            if not rebuild_on_mismatch:
                raise
            print(f"[grid cache] rebuilding: existing cache rejected.\n{error}")
            status = "rebuilt"
    else:
        print(f"[grid cache] no cache at {path}; scoring components.")

    models = ma.load_or_fit_components(history, cutoff=cutoff, directory=model_dir,
                                       refit_on_mismatch=refit_on_mismatch)
    components = component_scores_from_models(models, cohort)
    expected = expected_metadata(fingerprint, cohort.users, cohort.movie_ids,
                                 cutoff=cutoff, specs=specs)
    save_component_grids(components, path, expected, cutoff=cutoff, model_dir=model_dir)
    print(f"[grid cache] saved component grids to {path}")
    return components, models, status
