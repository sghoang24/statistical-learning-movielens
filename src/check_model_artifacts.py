"""Synthetic save/load checks for src/model_artifacts.py.

Run from the project root with ``python -m src.check_model_artifacts``. Uses a
small synthetic history and a temporary directory; it needs no MovieLens data
and never touches ``artifacts/``. The component settings are the real ones
read from the project (selected Item-CF and MF, and the bias fallback).
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src import model_artifacts as ma
from src.splits import VALIDATION_START

ROLES = ("item_cf", "matrix_factorization", "fallback")


def synthetic_history() -> pd.DataFrame:
    """Large-ID users/movies with gaps, so positions differ from original IDs."""
    rng = np.random.default_rng(5)
    users = rng.choice(np.arange(10_000, 90_000), 40, replace=False)
    movies = rng.choice(np.arange(500, 200_000), 30, replace=False)
    rows = []
    for user in users:
        for movie in rng.choice(movies, 8, replace=False):
            rows.append((int(user), int(movie), float(rng.choice(np.arange(0.5, 5.01, 0.5)))))
    frame = pd.DataFrame(rows, columns=["userId", "movieId", "rating"])
    return frame.astype({"userId": "int32", "movieId": "int32", "rating": "float32"})


def assert_same_model(role, before, after, users, movies, pairs) -> None:
    assert type(before) is type(after), role
    assert ma.model_settings(before) == ma.model_settings(after), role
    names = ma._fitted_attributes(before)
    assert names == ma._fitted_attributes(after), role
    for name in names:
        a, b = getattr(before, name), getattr(after, name)
        assert type(a) is type(b), (role, name, type(a), type(b))
        if isinstance(a, pd.Series):
            # Original user/movie ID mapping, including index dtype.
            pd.testing.assert_series_equal(a, b, check_exact=True)
        elif isinstance(a, sp.csr_matrix):
            assert a.shape == b.shape and a.dtype == b.dtype, (role, name)
            for part in ("data", "indices", "indptr"):
                x, y = getattr(a, part), getattr(b, part)
                assert x.dtype == y.dtype and np.array_equal(x, y), (role, name, part)
        elif isinstance(a, np.ndarray):
            assert a.dtype == b.dtype and np.array_equal(a, b), (role, name)
        else:
            assert a == b, (role, name)

    grid_a = before.score_candidates(users, movies)
    grid_b = after.score_candidates(users, movies)
    assert grid_a.dtype == grid_b.dtype and grid_a.shape == grid_b.shape, role
    assert np.array_equal(grid_a, grid_b, equal_nan=True), f"{role}: candidate scores differ"
    pred_a, pred_b = before.predict(pairs), after.predict(pairs)
    assert np.array_equal(pred_a, pred_b, equal_nan=True), f"{role}: predictions differ"
    if role == "item_cf":
        # Unsupported pairs must stay NaN after loading, in the same cells.
        assert np.isnan(grid_a).any() and np.isfinite(grid_a).any()
        assert np.array_equal(np.isnan(grid_a), np.isnan(grid_b))
        assert np.isnan(pred_b).any(), "unsupported Item-CF pair should be NaN"


def expect_error(label: str, fn, contains: str) -> None:
    try:
        fn()
    except ma.ArtifactError as error:
        assert contains in str(error), (label, str(error))
        print(f"ok  rejected {label}: {str(error).splitlines()[-1].strip()[:100]}")
    else:
        raise AssertionError(f"{label}: expected ArtifactError")


def rewrite_artifact(path: Path, edit, *, rehash: bool) -> None:
    """Rewrite an artifact after ``edit(record, arrays)`` changes it in place.

    ``rehash=False`` keeps the stored content hash (an inconsistent file);
    ``rehash=True`` recomputes it, simulating a consistent but wrong manifest,
    so the required-attribute and decoding checks are exercised on their own.
    """
    with np.load(path, allow_pickle=False) as archive:
        record = json.loads(str(archive[ma.METADATA_KEY]))
        arrays = {k: archive[k] for k in archive.files if k != ma.METADATA_KEY}
    edit(record, arrays)
    if rehash:
        record["content_sha256"] = ma._content_sha256(arrays, record["attributes"])
    np.savez(path, **arrays, **{ma.METADATA_KEY: np.asarray(json.dumps(record, sort_keys=True))})


def check_attribute_manifest(directory, history, specs, fitted, users, movies, pairs) -> None:
    kwargs = dict(directory=directory, specs=specs)
    fallback_path = ma.artifact_path("fallback", VALIDATION_START, directory)
    item_cf_path = ma.artifact_path("item_cf", VALIDATION_START, directory)
    pristine = {p: p.read_bytes() for p in (fallback_path, item_cf_path)}

    def drop_bias_movie(record, arrays):
        del record["attributes"]["bias_movie_"]  # arrays left unchanged

    def set_kind(name, kind):
        return lambda record, arrays: record["attributes"].__setitem__(name, kind)

    def drop_array(key):
        return lambda record, arrays: arrays.pop(key)

    def corrupt_indptr(record, arrays):
        arrays["ratings_/indptr"] = arrays["ratings_/indptr"][::-1].copy()

    cases = [
        ("bias_movie_ removed from manifest, hash kept", fallback_path,
         drop_bias_movie, False, "content hash"),
        ("bias_movie_ removed from manifest, hash recomputed", fallback_path,
         drop_bias_movie, True, "missing attributes ['bias_movie_']"),
        ("invalid attribute kind", fallback_path,
         set_kind("mu_", "pickle"), True, "invalid kind 'pickle'"),
        ("valid but wrong attribute kind", fallback_path,
         set_kind("bias_movie_", "series"), True, "expected 'ndarray'"),
        ("unknown extra attribute", fallback_path,
         set_kind("extra_", "ndarray"), True, "unexpected attributes ['extra_']"),
        ("array missing for a listed attribute", fallback_path,
         drop_array("user_index_/values"), True, "missing arrays"),
        ("undecodable sparse matrix", item_cf_path,
         corrupt_indptr, True, "damaged or invalid"),
    ]
    role_of = {fallback_path: "fallback", item_cf_path: "item_cf"}
    for label, path, edit, rehash, contains in cases:
        path.write_bytes(pristine[path])
        rewrite_artifact(path, edit, rehash=rehash)
        expect_error(label, lambda: ma.load_or_fit(role_of[path], history, **kwargs), contains)

    # refit_on_mismatch=True recovers from a manifest problem with a correct model.
    path = fallback_path
    path.write_bytes(pristine[path])
    rewrite_artifact(path, drop_bias_movie, rehash=True)
    recovered = ma.load_or_fit("fallback", history, refit_on_mismatch=True, **kwargs)
    assert_same_model("fallback", fitted["fallback"], recovered, users, movies, pairs)
    item_cf_path.write_bytes(pristine[item_cf_path])
    print("ok  refit_on_mismatch=True recovers from an invalid attribute manifest")

    # Saving refuses a model missing a required fitted attribute.
    model = fitted["fallback"]
    saved_value = model.bias_movie_
    del model.bias_movie_
    try:
        expect_error("save with a required attribute missing",
                     lambda: ma.save_model(model, directory / "incomplete.npz", {}),
                     "missing attributes ['bias_movie_']")
        assert not (directory / "incomplete.npz").exists()
    finally:
        model.bias_movie_ = saved_value


def main() -> None:
    history = synthetic_history()
    specs = ma.component_specs()
    print("settings under test:", {r: p for r, (_, p) in specs.items()})
    users = np.concatenate([history["userId"].unique()[:12], [7]])  # 7 = unknown user
    movies = np.concatenate([history["movieId"].unique(), [3]])     # 3 = unknown movie
    pairs = pd.DataFrame({
        "userId": np.repeat(users, 3)[:30],
        "movieId": np.tile(movies, 3)[:30],
    })

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        kwargs = dict(directory=directory, specs=specs)

        # 1. First call fits and saves; second call loads without fitting.
        fitted = {role: ma.load_or_fit(role, history, **kwargs) for role in ROLES}
        original_fit = {cls: cls.fit for cls, _ in specs.values()}
        try:
            for cls in original_fit:
                cls.fit = lambda self, h: (_ for _ in ()).throw(AssertionError("refit"))
            loaded = {role: ma.load_or_fit(role, history, **kwargs) for role in ROLES}
        finally:
            for cls, fit in original_fit.items():
                cls.fit = fit
        print("ok  second call loaded all three models without fitting")

        # 2. Loaded models are identical, including ID mappings and NaN cells.
        for role in ROLES:
            assert_same_model(role, fitted[role], loaded[role], users, movies, pairs)
        print("ok  scores, predictions, Item-CF NaN cells and ID mappings identical after load")

        path = ma.artifact_path("item_cf", VALIDATION_START, directory)
        meta = ma.read_metadata(path)
        assert meta["settings"] == ma.model_settings(fitted["item_cf"])
        assert meta["training_cutoff_utc"] == VALIDATION_START
        print("ok  metadata:", sorted(meta))

        # 3. Incompatible artifacts are rejected, never silently used.
        changed = history.copy()
        changed.loc[0, "rating"] = 5.0 if changed.loc[0, "rating"] != 5.0 else 4.0
        expect_error("different training history",
                     lambda: ma.load_or_fit("item_cf", changed, **kwargs), "training_history")
        other_specs = dict(specs)
        cls, params = specs["matrix_factorization"]
        other_specs["matrix_factorization"] = (cls, {**params, "n_factors": params["n_factors"] + 1})
        expect_error("different settings",
                     lambda: ma.load_or_fit("matrix_factorization", history,
                                            directory=directory, specs=other_specs), "settings")
        other_cutoff = VALIDATION_START + 86_400
        os.replace(path, ma.artifact_path("item_cf", other_cutoff, directory))
        expect_error("different training cutoff",
                     lambda: ma.load_or_fit("item_cf", history, cutoff=other_cutoff, **kwargs),
                     "training_cutoff_utc")
        os.replace(ma.artifact_path("item_cf", other_cutoff, directory), path)
        real_hash = ma._source_sha256
        try:
            ma._source_sha256 = lambda p: "changed-code"
            expect_error("changed model code",
                         lambda: ma.load_or_fit("fallback", history, **kwargs), "code")
        finally:
            ma._source_sha256 = real_hash

        # 4. Damaged artifacts are rejected.
        good = path.read_bytes()
        path.write_bytes(good[: len(good) // 2])
        expect_error("truncated file", lambda: ma.load_or_fit("item_cf", history, **kwargs),
                     "damaged")
        # Valid zip, original metadata, but one array altered -> content hash fails.
        with np.load(ma.artifact_path("fallback", VALIDATION_START, directory)) as archive:
            payload = {key: archive[key] for key in archive.files}
        payload["bias_movie_"] = payload["bias_movie_"] + 1e-9
        fallback_path = ma.artifact_path("fallback", VALIDATION_START, directory)
        np.savez(fallback_path, **payload)
        expect_error("altered array (content hash)",
                     lambda: ma.load_or_fit("fallback", history, **kwargs), "content hash")

        # 5. Explicit refit replaces a rejected artifact; the result then loads.
        for role in ("item_cf", "fallback"):
            ma.load_or_fit(role, history, refit_on_mismatch=True, **kwargs)
            again = ma.load_or_fit(role, history, **kwargs)
            assert_same_model(role, fitted[role], again, users, movies, pairs)
        print("ok  refit_on_mismatch=True refits, saves, and the new artifact loads")

        # 5b. Altered or incomplete attribute metadata never yields a loaded model.
        check_attribute_manifest(directory, history, specs, fitted, users, movies, pairs)

        # 6. Interrupted save leaves the previous artifact intact and no partial file.
        before = path.read_bytes()
        real_replace = ma.os.replace

        def interrupted(src, dst):
            raise OSError("simulated interruption")

        ma.os.replace = interrupted
        try:
            ma.save_model(fitted["item_cf"], path, ma.read_metadata(path))
        except OSError:
            pass
        else:
            raise AssertionError("save should have failed")
        finally:
            ma.os.replace = real_replace
        assert path.read_bytes() == before, "previous artifact changed"
        assert not list(directory.glob(".*.partial")), "partial file left behind"
        new_path = directory / "never-written.npz"
        ma.os.replace = interrupted
        try:
            ma.save_model(fitted["fallback"], new_path, {"role": "x"})
        except OSError:
            pass
        finally:
            ma.os.replace = real_replace
        assert not new_path.exists() and not list(directory.glob(".*.partial"))
        ma.load_or_fit("item_cf", history, **kwargs)
        print("ok  interrupted save: previous artifact intact, nothing partial, no new file")

        # 7. Unknown fitted attribute types are refused rather than dropped.
        model = fitted["fallback"]
        model.extra_ = {"not": "storable"}
        try:
            expect_error("unsupported attribute type",
                         lambda: ma.save_model(model, directory / "x.npz", {}), "Cannot store")
        finally:
            del model.extra_

    print("model artifact checks passed")


if __name__ == "__main__":
    main()
