"""Regenerate the fixed MovieLens catalogue, 2017 validation positives, and training ID mappings."""

import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.catalogue import MIN_PRE_2017_RATINGS, build_catalogue, pre_2017_movie_counts
from src.ratings import DEFAULT_RATINGS_PATH, load_ratings
from src.splits import TEST_END, TEST_START, VALIDATION_START, phase_ratings
from src.validation import build_validation_positives

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def build_id_mappings(
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map the distinct IDs in a fit's history to consecutive indices."""
    user_ids = sorted(history["userId"].unique())
    movie_ids = sorted(history["movieId"].unique())
    users = pd.DataFrame({"userId": user_ids, "user_index": range(len(user_ids))})
    movies = pd.DataFrame({"movieId": movie_ids, "movie_index": range(len(movie_ids))})
    return users, movies


def file_checksum(path: Path) -> str:
    with path.open("rb") as stream:
        for_digest = sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            for_digest.update(chunk)
    return for_digest.hexdigest()


def verify_mappings(
    phase: str,
    history: pd.DataFrame,
    mappings: tuple[pd.DataFrame, pd.DataFrame],
    catalogue: pd.DataFrame,
) -> None:
    users, movies = mappings
    for mapping, id_col, index_col in (
        (users, "userId", "user_index"),
        (movies, "movieId", "movie_index"),
    ):
        ids = mapping[id_col]
        if (
            ids.duplicated().any()
            or not ids.is_monotonic_increasing
            or len(ids) != history[id_col].nunique()
            or not history[id_col].isin(ids).all()
            or not mapping[index_col].eq(pd.RangeIndex(len(ids))).all()
        ):
            raise ValueError(f"{phase}: invalid {id_col} mapping")
    if not catalogue["movieId"].isin(movies["movieId"]).all():
        raise ValueError(f"{phase}: catalogue movies are missing from the training index")


def main() -> None:
    ratings = load_ratings()
    movie_counts = pre_2017_movie_counts(ratings)
    catalogue = build_catalogue(movie_counts)
    history, outcomes = phase_ratings(ratings, "validation")
    validation_maps = build_id_mappings(history)
    test_history = ratings.loc[
        ratings["timestamp"] < TEST_START, ["userId", "movieId"]
    ]
    test_maps = build_id_mappings(test_history)
    verify_mappings("validation", history, validation_maps, catalogue)
    verify_mappings("test", test_history, test_maps, catalogue)
    test_history_rows = len(test_history)
    raw_rows = len(ratings)
    del test_history, ratings

    positives, _ = build_validation_positives(
        history, outcomes, catalogue, min_user_ratings=1, positive_threshold=3.0
    )

    checks = (
        ("Pre-2017 ratings", len(history), 20_798_765),
        ("2017 ratings", len(outcomes), 1_689_935),
        ("Final-test training ratings", test_history_rows, 22_488_700),
        ("Validation training users", len(validation_maps[0]), 142_184),
        ("Validation training movies", len(validation_maps[1]), 37_363),
        ("Eligible movies", len(catalogue), 17_965),
        ("Eligible validation positive pairs", len(positives), 169_368),
        ("Users with eligible validation positives", positives["userId"].nunique(), 4_778),
    )
    for label, actual, expected in checks:
        if actual != expected:
            raise ValueError(f"{label}: expected {expected:,}, got {actual:,}")

    outputs = {
        "eligible_movies.csv": catalogue,
        "validation_positives.csv": positives,
        "mappings/validation/user_index.csv": validation_maps[0],
        "mappings/validation/movie_index.csv": validation_maps[1],
        "mappings/test/user_index.csv": test_maps[0],
        "mappings/test/movie_index.csv": test_maps[1],
    }
    manifest = {
        "schema_version": 1,
        "source": {
            "dataset": "MovieLens 25M",
            "path": DEFAULT_RATINGS_PATH.relative_to(OUTPUT_DIR.parents[1]).as_posix(),
            "rows": raw_rows,
            "sha256": file_checksum(DEFAULT_RATINGS_PATH),
        },
        "utc_unix_cutoffs": {
            "validation_start": VALIDATION_START,
            "test_start": TEST_START,
            "test_end_exclusive": TEST_END,
        },
        "rules": {
            "eligibility_history": "strictly before validation_start",
            "min_user_ratings": 1,
            "min_movie_ratings": MIN_PRE_2017_RATINGS,
            "positive_rating_strictly_above": 3.0,
            "exclude_previously_rated": True,
            "training_mappings": "All distinct IDs before each phase cutoff, sorted and zero-based",
        },
        "training_history_rows": {
            "validation": len(history),
            "test": test_history_rows,
        },
        "outputs": {},
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=OUTPUT_DIR) as staging:
        stage = Path(staging)
        for relative_path, frame in outputs.items():
            path = stage / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False, lineterminator="\n")
            manifest["outputs"][relative_path] = {
                "rows": len(frame),
                "sha256": file_checksum(path),
            }

        manifest_name = "dataset_manifest.json"
        with (stage / manifest_name).open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        # Publish the manifest last. Its hashes identify a complete set of outputs.
        for relative_path in (*outputs, manifest_name):
            destination = OUTPUT_DIR / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            (stage / relative_path).replace(destination)
            print(f"Saved data/processed/{relative_path}")

    for phase, (users, movies) in (
        ("validation", validation_maps), ("test", test_maps)
    ):
        print(f"{phase} mapping: {len(users):,} users, {len(movies):,} movies")

    for label, actual, _ in checks:
        print(f"{label}: {actual:,}")


if __name__ == "__main__":
    main()
