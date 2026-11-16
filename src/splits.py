from datetime import datetime, timezone
from typing import Literal

import pandas as pd

VALIDATION_START = int(datetime(2017, 1, 1, tzinfo=timezone.utc).timestamp())
TEST_START = int(datetime(2018, 1, 1, tzinfo=timezone.utc).timestamp())
TEST_END = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp())


def split_ratings(
    ratings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return pre-2017 train, 2017 validation, and 2018–2019 test ratings."""
    timestamp = ratings["timestamp"]

    train = ratings.loc[timestamp < VALIDATION_START]
    validation = ratings.loc[
        (timestamp >= VALIDATION_START) & (timestamp < TEST_START)
    ]
    test = ratings.loc[(timestamp >= TEST_START) & (timestamp < TEST_END)]

    return train, validation, test

def phase_ratings(
    ratings: pd.DataFrame,
    phase: Literal["validation", "test"],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the history and held-out outcomes for one recommendation date."""
    if phase == "validation":
        start, end = VALIDATION_START, TEST_START
    elif phase == "test":
        start, end = TEST_START, TEST_END
    else:
        raise ValueError(f"Unknown evaluation phase: {phase!r}")

    timestamp = ratings["timestamp"]
    history = ratings.loc[timestamp < start]
    outcomes = ratings.loc[(timestamp >= start) & (timestamp < end)]
    return history, outcomes