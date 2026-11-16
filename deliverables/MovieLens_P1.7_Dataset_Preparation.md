# MovieLens 25M P1.7 Dataset Preparation

**Status** — Full local run passed. Independent teammate reproduction is pending.

## Dataset definitions

Use the extracted MovieLens 25M `ratings.csv` at `data/raw/ml-25m/ratings.csv`. It contains 25,000,095 rows with `userId`, `movieId`, `rating`, and `timestamp`. Cutoffs are UTC and outcome windows include the start and exclude the end.

| Input or partition | Definition | Observed rows |
|---|---|---:|
| Validation training history | All ratings before 1 January 2017 | 20,798,765 |
| Validation outcomes | Ratings from 1 January 2017 through 31 December 2017 | 1,689,935 |
| Final-test training history | All ratings before 1 January 2018, including 2017 | 22,488,700 |
| Final-test outcomes | Ratings from 1 January 2018 up to 1 January 2020, exclusive. Last observed rating on 21 November 2019. | 2,511,395 |

The raw temporal partitions are available in memory through `phase_ratings` in `src/splits.py`. The preparation command materializes the derived files listed below.

The fixed evaluation cohort requires at least one rating before 2017. The fixed recommendation catalogue requires at least 10 pre-2017 ratings per movie. For each recommendation date, exclude movies the user already rated before that date. An eligible observed positive is a distinct eligible user–movie pair with a held-out rating above 3. The saved positives are for 2017 validation. Final-test outcomes remain reserved for the frozen evaluation.

## Reproduce the outputs

From the project root, use the reviewed `src/prepare_datasets.py` with `src/ratings.py`, `src/splits.py`, `src/catalogue.py`, and `src/validation.py`. Install the project requirements in the local virtual environment and extract the raw dataset to the path above. The command regenerates the files under `data/processed/`.

```powershell
.\.venv\Scripts\python.exe -m src.prepare_datasets
```

### Generated files

Paths are relative to `data/processed/`.

| File | Contents | Data rows |
|---|---|---:|
| eligible_movies.csv | Original movie IDs and pre-2017 rating counts | 17,965 |
| validation_positives.csv | Original user and movie IDs for 2017 eligible positives | 169,368 |
| mappings/validation/user_index.csv | Pre-2017 user IDs and indices | 142,184 |
| mappings/validation/movie_index.csv | Pre-2017 movie IDs and indices | 37,363 |
| mappings/test/user_index.csv | Pre-2018 user IDs and indices | 150,543 |
| mappings/test/movie_index.csv | Pre-2018 movie IDs and indices | 45,074 |
| dataset_manifest.json | Source SHA-256, UTC cutoffs, rules, training counts, and row counts and SHA-256 hashes for the six CSVs | One manifest |

### Verified console output

```text
Saved data/processed/eligible_movies.csv
Saved data/processed/validation_positives.csv
Saved data/processed/mappings/validation/user_index.csv
Saved data/processed/mappings/validation/movie_index.csv
Saved data/processed/mappings/test/user_index.csv
Saved data/processed/mappings/test/movie_index.csv
Saved data/processed/dataset_manifest.json
validation mapping: 142,184 users, 37,363 movies
test mapping: 150,543 users, 45,074 movies
Pre-2017 ratings: 20,798,765
2017 ratings: 1,689,935
Final-test training ratings: 22,488,700
Validation training users: 142,184
Validation training movies: 37,363
Eligible movies: 17,965
Eligible validation positive pairs: 169,368
Users with eligible validation positives: 4,778
```

## How the models use the mappings

Each mapping assigns consecutive zero-based indices to distinct original IDs in ascending order. User and movie indices are separate. For example, if user 10 maps to index 0 and movie 296 maps to index 1 in a particular fit, the rating `(10, 296, 4.0)` becomes the observed entry `(0, 1, 4.0)`.

- **Item-CF** can use the indices to build a sparse user–movie ratings matrix and identify item columns for similarity calculations.
- **Matrix factorization** can use the indices to look up user and movie factor vectors when fitting observed ratings and scoring candidate pairs.
- **Ranking and evaluation** use original MovieLens IDs. Translate indices back before saving predictions or top-10 lists, or retain the original candidate IDs alongside the computed scores. Save the mapping used by each fitted model with its parameters and reload them together.

The mappings include every user and movie present in that phase's training history. The 17,965-movie catalogue defines recommendation eligibility, while all 37,363 historically rated movies can contribute to the validation fit. The final-test mapping includes IDs first seen in 2017. An internal index is meaningful within its own phase mapping. The command checks that every catalogue movie has a training index in both phases.

## Teammate reproduction check

Run the command from a clean copy using the same code snapshot and extracted data. Compare the observed counts and the manifest's source and output hashes with the reference run. Obtain the reference `dataset_manifest.json` from the original verified run. For a quick comparison, place that copy at the project root as `reference_dataset_manifest.json` and run:

```powershell
Get-FileHash -Algorithm SHA256 -Path @(
    ".\reference_dataset_manifest.json",
    ".\data\processed\dataset_manifest.json"
)
```

Matching manifest hashes establish that the two runs recorded the same source hash, protocol metadata, output counts, and output hashes. If they differ, compare `source.sha256` first, then the cutoffs, rules, and individual `outputs` entries. Record the code snapshot and reproduction result before marking P1.7 Integrated.
