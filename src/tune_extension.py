"""Run only new Phase 2 grid settings and combine them with existing results."""

import json

import pandas as pd

from src.tune_models import OUTPUT_DIR, load_history, select, tune

BASE_FILE = OUTPUT_DIR / "phase2_tuning_grid.csv"
EXTENSION_FILE = OUTPUT_DIR / "phase2_tuning_extension.csv"
COMBINED_FILE = OUTPUT_DIR / "phase2_tuning_grid_extended.csv"
SELECTION_FILE = OUTPUT_DIR / "phase2_selected_config_extended.json"

# NEW_GRIDS = {
#     "bias": [
#         {"lambda_user": value, "lambda_movie": value}
#         for value in (160.0, 320.0)
#     ],
#     "item_cf": [
#         {"top_k": k, "min_similarity": 0.0}
#         for k in (6, 8)
#     ],
# }

NEW_GRIDS = {
    "item_cf": [
        {"top_k": k, "min_similarity": 0.0}
        for k in (3, 4)
    ],
}


def key(model: str, params: dict) -> tuple[str, str]:
    return model, json.dumps(params, sort_keys=True)


def main() -> None:
    if not BASE_FILE.is_file():
        raise FileNotFoundError(
            f"{BASE_FILE} is missing. Run the corrected full grid first."
        )

    base = pd.read_csv(BASE_FILE)
    # Guard against accidentally combining with the older, uncorrected run.
    if key("item_cf", {"top_k": 12, "min_similarity": 0.0}) not in {
        key(row.model, json.loads(row.params))
        for row in base.itertuples()
    }:
        raise ValueError("Base grid does not contain the corrected k=12 run.")

    extension = (
        pd.read_csv(EXTENSION_FILE)
        if EXTENSION_FILE.is_file()
        else pd.DataFrame(columns=base.columns)
    )

    recorded = {
        key(row.model, json.loads(row.params))
        for table in (base, extension)
        for row in table.itertuples()
    }
    remaining = {
        model: [
            params for params in grid
            if key(model, params) not in recorded
        ]
        for model, grid in NEW_GRIDS.items()
    }
    remaining = {model: grid for model, grid in remaining.items() if grid}

    if remaining:
        new_results = tune(load_history(), grids=remaining)
        extension = pd.concat([extension, new_results], ignore_index=True)
        extension.to_csv(EXTENSION_FILE, index=False)
    else:
        print("All extension settings have already been run.")

    combined = pd.concat([base, extension], ignore_index=True)
    combined.to_csv(COMBINED_FILE, index=False)

    selection = select(combined)
    SELECTION_FILE.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("\nSelected configurations across original and extension grids:")
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()