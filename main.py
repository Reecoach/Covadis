import os
import json
import pandas as pd

def analyze_results_in_folders(
    base_dir,
    target_field="cost_avg",
    compute_mi_decrease=True,
):
    """
    Analyze grid-search experiment results.

    Parameters
    ----------
    base_dir : str
        Root directory containing grid-search subfolders.
    target_field : str
        Field in results.json to average (e.g., "cost_avg").
    compute_mi_decrease : bool
        Whether to compute MI_decrease = (MI_embed - MI_perturb) / MI_embed.

    Returns
    -------
    pd.DataFrame
        Columns: gamma, cap, <target_field>, [MI_decrease]
    """

    rows = []

    for subdir in os.listdir(base_dir):
        subfolder_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(subfolder_path):
            continue

        # --------------------------------------------------
        # Parse gamma / cap from folder name
        # expected: gamma_X_cap_Y
        # --------------------------------------------------
        try:
            name = subdir
            gamma_str, cap_str = name.replace("gamma_", "").split("_cap_")
            gamma = float(gamma_str)
            cap = float(cap_str)
        except Exception:
            print(f"[Skip] Cannot parse gamma/cap from folder name: {subdir}")
            continue

        # --------------------------------------------------
        # Load results.json
        # --------------------------------------------------
        result_path = os.path.join(subfolder_path, "test_results", "results.json")
        if not os.path.exists(result_path):
            print(f"[Skip] Missing results.json in {subdir}")
            continue

        with open(result_path, "r") as f:
            records = json.load(f)

        if not isinstance(records, list) or len(records) == 0:
            print(f"[Skip] Empty or invalid results.json in {subdir}")
            continue

        # --------------------------------------------------
        # Extract target field
        # --------------------------------------------------
        values = [
            r[target_field]
            for r in records
            if target_field in r and r[target_field] is not None
        ]

        avg_target = sum(values) / len(values) if values else None

        row = {
            "gamma": gamma,
            "cap": cap,
            target_field: avg_target,
        }

        # --------------------------------------------------
        # Compute MI decrease if requested
        # --------------------------------------------------
        if compute_mi_decrease:
            mi_decreases = []
            for r in records:
                mi_embed = r.get("MI_embed_bits_per_symbol", None)
                mi_perturb = r.get("MI_perturb_bits_per_symbol", None)

                if mi_embed is None or mi_perturb is None:
                    continue
                if mi_embed <= 0:
                    continue

                mi_decreases.append(
                    (mi_embed - mi_perturb) / mi_embed
                )

            row["MI_decrease"] = (
                sum(mi_decreases) / len(mi_decreases)
                if mi_decreases else None
            )

        rows.append(row)

    # --------------------------------------------------
    # Build DataFrame and sort
    # --------------------------------------------------
    df = pd.DataFrame(rows)
    df = df.sort_values(["gamma", "cap"]).reset_index(drop=True)

    # --------------------------------------------------
    # Save CSV
    # --------------------------------------------------
    out_csv = os.path.join(base_dir, "grid_search_summary.csv")
    df.to_csv(out_csv, index=False)

    print(f"[Done] Saved summary to {out_csv}")
    return df


if __name__ == '__main__':
    base_dir = "out/grid-gamma-cap-size-full"
    df = analyze_results_in_folders(
        base_dir,
        target_field="cost_avg"
    )
    print(df)

