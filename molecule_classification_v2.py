import pandas as pd
import numpy as np
from collections import defaultdict


# ============================================================
# Helper: determine which molecule end is the expected
# telomeric end based on:
# - chromosome arm
# - contig orientation
# - molecule orientation
# ============================================================
def expected_telomere_end(chrom_arm, contig_orientation, molecule_orientation):
    """
    Returns either 'START' or 'END'.

    Rules:
    - p arm, contig + : mol + -> START ; mol - -> END
    - p arm, contig - : mol + -> END   ; mol - -> START
    - q arm, contig + : mol + -> END   ; mol - -> START
    - q arm, contig - : mol + -> START ; mol - -> END
    """
    if chrom_arm == "p" and contig_orientation == "+":
        return "START" if molecule_orientation == "+" else "END"
    elif chrom_arm == "p" and contig_orientation == "-":
        return "END" if molecule_orientation == "+" else "START"
    elif chrom_arm == "q" and contig_orientation == "+":
        return "END" if molecule_orientation == "+" else "START"
    elif chrom_arm == "q" and contig_orientation == "-":
        return "START" if molecule_orientation == "+" else "END"
    else:
        raise ValueError("Invalid combination of chromosome arm / contig orientation / molecule orientation")


# ============================================================
# Helper: build molecule-level metadata
# ============================================================
def extract_molecule_info(df):
    """
    For each unique molecule:
    - start_qmap = 0
    - end_qmap = Molecule Length
      If Molecule Length is missing, fallback to Qmap_position where
      LabelChannel == 0. If that also fails, use max Qmap_position.

    For row/site distance:
    - start_site = 0
    - end_site = siteID at LabelChannel == 0 if available,
      otherwise max siteID

    Also stores molecule orientation from Ori.
    """
    molecule_info = {}

    for molecule_id, sub_df in df.groupby("Molecule ID", sort=False):
        sub_df = sub_df.reset_index(drop=True)

        # Determine molecule end in Qmap space
        if "Molecule Length" in sub_df.columns and pd.notna(sub_df["Molecule Length"].iloc[0]):
            end_qmap = float(sub_df["Molecule Length"].iloc[0])
        else:
            zero_rows = sub_df[sub_df["LabelChannel"] == 0]
            if not zero_rows.empty:
                end_qmap = float(zero_rows["Qmap_position"].max())
            else:
                end_qmap = float(sub_df["Qmap_position"].max())

        # Determine molecule end in siteID space
        zero_rows = sub_df[sub_df["LabelChannel"] == 0]
        if not zero_rows.empty:
            end_site = int(zero_rows["siteID"].max())
        else:
            end_site = int(sub_df["siteID"].max())

        molecule_info[molecule_id] = {
            "start_qmap": 0.0,
            "end_qmap": end_qmap,
            "start_site": 0,
            "end_site": end_site,
            "ori": str(sub_df["Ori"].iloc[0]).strip()
        }

    return molecule_info


# ============================================================
# Helper: get expected telomeric end positions
# ============================================================
def get_expected_end_positions(mol_info, chrom_arm, contig_orientation):
    """
    Returns:
    - expected end name ('START' or 'END')
    - expected end qmap
    - expected end site
    """
    molecule_orientation = mol_info["ori"]
    end_name = expected_telomere_end(chrom_arm, contig_orientation, molecule_orientation)

    if end_name == "START":
        return end_name, mol_info["start_qmap"], mol_info["start_site"]
    else:
        return end_name, mol_info["end_qmap"], mol_info["end_site"]


# ============================================================
# Helper: choose telomere on expected side of anchor
# ============================================================
def choose_telomere_on_expected_side(sub_df, anchor_idx, expected_end_name, window=3):
    """
    Only look for LabelChannel == 1 on the expected telomeric side
    of the anchor row.

    If expected_end_name == 'END':
      keep rows from anchor_idx to anchor_idx + window

    If expected_end_name == 'START':
      keep rows from anchor_idx - window to anchor_idx

    Among valid telomere labels, choose the one closest to the anchor.
    """
    if expected_end_name == "END":
        left = anchor_idx
        right = min(len(sub_df), anchor_idx + window + 1)
    elif expected_end_name == "START":
        left = max(0, anchor_idx - window)
        right = anchor_idx + 1
    else:
        raise ValueError("expected_end_name must be START or END")

    candidate_df = sub_df.iloc[left:right].copy()
    tel_rows = candidate_df[candidate_df["LabelChannel"] == 1].copy()

    if tel_rows.empty:
        return None

    # Closest to anchor by row index distance
    tel_rows["row_offset_from_anchor"] = abs(tel_rows.index - anchor_idx)

    # Tie-break by qmap distance to anchor
    anchor_qmap = float(sub_df.iloc[anchor_idx]["Qmap_position"])
    tel_rows["qmap_offset_from_anchor"] = abs(tel_rows["Qmap_position"] - anchor_qmap)

    best_tel = tel_rows.sort_values(
        by=["row_offset_from_anchor", "qmap_offset_from_anchor"]
    ).iloc[0]

    return best_tel


# ============================================================
# Helper: compute direction-consistent averages for fallback
# ============================================================
def finding_averages(df, molecule_info, contig_id, chrom_arm, contig_orientation, target_site, tel_window=3):
    """
    Computes direction-consistent averages for fallback estimation.

    1) label_qmap_avg_by_end:
       signed average offset from target site qmap to telomere qmap
       separately for expected START vs END molecules

    2) label_site_avg_by_end:
       signed average offset from target site siteID to telomere siteID
       separately for expected START vs END molecules

    3) gap_qmap_avg:
       signed average offset from nearby site qmap to target site qmap
       keyed by nearby site number

    4) gap_site_avg:
       signed average offset from nearby site siteID to target site siteID
       keyed by nearby site number

    These preserve the v1 skeleton but make it direction-aware and
    consistent with the updated side-restricted telomere selection.
    """
    df_contig = df[df["Contig_ID"] == contig_id].copy()

    label_qmap_offsets = {"START": [], "END": []}
    label_site_offsets = {"START": [], "END": []}

    gap_qmap_offsets = defaultdict(list)
    gap_site_offsets = defaultdict(list)

    for molecule_id, sub_df in df_contig.groupby("Molecule ID", sort=False):
        sub_df = sub_df.reset_index(drop=True)

        target_hits = sub_df.index[sub_df["Contig_Site"] == target_site].tolist()
        if not target_hits:
            continue

        idx = target_hits[0]
        target_qmap = float(sub_df.iloc[idx]["Qmap_position"])
        target_siteid = int(sub_df.iloc[idx]["siteID"])

        mol_info = molecule_info[molecule_id]
        end_name, _, _ = get_expected_end_positions(mol_info, chrom_arm, contig_orientation)

        # Find direct telomere using the same side-aware logic as classification
        best_tel = choose_telomere_on_expected_side(
            sub_df=sub_df,
            anchor_idx=idx,
            expected_end_name=end_name,
            window=tel_window
        )

        if best_tel is not None:
            tel_qmap = float(best_tel["Qmap_position"])
            tel_siteid = int(best_tel["siteID"])

            # signed target -> tel offsets
            label_qmap_offsets[end_name].append(tel_qmap - target_qmap)
            label_site_offsets[end_name].append(tel_siteid - target_siteid)

        # Compute signed nearby-site -> target-site offsets
        for other_site in range(target_site - 5, target_site + 6):
            if other_site == target_site:
                continue

            other_hits = sub_df.index[sub_df["Contig_Site"] == other_site].tolist()
            if other_hits:
                other_idx = other_hits[0]
                other_qmap = float(sub_df.iloc[other_idx]["Qmap_position"])
                other_siteid = int(sub_df.iloc[other_idx]["siteID"])

                # signed offset from nearby site to target site
                gap_qmap_offsets[other_site].append(target_qmap - other_qmap)
                gap_site_offsets[other_site].append(target_siteid - other_siteid)

    label_qmap_avg_by_end = {}
    label_site_avg_by_end = {}

    for end_name in ["START", "END"]:
        label_qmap_avg_by_end[end_name] = (
            float(np.mean(label_qmap_offsets[end_name]))
            if label_qmap_offsets[end_name] else 0.0
        )
        label_site_avg_by_end[end_name] = (
            float(np.mean(label_site_offsets[end_name]))
            if label_site_offsets[end_name] else 0.0
        )

    gap_qmap_avg = {
        site: float(np.mean(values))
        for site, values in gap_qmap_offsets.items()
        if values
    }

    gap_site_avg = {
        site: float(np.mean(values))
        for site, values in gap_site_offsets.items()
        if values
    }

    return label_qmap_avg_by_end, label_site_avg_by_end, gap_qmap_avg, gap_site_avg


# ============================================================
# Main classification logic
# ============================================================
def classify_molecules(
    df,
    molecule_info,
    contig_id,
    chrom_arm,
    contig_orientation,
    target_site,
    label_qmap_avg_by_end,
    label_site_avg_by_end,
    gap_qmap_avg,
    gap_site_avg,
    tel_window=3,
    fusion_threshold=10000
):
    """
    Classifies molecules into:
    - Normal_Telomere
    - Fused_Telomere
    - Normal_No_Telomere
    - Fused_No_Telomere

    Updated direct-label logic:
    - determine expected telomeric molecule end
    - only search for LabelChannel == 1 on that expected side of anchor
    - among those valid-side telomere labels, choose the one closest to anchor

    Updated estimation logic:
    - keep v1-style skeleton
    - use signed, direction-aware target->tel averages
    - use signed nearby-site->target-site gap averages
    """
    categories = {
        "Normal_Telomere": [],
        "Fused_Telomere": [],
        "Normal_No_Telomere": [],
        "Fused_No_Telomere": []
    }

    result_rows = []

    df_contig = df[df["Contig_ID"] == contig_id].copy()
    total_unique_molecules = df_contig["Molecule ID"].nunique()

    for molecule_id, sub_df in df_contig.groupby("Molecule ID", sort=False):
        sub_df = sub_df.reset_index(drop=True)
        mol_info = molecule_info[molecule_id]

        end_name, expected_end_qmap, expected_end_site = get_expected_end_positions(
            mol_info, chrom_arm, contig_orientation
        )

        # ----------------------------------------------------
        # Step 1: find exact target site
        # ----------------------------------------------------
        target_hits = sub_df.index[sub_df["Contig_Site"] == target_site].tolist()

        anchor_idx = None
        used_site = target_site
        used_fallback = False

        if target_hits:
            anchor_idx = target_hits[0]
        else:
            # ------------------------------------------------
            # Step 2: fallback to nearest site within +/- 5
            # ------------------------------------------------
            fallback_order = [
                target_site - 1, target_site + 1,
                target_site - 2, target_site + 2,
                target_site - 3, target_site + 3,
                target_site - 4, target_site + 4,
                target_site - 5, target_site + 5
            ]

            for candidate_site in fallback_order:
                hits = sub_df.index[sub_df["Contig_Site"] == candidate_site].tolist()
                if hits:
                    anchor_idx = hits[0]
                    used_site = candidate_site
                    used_fallback = True
                    break

        if anchor_idx is None:
            result_rows.append({
                "Molecule_ID": molecule_id,
                "Distance_bp": np.nan,
                "Number_of_Sites": np.nan,
                "Category": "UNCLASSIFIED",
                "Telomere_Method": "no_anchor_found"
            })
            continue

        anchor_qmap = float(sub_df.iloc[anchor_idx]["Qmap_position"])
        anchor_siteid = int(sub_df.iloc[anchor_idx]["siteID"])

        # ----------------------------------------------------
        # Step 3: direct telomere selection using updated rule
        # ----------------------------------------------------
        best_tel = choose_telomere_on_expected_side(
            sub_df=sub_df,
            anchor_idx=anchor_idx,
            expected_end_name=end_name,
            window=tel_window
        )

        if best_tel is not None:
            tel_qmap = float(best_tel["Qmap_position"])
            tel_siteid = int(best_tel["siteID"])

            distance_bp = abs(expected_end_qmap - tel_qmap)
            number_of_sites = abs(expected_end_site - tel_siteid)

            if distance_bp >= fusion_threshold:
                category = "Fused_Telomere"
            else:
                category = "Normal_Telomere"

            tel_method = "direct_telomere"

        else:
            # ------------------------------------------------
            # Step 4: estimate telomere position/site
            # keeping v1-style skeleton but making it signed
            # and direction-consistent
            # ------------------------------------------------
            if used_fallback:
                estimated_target_qmap = anchor_qmap + gap_qmap_avg.get(used_site, 0.0)
                estimated_target_siteid = anchor_siteid + gap_site_avg.get(used_site, 0.0)
            else:
                estimated_target_qmap = anchor_qmap
                estimated_target_siteid = anchor_siteid

            estimated_tel_qmap = estimated_target_qmap + label_qmap_avg_by_end.get(end_name, 0.0)
            estimated_tel_siteid = estimated_target_siteid + label_site_avg_by_end.get(end_name, 0.0)

            distance_bp = abs(expected_end_qmap - estimated_tel_qmap)
            number_of_sites = int(round(abs(expected_end_site - estimated_tel_siteid)))

            if distance_bp >= fusion_threshold:
                category = "Fused_No_Telomere"
            else:
                category = "Normal_No_Telomere"

            tel_method = "estimated_telomere"

        categories[category].append(molecule_id)

        result_rows.append({
            "Molecule_ID": molecule_id,
            "Distance_bp": distance_bp,
            "Number_of_Sites": number_of_sites,
            "Category": category,
            "Telomere_Method": tel_method
        })

    result_df = pd.DataFrame(result_rows)
    return categories, result_df, total_unique_molecules


# ============================================================
# Write human-readable summary TXT
# ============================================================
def write_summary_txt(categories, dataset_label, contig_id, target_site, total_molecules):
    """
    Writes overall classification summary to TXT.
    """
    output_txt = f"classification_summary_{dataset_label}_contig{contig_id}_site{target_site}.txt"

    with open(output_txt, "w") as f:
        f.write(f"Dataset: {dataset_label}\n")
        f.write(f"Contig ID: {contig_id}\n")
        f.write(f"Target Contig Site: {target_site}\n")
        f.write(f"Total Molecules: {total_molecules}\n\n")

        f.write("Category\tCount\tPercentage\n")

        for category in ["Normal_Telomere", "Fused_Telomere", "Normal_No_Telomere", "Fused_No_Telomere"]:
            count = len(categories[category])
            percentage = (count / total_molecules * 100) if total_molecules > 0 else 0.0
            f.write(f"{category}\t{count}\t{percentage:.2f}%\n")

    return output_txt


# ============================================================
# Write per-molecule CSV for downstream module
# ============================================================
def write_per_molecule_csv(result_df, dataset_label, contig_id, target_site):
    """
    Writes per-molecule structured CSV.
    Only includes columns needed for downstream module.
    """
    output_csv = f"per_molecule_summary_{dataset_label}_contig{contig_id}_site{target_site}.csv"

    output_df = result_df[[
        "Molecule_ID",
        "Distance_bp",
        "Number_of_Sites",
        "Category",
        "Telomere_Method"
    ]].copy()

    output_df.to_csv(output_csv, index=False)
    return output_csv


# ============================================================
# Pretty-print summary table in terminal
# ============================================================
def print_summary_table(categories, total_molecules):
    print("Category\tCount\tPercentage")
    for category in ["Normal_Telomere", "Fused_Telomere", "Normal_No_Telomere", "Fused_No_Telomere"]:
        count = len(categories[category])
        percentage = (count / total_molecules * 100) if total_molecules > 0 else 0.0
        print(f"{category}\t{count}\t{percentage:.2f}%")


# ============================================================
# Print per-molecule summary in terminal
# ============================================================
def print_per_molecule_terminal(result_df):
    if result_df.empty:
        print("No molecules were processed.")
        return

    for _, row in result_df.iterrows():
        if pd.isna(row["Distance_bp"]) or pd.isna(row["Number_of_Sites"]):
            print(f"{row['Molecule_ID']}_(NA,NA)")
        else:
            print(f"{row['Molecule_ID']}_({row['Distance_bp']},{row['Number_of_Sites']})")


# ============================================================
# Main
# ============================================================
def main():
    # Terminal inputs
    path = input("Path to Data (.xlsx): ").strip()
    chrom_num = input("Chromosome Number (e.g., 2): ").strip()
    chrom_arm = input("Chromosome Arm (p or q): ").strip()
    contig_orientation = input("Contig Orientation (+ or -): ").strip()
    contig_id = int(input("Contig ID (integer): ").strip())
    target_site = int(input("Target Contig Site (integer): ").strip())

    # Build sheet name from chromosome number + arm + contig orientation
    sheet_name = f"{chrom_num}{chrom_arm}{contig_orientation}"
    dataset_label = sheet_name

    # Load Excel sheet
    df = pd.read_excel(path, sheet_name=sheet_name)

    # Check required columns
    required_cols = [
        "Molecule ID",
        "Qmap_position",
        "siteID",
        "Ori",
        "Contig_ID",
        "Contig_Site",
        "LabelChannel"
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in sheet '{sheet_name}': {missing_cols}")

    # Build molecule-level metadata
    molecule_info = extract_molecule_info(df)

    # Compute direction-consistent averages for fallback estimation
    label_qmap_avg_by_end, label_site_avg_by_end, gap_qmap_avg, gap_site_avg = finding_averages(
        df=df,
        molecule_info=molecule_info,
        contig_id=contig_id,
        chrom_arm=chrom_arm,
        contig_orientation=contig_orientation,
        target_site=target_site,
        tel_window=3
    )

    # Run classification
    categories, result_df, total_molecules = classify_molecules(
        df=df,
        molecule_info=molecule_info,
        contig_id=contig_id,
        chrom_arm=chrom_arm,
        contig_orientation=contig_orientation,
        target_site=target_site,
        label_qmap_avg_by_end=label_qmap_avg_by_end,
        label_site_avg_by_end=label_site_avg_by_end,
        gap_qmap_avg=gap_qmap_avg,
        gap_site_avg=gap_site_avg,
        tel_window=3,
        fusion_threshold=10000
    )

    # Terminal output
    print("\n==================================================")
    print(f"Dataset: {dataset_label}")
    print(f"Contig ID: {contig_id}")
    print(f"Target Contig Site: {target_site}")
    print(f"Total Molecules: {total_molecules}")
    print("==================================================")

    print("\nClassification Summary")
    print_summary_table(categories, total_molecules)

    print("\n--------------------------------------------------")
    print("Per-Molecule Summary")
    print_per_molecule_terminal(result_df)
    print("--------------------------------------------------")

    # Write files
    summary_txt = write_summary_txt(
        categories=categories,
        dataset_label=dataset_label,
        contig_id=contig_id,
        target_site=target_site,
        total_molecules=total_molecules
    )

    per_molecule_csv = write_per_molecule_csv(
        result_df=result_df,
        dataset_label=dataset_label,
        contig_id=contig_id,
        target_site=target_site
    )

    print("\nFiles written:")
    print(f"Summary TXT: {summary_txt}")
    print(f"Per-molecule CSV: {per_molecule_csv}")


if __name__ == "__main__":
    main()