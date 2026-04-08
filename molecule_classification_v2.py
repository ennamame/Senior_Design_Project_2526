import pandas as pd
import numpy as np
from collections import defaultdict


# ============================================================
# Helper: determine which molecule end should be treated
# as the telomeric end based on:
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
# Helper: build molecule-level information
# ============================================================
def extract_molecule_info(df):
    """
    For each unique molecule:
    - molecule start qmap = 0
    - molecule end qmap = Molecule Length
      If Molecule Length is missing, fallback to Qmap_position where
      LabelChannel == 0. If that also fails, use max Qmap_position.

    For site-based row distance:
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
# Helper: find nearby telomere rows around anchor row
# ============================================================
def find_telomere_rows(sub_df, anchor_idx, window=3):
    """
    Searches +/- window rows around the anchor row for telomere labels
    (LabelChannel == 1).
    """
    left = max(0, anchor_idx - window)
    right = min(len(sub_df), anchor_idx + window + 1)
    window_df = sub_df.iloc[left:right].copy()
    tel_rows = window_df[window_df["LabelChannel"] == 1].copy()
    return tel_rows


# ============================================================
# Helper: get the chosen molecule end positions
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
# Helper: compute v1-style averages for fallback estimation
# ============================================================
def finding_averages(df, contig_id, target_site, tel_window=3):
    """
    Computes two fallback averages using molecules in the selected Contig_ID:

    1) label_avg:
       average Qmap distance between the target contig site and the nearest
       telomere label near that target site

    2) gap_avg:
       average Qmap distance between the target contig site and nearby
       contig sites within +/- 5

    These are used when:
    - a molecule does not have the exact target contig site
    - or no direct telomere is found near the target/anchor site
    """
    df_contig = df[df["Contig_ID"] == contig_id].copy()

    general_dist_array = []
    contig_gap_dist = defaultdict(list)

    for molecule_id, sub_df in df_contig.groupby("Molecule ID", sort=False):
        sub_df = sub_df.reset_index(drop=True)

        target_hits = sub_df.index[sub_df["Contig_Site"] == target_site].tolist()
        if not target_hits:
            continue

        idx = target_hits[0]
        target_qmap = float(sub_df.iloc[idx]["Qmap_position"])

        # Find nearby telomere rows around exact target site
        tel_rows = find_telomere_rows(sub_df, idx, window=tel_window)
        if not tel_rows.empty:
            tel_rows = tel_rows.copy()
            tel_rows["dist_to_target"] = (tel_rows["Qmap_position"] - target_qmap).abs()
            best_tel = tel_rows.sort_values("dist_to_target").iloc[0]
            tel_qmap = float(best_tel["Qmap_position"])
            general_dist_array.append(abs(target_qmap - tel_qmap))

        # Compute average gap distances to nearby sites
        for other_site in range(target_site - 5, target_site + 6):
            if other_site == target_site:
                continue

            other_hits = sub_df.index[sub_df["Contig_Site"] == other_site].tolist()
            if other_hits:
                other_qmap = float(sub_df.iloc[other_hits[0]]["Qmap_position"])
                gap_dist = abs(target_qmap - other_qmap)
                contig_gap_dist[other_site].append(gap_dist)

    label_avg = float(np.mean(general_dist_array)) if general_dist_array else 0.0

    gap_avg = {}
    for site, values in contig_gap_dist.items():
        if values:
            gap_avg[site] = float(np.mean(values))

    return label_avg, gap_avg


# ============================================================
# Main classification logic
# ============================================================
def classify_molecules(
    df,
    molecule_info,
    chrom_num,
    chrom_arm,
    contig_orientation,
    contig_id,
    target_site,
    label_avg,
    gap_avg,
    tel_window=3,
    fusion_threshold=10000
):
    """
    Classifies molecules into:
    - Normal_Telomere
    - Fused_Telomere
    - Normal_No_Telomere
    - Fused_No_Telomere

    Logic:
    1) Restrict to selected Contig_ID
    2) For each molecule, try exact target site
    3) If missing, fallback to nearest site within +/- 5
    4) Search for telomere labels within +/- 3 rows around anchor
    5) If found, choose the telomere label closest to the anchor site
    6) If not found, estimate telomere position using averages
    7) Measure distance from expected molecule end to telomere position
    8) Classify using fusion threshold
    """
    categories = {
        "Normal_Telomere": [],
        "Fused_Telomere": [],
        "Normal_No_Telomere": [],
        "Fused_No_Telomere": []
    }

    result_rows = []

    df_contig = df[df["Contig_ID"] == contig_id].copy()

    for molecule_id, sub_df in df_contig.groupby("Molecule ID", sort=False):
        sub_df = sub_df.reset_index(drop=True)
        mol_info = molecule_info[molecule_id]

        end_name, expected_end_qmap, expected_end_site = get_expected_end_positions(
            mol_info, chrom_arm, contig_orientation
        )

        # Step 1: find exact target site
        target_hits = sub_df.index[sub_df["Contig_Site"] == target_site].tolist()

        anchor_idx = None
        used_site = target_site
        used_fallback = False

        if target_hits:
            anchor_idx = target_hits[0]
        else:
            # Step 2: fallback to nearest site within +/- 5
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
            # No usable target/nearby site found for this molecule
            result_rows.append({
                "Sheet_Name": f"{chrom_num}{chrom_arm}{contig_orientation}",
                "Chromosome_Number": chrom_num,
                "Chromosome_Arm": chrom_arm,
                "Contig_Orientation": contig_orientation,
                "Contig_ID": contig_id,
                "Target_Contig_Site": target_site,
                "Molecule_ID": molecule_id,
                "Category": "UNCLASSIFIED",
                "Expected_End": end_name,
                "Used_Site": np.nan,
                "Used_Fallback": False,
                "Method": "no_anchor_found",
                "Distance_bp": np.nan,
                "Row_Distance": np.nan
            })
            continue

        anchor_qmap = float(sub_df.iloc[anchor_idx]["Qmap_position"])
        anchor_siteid = int(sub_df.iloc[anchor_idx]["siteID"])

        # Step 3: search for telomere labels near anchor
        tel_rows = find_telomere_rows(sub_df, anchor_idx, window=tel_window)

        if not tel_rows.empty:
            # Choose telomere label closest to anchor/target site
            tel_rows = tel_rows.copy()
            tel_rows["dist_to_anchor"] = (tel_rows["Qmap_position"] - anchor_qmap).abs()
            best_tel = tel_rows.sort_values("dist_to_anchor").iloc[0]

            tel_qmap = float(best_tel["Qmap_position"])
            tel_siteid = int(best_tel["siteID"])

            distance_bp = abs(expected_end_qmap - tel_qmap)
            row_distance = abs(expected_end_site - tel_siteid)

            if distance_bp >= fusion_threshold:
                category = "Fused_Telomere"
            else:
                category = "Normal_Telomere"

            method = "direct_telomere"

        else:
            # Step 4: estimate telomere/target location using averages
            if used_fallback:
                estimated_qmap = anchor_qmap + label_avg + gap_avg.get(used_site, 0.0)
                estimated_siteid = anchor_siteid + abs(target_site - used_site)
            else:
                estimated_qmap = anchor_qmap + label_avg
                estimated_siteid = anchor_siteid

            distance_bp = abs(expected_end_qmap - estimated_qmap)
            row_distance = abs(expected_end_site - estimated_siteid)

            if distance_bp >= fusion_threshold:
                category = "Fused_No_Telomere"
            else:
                category = "Normal_No_Telomere"

            method = "estimated_telomere"

        categories[category].append(molecule_id)

        result_rows.append({
            "Sheet_Name": f"{chrom_num}{chrom_arm}{contig_orientation}",
            "Chromosome_Number": chrom_num,
            "Chromosome_Arm": chrom_arm,
            "Contig_Orientation": contig_orientation,
            "Contig_ID": contig_id,
            "Target_Contig_Site": target_site,
            "Molecule_ID": molecule_id,
            "Category": category,
            "Expected_End": end_name,
            "Used_Site": used_site,
            "Used_Fallback": used_fallback,
            "Method": method,
            "Distance_bp": distance_bp,
            "Row_Distance": row_distance
        })

    result_df = pd.DataFrame(result_rows)
    return categories, result_df


# ============================================================
# Helper: write summary TXT file
# ============================================================
def write_summary_txt(categories, chrom_num, chrom_arm, contig_orientation, contig_id, target_site):
    """
    Writes a human-readable summary TXT file with counts and percentages.
    """
    prefix = f"{chrom_num}{chrom_arm}{contig_orientation}"
    output_txt = f"classification_summary_{prefix}_contig{contig_id}_site{target_site}.txt"

    total = sum(len(v) for v in categories.values())

    with open(output_txt, "w") as f:
        f.write(f"Sheet Name: {prefix}\n")
        f.write(f"Chromosome Arm + Contig Orientation: {chrom_arm}{contig_orientation}\n")
        f.write(f"Contig ID: {contig_id}\n")
        f.write(f"Target Contig Site: {target_site}\n\n")

        for category in ["Normal_Telomere", "Fused_Telomere", "Normal_No_Telomere", "Fused_No_Telomere"]:
            count = len(categories[category])
            percentage = (count / total * 100) if total > 0 else 0.0
            f.write(f"{category}\n")
            f.write(f"Count = {count}\n")
            f.write(f"Percentage = {percentage:.2f}%\n\n")

    return output_txt


# ============================================================
# Helper: write per-molecule CSV file
# ============================================================
def write_per_molecule_csv(result_df, chrom_num, chrom_arm, contig_orientation, contig_id, target_site):
    """
    Writes structured per-molecule CSV file for downstream use in another module.
    """
    prefix = f"{chrom_num}{chrom_arm}{contig_orientation}"
    output_csv = f"per_molecule_summary_{prefix}_contig{contig_id}_site{target_site}.csv"
    result_df.to_csv(output_csv, index=False)
    return output_csv


# ============================================================
# Main
# ============================================================
def main():
    # Terminal inputs
    path = input('Path to Data (.xlsx): ').strip()
    chrom_num = input('Chromosome Number (e.g., 4): ').strip()
    chrom_arm = input('Chromosome Arm (p or q): ').strip()
    contig_orientation = input('Contig Orientation (+ or -): ').strip()
    contig_id = int(input('Contig ID (integer): ').strip())
    target_site = int(input('Target Contig Site (integer): ').strip())

    # Build sheet name from chromosome number + chromosome arm + contig orientation
    sheet_name = f"{chrom_num}{chrom_arm}{contig_orientation}"

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

    # Compute averages used for estimation fallback
    label_avg, gap_avg = finding_averages(
        df=df,
        contig_id=contig_id,
        target_site=target_site,
        tel_window=3
    )

    # Run classification
    categories, result_df = classify_molecules(
        df=df,
        molecule_info=molecule_info,
        chrom_num=chrom_num,
        chrom_arm=chrom_arm,
        contig_orientation=contig_orientation,
        contig_id=contig_id,
        target_site=target_site,
        label_avg=label_avg,
        gap_avg=gap_avg,
        tel_window=3,
        fusion_threshold=10000
    )

    # Print summary to terminal
    print("\n==============================")
    print(f"Chromosome Arm + Contig Orientation: {chrom_arm}{contig_orientation}")
    print(f"Contig ID: {contig_id}")
    print(f"Target Contig Site: {target_site}")
    print("==============================")

    total = sum(len(v) for v in categories.values())

    for category in ["Normal_Telomere", "Fused_Telomere", "Normal_No_Telomere", "Fused_No_Telomere"]:
        count = len(categories[category])
        percentage = (count / total * 100) if total > 0 else 0.0
        print(f"{category}:")
        print(f"  Count = {count}")
        print(f"  Percentage = {percentage:.2f}%")

    # Optional terminal display of per-molecule results
    print("\nPer-molecule summary:")
    if result_df.empty:
        print("  No molecules were classified.")
    else:
        for _, row in result_df.iterrows():
            if row["Category"] == "UNCLASSIFIED":
                print(f"  {row['Molecule_ID']}: UNCLASSIFIED")
            else:
                print(
                    f"  {row['Molecule_ID']}_"
                    f"({row['Distance_bp']},{row['Row_Distance']}) "
                    f"[{row['Category']}]"
                )

    # Write files
    summary_txt = write_summary_txt(
        categories=categories,
        chrom_num=chrom_num,
        chrom_arm=chrom_arm,
        contig_orientation=contig_orientation,
        contig_id=contig_id,
        target_site=target_site
    )

    per_molecule_csv = write_per_molecule_csv(
        result_df=result_df,
        chrom_num=chrom_num,
        chrom_arm=chrom_arm,
        contig_orientation=contig_orientation,
        contig_id=contig_id,
        target_site=target_site
    )

    print("\nFiles written:")
    print(f"  Summary TXT: {summary_txt}")
    print(f"  Per-molecule CSV: {per_molecule_csv}")


if __name__ == "__main__":
    main()