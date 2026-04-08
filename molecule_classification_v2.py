# Molecule Classification Module
# BMES 493 - Team 9 
# Anne Nguyen 
# Last Updated: 2026-04-07

# Import necessary libraries
import pandas as pd
import numpy as np
from collections import defaultdict


def expected_telomere_end(chrom_arm, contig_orientation, molecule_orientation):
    """
    Decide whether the telomere is expected at the molecule START or END based on chromosome arm, contig orientation, and molecule orientation

    INPUTS:
        • chrom_arm (str): chromosome of interest's arm (Expected values: "p" or "q")
        • contig_orientation (str): orientation of contig (Expected values: "+" or "-")
        • molecule_orientation (str): orientation of molecule (Expected values: "+" or "-"; derived from "Ori" column in the data)
            "+" means the molecule is oriented in the same direction as the contig
            "-" means the molecule is oriented in the opposite direction as the contig 

    RETURNS (str): "START" or "END"

    RAISES: ValueError if the combination of inputs is invalid
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
        raise ValueError(
            "Invalid chromosome arm / contig orientation / molecule orientation combination.")


def extract_molecule_info(df):
    """
    Build molecule-level metadata used later for distance calculations

    LOGICS:
    • For each unique molecule, define the molecule START and END in both:
        - Qmap position space
        - siteID space
    • Default logic for determining molecule START and END:
        - molecule start qmap = 0
        - molecule end qmap = Molecule Length
        - if Molecule Length is missing, fallback to Qmap_position where
          LabelChannel == 0
        - if that is also missing, fallback to max Qmap_position
    • For siteID:
        - start site = 0
        - end site = siteID where LabelChannel == 0
        - if missing, fallback to max siteID

    INPUTS:
    • df (pd.DataFrame):
        Data for one worksheet. Must contain at least:
            - "Molecule ID"
            - "LabelChannel"
            - "Qmap_position"
            - "siteID"
            - "Ori"
            Optional:
            - "Molecule Length"

    RETURNS:
    • dict:
        Dictionary keyed by Molecule ID, where each value is another dict with molecule metadata
        Each value is another dict with:
            - "start_qmap" (float)
            - "end_qmap" (float)
            - "start_site" (int)
            - "end_site" (int)
            - "ori" (str)
    """
    molecule_info = {}

    for molecule_id, sub_df in df.groupby("Molecule ID", sort=False):
        sub_df = sub_df.reset_index(drop=True)

        # Determine molecule END in qmap coordinates
        if "Molecule Length" in sub_df.columns and pd.notna(sub_df["Molecule Length"].iloc[0]):
            end_qmap = float(sub_df["Molecule Length"].iloc[0])
        else:
            zero_rows = sub_df[sub_df["LabelChannel"] == 0]
            if not zero_rows.empty:
                end_qmap = float(zero_rows["Qmap_position"].max())
            else:
                end_qmap = float(sub_df["Qmap_position"].max())

        # Determine molecule END in site coordinates
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


def get_expected_end_positions(mol_info, chrom_arm, contig_orientation):
    """
    Get the expected telomeric end and its coordinates for one molecule

    LOGICS:
    Combines molecule metadata with chromosome/contig information to determine:
        - whether the expected telomeric end is START or END
        - the qmap coordinate of that end
        - the siteID of that end

    INPUTS:
    • mol_info (dict):
        One molecule's metadata from extract_molecule_info()
        Expected keys:
            - "start_qmap"
            - "end_qmap"
            - "start_site"
            - "end_site"
            - "ori"
    • chrom_arm (str):
        "p" or "q"
    • contig_orientation (str):
        "+" or "-"

    RETURNS:
        tuple:
            (
                end_name (str),       # "START" or "END"
                end_qmap (float),     # qmap coordinate of expected telomeric end
                end_site (int)        # siteID coordinate of expected telomeric end
            )
    """
    end_name = expected_telomere_end(chrom_arm, contig_orientation, mol_info["ori"])

    if end_name == "START":
        return end_name, mol_info["start_qmap"], mol_info["start_site"]
    else:
        return end_name, mol_info["end_qmap"], mol_info["end_site"]


def choose_telomere_on_expected_side(sub_df, anchor_idx, expected_end_name, window=3):
    """
    Select a direct telomere label near the anchor on the correct side only

    LOGICS:
    • Search for rows with LabelChannel == 1 near the anchor row, but only on the expected telomeric side of the anchor:
        - if expected telomeric end is END, search from anchor toward END
        - if expected telomeric end is START, search from anchor toward START
    • Among valid candidate telomere rows, choose the one closest to the anchor row

    INPUTS:
    • sub_df (pd.DataFrame):
        Data for a single molecule, already reset_index(drop=True)
        Must contain:
            - "LabelChannel"
            - "Qmap_position"
    • anchor_idx (int):
        Row index of the target/anchor contig site (or fallback anchor site) within sub_df
    • expected_end_name (str):
        "START" or "END"
    • window (int, optional):
        - Number of rows to search on the valid side of the anchor
        - Default = 3

    RETURNS:
        pd.Series or None:
            - Returns the selected telomere row if found
            - Returns None if no valid telomere row exists in the search region
    """
    # Restrict search to only the biologically valid side of the anchor
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

    anchor_qmap = float(sub_df.iloc[anchor_idx]["Qmap_position"])

    # Choose the telomere closest to the anchor row first, then break ties using qmap distance to the anchor
    tel_rows["row_offset_from_anchor"] = abs(tel_rows.index - anchor_idx)
    tel_rows["qmap_offset_from_anchor"] = abs(tel_rows["Qmap_position"] - anchor_qmap)

    best_tel = tel_rows.sort_values(
        by=["row_offset_from_anchor", "qmap_offset_from_anchor"]
    ).iloc[0]

    return best_tel


def finding_averages(df, molecule_info, contig_id, chrom_arm, contig_orientation, target_site, tel_window=3):
    """
    Compute direction-aware averages used for fallback estimation

    LOGICS:
    • Keep the same general fallback idea as v1, but make it consistent with the updated v2 logic
    • This function computes:
        1) signed average target-site -> telomere offsets separately for START-ending and END-ending molecules
        2) signed average nearby-site -> target-site offsets
    • These averages are later used when:
        - the exact target/anchor contig site is missing
        - or a direct telomere label cannot be found

    INPUTS:
    • df (pd.DataFrame):
        Worksheet data. Must contain:
            - "Molecule ID"
            - "Contig_ID"
            - "Contig_Site"
            - "Qmap_position"
            - "siteID"
            - "LabelChannel"
    • molecule_info (dict):
        Output from extract_molecule_info()
    • contig_id (int):
        Contig ID selected by the user
    • chrom_arm (str):
            "p" or "q"
    • contig_orientation (str):
            "+" or "-"
    • target_site (int):
            Target/anchor contig site selected by the user
    • tel_window (int, optional):
        - Search window for direct telomere lookup around the target site
        - Default = 3

    RETURNS:
    tuple:
            (
                label_qmap_avg_by_end (dict),
                label_site_avg_by_end (dict),
                gap_qmap_avg (dict),
                gap_site_avg (dict)
            )
        • label_qmap_avg_by_end:
            {"START": float, "END": float}
        • label_site_avg_by_end:
            {"START": float, "END": float}
        • gap_qmap_avg:
            {nearby_site: float}
        • gap_site_avg:
            {nearby_site: float}
    """
    df_contig = df[df["Contig_ID"] == contig_id].copy()

    # Offsets from target site to direct telomere
    label_qmap_offsets = {"START": [], "END": []}
    label_site_offsets = {"START": [], "END": []}

    # Offsets from nearby site to target site
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

        # Find direct telomere using the same side-aware rule used in classification
        best_tel = choose_telomere_on_expected_side(
            sub_df=sub_df,
            anchor_idx=idx,
            expected_end_name=end_name,
            window=tel_window
        )

        if best_tel is not None:
            tel_qmap = float(best_tel["Qmap_position"])
            tel_siteid = int(best_tel["siteID"])

            # Signed offsets preserve direction
            label_qmap_offsets[end_name].append(tel_qmap - target_qmap)
            label_site_offsets[end_name].append(tel_siteid - target_siteid)

        # Look for nearby sites within +/- 5 around target site
        for other_site in range(target_site - 5, target_site + 6):
            if other_site == target_site:
                continue

            other_hits = sub_df.index[sub_df["Contig_Site"] == other_site].tolist()
            if other_hits:
                other_idx = other_hits[0]
                other_qmap = float(sub_df.iloc[other_idx]["Qmap_position"])
                other_siteid = int(sub_df.iloc[other_idx]["siteID"])

                # Signed offset from nearby site to target site
                gap_qmap_offsets[other_site].append(target_qmap - other_qmap)
                gap_site_offsets[other_site].append(target_siteid - other_siteid)

    label_qmap_avg_by_end = {
        end_name: float(np.mean(values)) if values else 0.0
        for end_name, values in label_qmap_offsets.items()
    }

    label_site_avg_by_end = {
        end_name: float(np.mean(values)) if values else 0.0
        for end_name, values in label_site_offsets.items()
    }

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
    Classify molecules into the 4 output categories

    LOGICS:
    For each molecule in the selected Contig_ID:
        1) try to find the exact target/anchor contig site
        2) if missing, fallback to the nearest site within +/- 5
        3) determine which molecule end is expected to be telomeric
        4) search for a direct telomere label only on that valid side
        5) if found, use direct distance
        6) if not found, estimate telomere location using fallback averages

    INPUTS:
    • df (pd.DataFrame):
        Worksheet data
    • molecule_info (dict):
        Output from extract_molecule_info()
    • contig_id (int):
        Selected Contig_ID
    • chrom_arm (str):
        "p" or "q"
    • contig_orientation (str):
        "+" or "-"
    • target_site (int):
        Selected target/anchor contig site
    • label_qmap_avg_by_end (dict):
        Output from finding_averages()
    • label_site_avg_by_end (dict):
        Output from finding_averages()
    • gap_qmap_avg (dict):
        Output from finding_averages()
    • gap_site_avg (dict):
        Output from finding_averages()
    • tel_window (int, optional):
        - Direct telomere search window
        - Default = 3
    • fusion_threshold (int or float, optional):
        - Threshold in bp for fused vs normal
        - Default = 10000

    RETURNS:
        tuple:
            (
                categories (dict),
                result_df (pd.DataFrame),
                total_unique_molecules (int)
            )

            • categories:
                Dictionary with keys:
                - "Normal_Telomere"
                - "Fused_Telomere"
                - "Normal_No_Telomere"
                - "Fused_No_Telomere"

            • result_df:
                Per-molecule summary with columns:
                - "Molecule_ID"
                - "Distance_bp"
                - "Number_of_Sites"
                - "Category"
                - "Telomere_Method"

            • total_unique_molecules:
                Number of unique molecules in the selected Contig_ID subset.
    """
    categories = {
        "Normal_Telomere": [],
        "Fused_Telomere": [],
        "Normal_No_Telomere": [],
        "Fused_No_Telomere": []
    }

    result_rows = []

    # Restrict analysis to the selected Contig_ID
    df_contig = df[df["Contig_ID"] == contig_id].copy()
    total_unique_molecules = df_contig["Molecule ID"].nunique()

    for molecule_id, sub_df in df_contig.groupby("Molecule ID", sort=False):
        sub_df = sub_df.reset_index(drop=True)
        mol_info = molecule_info[molecule_id]

        end_name, expected_end_qmap, expected_end_site = get_expected_end_positions(
            mol_info, chrom_arm, contig_orientation
        )

        # Step 1: try exact target site
        target_hits = sub_df.index[sub_df["Contig_Site"] == target_site].tolist()

        anchor_idx = None
        used_site = target_site
        used_fallback = False

        if target_hits:
            anchor_idx = target_hits[0]
        else:
            # Step 2: fallback to nearest available site within +/- 5
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

        # If no exact or fallback anchor can be found, record as unclassified
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

        # Step 3: look for direct telomere on the expected side only
        best_tel = choose_telomere_on_expected_side(
            sub_df=sub_df,
            anchor_idx=anchor_idx,
            expected_end_name=end_name,
            window=tel_window
        )

        if best_tel is not None:
            # Direct telomere-based classification
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
            # Fallback estimation logic:
            # estimate where the target site would be, then estimate the telomere
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


def write_summary_txt(categories, dataset_label, contig_id, target_site, total_molecules):
    """
    Write overall classification summary to a TXT file.

    Purpose:
        Save the category counts and percentages in a simple human-readable format.

    Inputs:
        categories (dict):
            Output category dictionary from classify_molecules().
        dataset_label (str):
            Usually worksheet name, e.g. "2q-".
        contig_id (int):
            Selected Contig_ID.
        target_site (int):
            Selected target/anchor contig site.
        total_molecules (int):
            Total number of unique molecules analyzed.

    Returns:
        str:
            Output TXT filename.
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


def write_per_molecule_csv(result_df, dataset_label, contig_id, target_site):
    """
    Write per-molecule summary to CSV for downstream analysis.

    Purpose:
        Save structured molecule-level output for use in later Python modules.

    Inputs:
        result_df (pd.DataFrame):
            Output dataframe from classify_molecules().
        dataset_label (str):
            Usually worksheet name, e.g. "2q-".
        contig_id (int):
            Selected Contig_ID.
        target_site (int):
            Selected target/anchor contig site.

    Returns:
        str:
            Output CSV filename.
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


def print_summary_table(categories, total_molecules):
    """
    Print classification summary in a tab-delimited table.

    Inputs:
        categories (dict):
            Output category dictionary from classify_molecules().
        total_molecules (int):
            Total number of unique molecules analyzed.

    Returns:
        None
    """
    print("Category\tCount\tPercentage")
    for category in ["Normal_Telomere", "Fused_Telomere", "Normal_No_Telomere", "Fused_No_Telomere"]:
        count = len(categories[category])
        percentage = (count / total_molecules * 100) if total_molecules > 0 else 0.0
        print(f"{category}\t{count}\t{percentage:.2f}%")


def print_per_molecule_terminal(result_df):
    """
    Print per-molecule summary lines to the terminal.

    Format:
        MoleculeID_(distance_bp,number_of_sites)

    Inputs:
        result_df (pd.DataFrame):
            Output dataframe from classify_molecules().

    Returns:
        None
    """
    if result_df.empty:
        print("No molecules were processed.")
        return

    for _, row in result_df.iterrows():
        if pd.isna(row["Distance_bp"]) or pd.isna(row["Number_of_Sites"]):
            print(f"{row['Molecule_ID']}_(NA,NA)")
        else:
            print(f"{row['Molecule_ID']}_({row['Distance_bp']},{row['Number_of_Sites']})")


def main():
    """
    Main entry point for running the molecule classification module.

    Purpose:
        1) collect terminal inputs
        2) load the correct Excel worksheet
        3) build molecule metadata
        4) compute fallback averages
        5) classify molecules
        6) print terminal output
        7) write summary TXT and per-molecule CSV

    Inputs from terminal:
        Path to Data (.xlsx)
        Chromosome Number
        Chromosome Arm
        Contig Orientation
        Contig ID
        Target Contig Site

    Returns:
        None
    """
    # Collect run settings from the user
    path = input("Path to Data (.xlsx): ").strip()
    chrom_num = input("Chromosome Number (e.g., 2): ").strip()
    chrom_arm = input("Chromosome Arm (p or q): ").strip()
    contig_orientation = input("Contig Orientation (+ or -): ").strip()
    contig_id = int(input("Contig ID (integer): ").strip())
    target_site = int(input("Target Contig Site (integer): ").strip())

    # Worksheet names follow the pattern: chromosome number + arm + orientation
    # Example: 2q-
    sheet_name = f"{chrom_num}{chrom_arm}{contig_orientation}"
    dataset_label = sheet_name

    # Load selected worksheet from Excel workbook
    df = pd.read_excel(path, sheet_name=sheet_name)

    # Basic input validation for required columns
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

    # Build reusable metadata for each molecule
    molecule_info = extract_molecule_info(df)

    # Compute averages used by fallback estimation logic
    label_qmap_avg_by_end, label_site_avg_by_end, gap_qmap_avg, gap_site_avg = finding_averages(
        df=df,
        molecule_info=molecule_info,
        contig_id=contig_id,
        chrom_arm=chrom_arm,
        contig_orientation=contig_orientation,
        target_site=target_site,
        tel_window=3
    )

    # Run the main classification step
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

    # Print run context
    print("\n==================================================")
    print(f"Dataset: {dataset_label}")
    print(f"Contig ID: {contig_id}")
    print(f"Target Contig Site: {target_site}")
    print(f"Total Molecules: {total_molecules}")
    print("==================================================")

    # Print classification summary
    print("\nClassification Summary")
    print_summary_table(categories, total_molecules)

    # Print per-molecule summary lines
    print("\n--------------------------------------------------")
    print("Per-Molecule Summary")
    print_per_molecule_terminal(result_df)
    print("--------------------------------------------------")

    # Save outputs to files
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