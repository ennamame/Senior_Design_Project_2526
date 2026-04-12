# Molecule Classification Module
# BMES 493 - Team 9
# Anne Nguyen
# Last Updated: 2026-04-12
#
# Updates in this version:
# 1) Adds a fused-only TXT export with 3 columns:
#       Chromosome, Contig ID, Molecule ID
#    This includes only molecules classified as:
#       - Fused_Telomere
#       - Fused_No_Telomere
#
# 2) Adds support for multiple input configurations in one run so the user can
#    generate outputs for multiple chromosome/arm/contig/target-site combinations
#    and optionally combine all fused molecules into one TXT file.
#
# 3) Chromosome column in fused-only output is formatted as:
#       chromosome number + chromosome arm
#    Example:
#       2p, 2q, 4p, 13q

import pandas as pd
import numpy as np
from collections import defaultdict


def expected_telomere_end(chrom_arm, contig_orientation, molecule_orientation):
    """
    Decide whether the telomere is expected at the molecule START or END based on
    chromosome arm, contig orientation, and molecule orientation.

    INPUTS:
        chrom_arm (str):
            Chromosome arm. Expected values: "p" or "q"
        contig_orientation (str):
            Orientation of contig. Expected values: "+" or "-"
        molecule_orientation (str):
            Orientation of molecule from the "Ori" column. Expected values: "+" or "-"

    RETURNS:
        str:
            "START" or "END"

    RAISES:
        ValueError:
            If the combination of inputs is invalid.
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
            "Invalid chromosome arm / contig orientation / molecule orientation combination."
        )


def extract_molecule_info(df):
    """
    Build molecule-level metadata used later for distance calculations.

    LOGIC:
    For each unique molecule, define the molecule START and END in both:
        - qmap position space
        - siteID space

    Default logic:
        - molecule start qmap = 0
        - molecule end qmap = Molecule Length if available
        - otherwise fallback to Qmap_position where LabelChannel == 0
        - if that is missing too, fallback to max Qmap_position

    For siteID:
        - start site = 0
        - end site = siteID where LabelChannel == 0
        - if missing, fallback to max siteID

    INPUTS:
        df (pd.DataFrame):
            Data for one worksheet.

    RETURNS:
        dict:
            Keyed by Molecule ID with:
                - start_qmap (float)
                - end_qmap (float)
                - start_site (int)
                - end_site (int)
                - ori (str)
    """
    molecule_info = {}

    for molecule_id, sub_df in df.groupby("Molecule ID", sort=False):
        sub_df = sub_df.reset_index(drop=True)

        if "Molecule Length" in sub_df.columns and pd.notna(sub_df["Molecule Length"].iloc[0]):
            end_qmap = float(sub_df["Molecule Length"].iloc[0])
        else:
            zero_rows = sub_df[sub_df["LabelChannel"] == 0]
            if not zero_rows.empty:
                end_qmap = float(zero_rows["Qmap_position"].max())
            else:
                end_qmap = float(sub_df["Qmap_position"].max())

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
    Get the expected telomeric end and its coordinates for one molecule.

    INPUTS:
        mol_info (dict):
            One molecule's metadata from extract_molecule_info()
        chrom_arm (str):
            "p" or "q"
        contig_orientation (str):
            "+" or "-"

    RETURNS:
        tuple:
            (
                end_name (str),   # "START" or "END"
                end_qmap (float),
                end_site (int)
            )
    """
    end_name = expected_telomere_end(chrom_arm, contig_orientation, mol_info["ori"])

    if end_name == "START":
        return end_name, mol_info["start_qmap"], mol_info["start_site"]
    else:
        return end_name, mol_info["end_qmap"], mol_info["end_site"]


def choose_telomere_on_expected_side(sub_df, anchor_idx, expected_end_name, window=3):
    """
    Select a direct telomere label near the anchor on the biologically valid side only.

    INPUTS:
        sub_df (pd.DataFrame):
            Data for one molecule
        anchor_idx (int):
            Row index of target/anchor site
        expected_end_name (str):
            "START" or "END"
        window (int):
            Number of rows to search from the anchor on the valid side

    RETURNS:
        pd.Series or None:
            Selected telomere row if found, otherwise None
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

    anchor_qmap = float(sub_df.iloc[anchor_idx]["Qmap_position"])

    tel_rows["row_offset_from_anchor"] = abs(tel_rows.index - anchor_idx)
    tel_rows["qmap_offset_from_anchor"] = abs(tel_rows["Qmap_position"] - anchor_qmap)

    best_tel = tel_rows.sort_values(
        by=["row_offset_from_anchor", "qmap_offset_from_anchor"]
    ).iloc[0]

    return best_tel


def finding_averages(df, molecule_info, contig_id, chrom_arm, contig_orientation, target_site, tel_window=3):
    """
    Compute direction-aware averages used for fallback estimation.

    INPUTS:
        df (pd.DataFrame):
            Worksheet data
        molecule_info (dict):
            Output from extract_molecule_info()
        contig_id (int):
            Selected contig ID
        chrom_arm (str):
            "p" or "q"
        contig_orientation (str):
            "+" or "-"
        target_site (int):
            Selected target/anchor contig site
        tel_window (int):
            Search window for direct telomere lookup

    RETURNS:
        tuple:
            (
                label_qmap_avg_by_end,
                label_site_avg_by_end,
                gap_qmap_avg,
                gap_site_avg
            )
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

        best_tel = choose_telomere_on_expected_side(
            sub_df=sub_df,
            anchor_idx=idx,
            expected_end_name=end_name,
            window=tel_window
        )

        if best_tel is not None:
            tel_qmap = float(best_tel["Qmap_position"])
            tel_siteid = int(best_tel["siteID"])

            label_qmap_offsets[end_name].append(tel_qmap - target_qmap)
            label_site_offsets[end_name].append(tel_siteid - target_siteid)

        for other_site in range(target_site - 5, target_site + 6):
            if other_site == target_site:
                continue

            other_hits = sub_df.index[sub_df["Contig_Site"] == other_site].tolist()
            if other_hits:
                other_idx = other_hits[0]
                other_qmap = float(sub_df.iloc[other_idx]["Qmap_position"])
                other_siteid = int(sub_df.iloc[other_idx]["siteID"])

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
    Classify molecules into 4 categories.

    RETURNS:
        tuple:
            (
                categories (dict),
                result_df (pd.DataFrame),
                total_unique_molecules (int)
            )
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

        target_hits = sub_df.index[sub_df["Contig_Site"] == target_site].tolist()

        anchor_idx = None
        used_site = target_site
        used_fallback = False

        if target_hits:
            anchor_idx = target_hits[0]
        else:
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


def build_fused_only_df(result_df, chrom_num, chrom_arm, contig_id):
    """
    Build a dataframe containing only fused molecules for TXT export.

    Included categories:
        - Fused_Telomere
        - Fused_No_Telomere

    OUTPUT COLUMNS:
        - Chromosome
        - Contig ID
        - Molecule ID

    INPUTS:
        result_df (pd.DataFrame):
            Per-molecule classification output from classify_molecules()
        chrom_num (str):
            Chromosome number, e.g. "2", "4", "13"
        chrom_arm (str):
            Chromosome arm, "p" or "q"
        contig_id (int):
            Contig ID

    RETURNS:
        pd.DataFrame
    """
    fused_categories = ["Fused_Telomere", "Fused_No_Telomere"]

    fused_df = result_df[result_df["Category"].isin(fused_categories)].copy()

    output_df = pd.DataFrame({
        "Chromosome": [f"{chrom_num}{chrom_arm}"] * len(fused_df),
        "Contig ID": [contig_id] * len(fused_df),
        "Molecule ID": fused_df["Molecule_ID"].astype(int).tolist() if not fused_df.empty else []
    })

    return output_df


def write_fused_only_txt(fused_df, output_txt):
    """
    Write fused-only molecules to a TXT file with exactly 3 columns:
        Chromosome, Contig ID, Molecule ID

    INPUTS:
        fused_df (pd.DataFrame):
            DataFrame from build_fused_only_df()
        output_txt (str):
            Output TXT filename

    RETURNS:
        str:
            Output filename
    """
    fused_df.to_csv(output_txt, sep="\t", index=False)
    return output_txt


def print_summary_table(categories, total_molecules):
    """
    Print classification summary in a tab-delimited table.
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
    """
    if result_df.empty:
        print("No molecules were processed.")
        return

    for _, row in result_df.iterrows():
        if pd.isna(row["Distance_bp"]) or pd.isna(row["Number_of_Sites"]):
            print(f"{row['Molecule_ID']}_(NA,NA)")
        else:
            print(f"{row['Molecule_ID']}_({row['Distance_bp']},{row['Number_of_Sites']})")


def validate_required_columns(df, sheet_name):
    """
    Check that all required columns exist in the worksheet.
    """
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


def run_single_configuration(path, chrom_num, chrom_arm, contig_orientation, contig_id, target_site):
    """
    Run the full classification workflow for one user-specified configuration.

    INPUTS:
        path (str):
            Path to Excel workbook
        chrom_num (str):
            Chromosome number, e.g. "2"
        chrom_arm (str):
            Chromosome arm, "p" or "q"
        contig_orientation (str):
            "+" or "-"
        contig_id (int):
            Selected contig ID
        target_site (int):
            Selected target/anchor contig site

    RETURNS:
        dict:
            Contains summary info, result dataframe, fused dataframe, and output filenames
    """
    sheet_name = f"{chrom_num}{chrom_arm}{contig_orientation}"
    dataset_label = sheet_name

    df = pd.read_excel(path, sheet_name=sheet_name)
    validate_required_columns(df, sheet_name)

    molecule_info = extract_molecule_info(df)

    label_qmap_avg_by_end, label_site_avg_by_end, gap_qmap_avg, gap_site_avg = finding_averages(
        df=df,
        molecule_info=molecule_info,
        contig_id=contig_id,
        chrom_arm=chrom_arm,
        contig_orientation=contig_orientation,
        target_site=target_site,
        tel_window=3
    )

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

    fused_df = build_fused_only_df(
        result_df=result_df,
        chrom_num=chrom_num,
        chrom_arm=chrom_arm,
        contig_id=contig_id
    )

    fused_txt = f"fused_only_{chrom_num}{chrom_arm}_contig{contig_id}_site{target_site}.txt"
    write_fused_only_txt(fused_df, fused_txt)

    return {
        "sheet_name": sheet_name,
        "dataset_label": dataset_label,
        "chrom_num": chrom_num,
        "chrom_arm": chrom_arm,
        "contig_orientation": contig_orientation,
        "contig_id": contig_id,
        "target_site": target_site,
        "categories": categories,
        "result_df": result_df,
        "fused_df": fused_df,
        "total_molecules": total_molecules,
        "summary_txt": summary_txt,
        "per_molecule_csv": per_molecule_csv,
        "fused_txt": fused_txt
    }


def collect_multiple_configurations():
    """
    Collect multiple run configurations from the user in the terminal.

    RETURNS:
        list[dict]:
            A list of run settings
    """
    configs = []

    while True:
        print("\nEnter a new run configuration:")
        chrom_num = input("Chromosome Number (e.g., 2, 4, 13): ").strip()
        chrom_arm = input("Chromosome Arm (p or q): ").strip().lower()
        contig_orientation = input("Contig Orientation (+ or -): ").strip()
        contig_id = int(input("Contig ID (integer): ").strip())
        target_site = int(input("Target Contig Site (integer): ").strip())

        configs.append({
            "chrom_num": chrom_num,
            "chrom_arm": chrom_arm,
            "contig_orientation": contig_orientation,
            "contig_id": contig_id,
            "target_site": target_site
        })

        add_more = input("Do you want to add another configuration? (y/n): ").strip().lower()
        if add_more != "y":
            break

    return configs


def main():
    """
    Main entry point.

    Workflow:
        1) collect Excel path once
        2) collect one or more run configurations
        3) run each configuration separately
        4) write the usual outputs for each run:
            - summary TXT
            - per-molecule CSV
            - fused-only TXT
        5) optionally combine all fused molecules from all runs into one TXT
    """
    path = input("Path to Data (.xlsx): ").strip()

    configs = collect_multiple_configurations()
    all_run_outputs = []
    combined_fused_dfs = []

    for i, cfg in enumerate(configs, start=1):
        print("\n==================================================")
        print(f"Running configuration {i} of {len(configs)}")
        print("==================================================")

        run_output = run_single_configuration(
            path=path,
            chrom_num=cfg["chrom_num"],
            chrom_arm=cfg["chrom_arm"],
            contig_orientation=cfg["contig_orientation"],
            contig_id=cfg["contig_id"],
            target_site=cfg["target_site"]
        )

        all_run_outputs.append(run_output)
        combined_fused_dfs.append(run_output["fused_df"])

        print("\n==================================================")
        print(f"Dataset: {run_output['dataset_label']}")
        print(f"Contig ID: {run_output['contig_id']}")
        print(f"Target Contig Site: {run_output['target_site']}")
        print(f"Total Molecules: {run_output['total_molecules']}")
        print("==================================================")

        print("\nClassification Summary")
        print_summary_table(run_output["categories"], run_output["total_molecules"])

        print("\n--------------------------------------------------")
        print("Per-Molecule Summary")
        print_per_molecule_terminal(run_output["result_df"])
        print("--------------------------------------------------")

        print("\nFiles written:")
        print(f"Summary TXT: {run_output['summary_txt']}")
        print(f"Per-molecule CSV: {run_output['per_molecule_csv']}")
        print(f"Fused-only TXT: {run_output['fused_txt']}")

    combine_all = input(
        "\nDo you want to combine all fused molecules from all runs into one TXT file? (y/n): "
    ).strip().lower()

    if combine_all == "y":
        if combined_fused_dfs:
            combined_fused_df = pd.concat(combined_fused_dfs, ignore_index=True)

            # Optional cleanup: remove duplicate rows if the same exact combination appears more than once
            combined_fused_df = combined_fused_df.drop_duplicates()

            combined_output_txt = "combined_fused_only_all_runs.txt"
            write_fused_only_txt(combined_fused_df, combined_output_txt)

            print(f"\nCombined fused-only TXT written: {combined_output_txt}")
        else:
            print("\nNo fused molecule data available to combine.")


if __name__ == "__main__":
    main()