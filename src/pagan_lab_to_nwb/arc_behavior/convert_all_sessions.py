"""Convert all BControl .mat files in a directory to NWB."""

import traceback
from pathlib import Path
from typing import List

from pydantic import DirectoryPath, FilePath
from tqdm import tqdm

from pagan_lab_to_nwb.arc_behavior.convert_session import session_to_nwb


def dataset_to_nwb(
    *,
    data_dir_path: DirectoryPath,
    output_dir_path: DirectoryPath,
    task_params_file_path: FilePath,
    glob_pattern: str = "data_@*.mat",
    stub_test: bool = False,
    overwrite: bool = True,
    error_log_filename: str = "conversion_errors.txt",
) -> List[FilePath]:
    """
    Convert all BControl .mat files in a directory to NWB sequentially.

    Parameters
    ----------
    data_dir_path : Path
        Directory containing the BControl .mat files (and optionally Protocol_code with YAML).
    output_dir_path : Path
        Directory where NWB files will be saved.
    task_params_file_path : Path
        Path to YAML with task argument descriptions.
    glob_pattern : str
        Glob pattern to find .mat files (default 'data_@*.mat').
    stub_test : bool
        If True, run in stub mode passed to session conversion.
    overwrite : bool
        Overwrite existing NWB files.
    error_log_filename : str
        Name of the aggregated error log file (created inside output_dir_path).

    Returns
    -------
    List[FilePath]
        List of paths to the successfully converted NWB files.
    """
    data_dir_path = Path(data_dir_path)
    output_dir_path = Path(output_dir_path)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    file_paths = sorted(data_dir_path.glob(glob_pattern))
    if not file_paths:
        print(f"No files matched pattern '{glob_pattern}' in '{data_dir_path}'. Nothing to convert.")
        return []

    error_log_path = output_dir_path / error_log_filename
    if error_log_path.exists():
        error_log_path.unlink()  # start fresh

    nwb_file_paths = []
    for mat_file in tqdm(file_paths, desc="Converting sessions"):
        try:
            nwb_file_path = session_to_nwb(
                file_path=mat_file,
                nwb_folder_path=output_dir_path,
                task_params_file_path=task_params_file_path,
                stub_test=stub_test,
                overwrite=overwrite,
            )
            nwb_file_paths.append(nwb_file_path)
        except Exception:
            with error_log_path.open("a") as f:
                f.write(f"\n--- Error converting: {mat_file.name} ---\n")
                f.write(traceback.format_exc())
    num_failures = len(file_paths) - len(nwb_file_paths)
    print(f"\nConversion complete. Success: {len(nwb_file_paths)} | Failed: {len(file_paths) - len(nwb_file_paths)}")
    if num_failures:
        print(f"See error log: {error_log_path}")
    return nwb_file_paths


if __name__ == "__main__":
    # Example usage (adjust paths):
    # Directory containing the BControl .mat files and Protocol_code folder
    data_dir = Path('/Users/weian/data/Pagan/Protocol "TaskSwitch6"')
    # Directory where the NWB files will be saved
    output_dir = Path("/Volumes/T9/data/Pagan/raw")

    # Path to the YAML file containing task parameters and their descriptions
    # This file should be generated from the MATLAB code files in the Protocol_code folder
    # See src/pagan_lab_to_nwb/arc_behavior/utils/notes.md for instructions on how to generate this file
    task_yaml = data_dir / "Protocol_code" / "task_switch6_params.yaml"

    dataset_to_nwb(
        data_dir_path=data_dir,
        output_dir_path=output_dir,
        task_params_file_path=task_yaml,
        glob_pattern="data_@TaskSwitch6*.mat",
        stub_test=False,
        overwrite=True,
    )
