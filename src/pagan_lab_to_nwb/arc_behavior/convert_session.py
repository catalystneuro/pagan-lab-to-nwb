"""Primary script to run to convert an entire session for of data using the NWBConverter."""

import re
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import DirectoryPath, FilePath

from neuroconv.utils import dict_deep_update, load_dict_from_file
from pagan_lab_to_nwb.arc_behavior import ArcBehaviorNWBConverter

_RAT_INFO_PATH = Path(__file__).parent / "rat_information.xlsx"
_TASK_PARAMS_YAML_PATH = Path(__file__).parent / "task_switch6_params.yaml"
_SEX_MAP = {"male": "M", "female": "F"}


def _load_rat_info() -> pd.DataFrame:
    df = pd.read_excel(_RAT_INFO_PATH)
    df["Rat"] = df["Rat"].str.strip()
    return df.set_index("Rat")


def session_to_nwb(
    file_path: FilePath,
    nwb_folder_path: DirectoryPath,
    task_params_file_path: FilePath = None,
    stub_test: bool = False,
    overwrite: bool = True,
) -> FilePath:
    """
    Convert a session of BControl data to NWB format.

    Parameters
    ----------
    file_path : FilePath
        Path to the BControl behavior data file (e.g., .mat file).
    nwb_folder_path : DirectoryPath
        Path to the directory where the NWB file will be saved.
    task_params_file_path : FilePath, optional
        Path to the YAML file containing task parameters and their descriptions.
        If None, uses 'no description' for all task parameter descriptions.
    stub_test : bool, optional
        If True, runs a stub test without full conversion. Default is False.
    overwrite : bool, optional
        If True, overwrites the existing NWB file if it exists. Default is True.

    Returns
    -------
    FilePath
        Path to the converted NWB file.
    """
    file_path = Path(file_path)
    file_name = file_path.stem  # data_@TaskSwitch6_Nuria_H7015_250516a
    # extract data_@{protocol_name}_{experimenter}_{subject_id}_{session_id} pattern from file name
    file_name = file_name.replace("data_@", "")  # Remove 'data_@' prefix
    pattern = re.compile(
        r"^(?:data_@)?(?P<protocol>[^_]+)_(?P<experimenter>[^_]+)_(?P<subject_id>[^_]+)_(?P<date_str>.+)$"
    )
    match = pattern.match(file_name)
    if not match:
        raise ValueError(
            f"Filename does not match expected pattern (e.g. 'data_@TaskSwitch6_Nuria_H7015_250516a'): '{file_name}'"
        )

    protocol_name = match.group("protocol")
    experimenter = match.group("experimenter")
    subject_id = match.group("subject_id")
    date_str = match.group("date_str")

    # protocol_name and date_str are used to create session_id
    # DANDI requires no underscores inside the session label; use hyphens throughout
    session_id = "-".join([protocol_name.replace("_", "-"), date_str])

    nwb_folder_path = Path(nwb_folder_path)
    subject_folder = nwb_folder_path / f"sub-{subject_id.replace('_', '-')}"
    subject_folder.mkdir(parents=True, exist_ok=True)
    nwbfile_path = subject_folder / f"sub-{subject_id}_ses-{session_id}.nwb"

    if nwbfile_path.exists() and not overwrite:
        print(f"NWB file already exists and overwrite is False. Skipping conversion: '{nwbfile_path}'")
        return nwbfile_path

    source_data = dict()
    conversion_options = dict()

    # Add Behavior
    source_data.update(dict(Behavior=dict(file_path=file_path)))
    conversion_options.update(dict(Behavior=dict(stub_test=stub_test)))
    # Add task parameters from YAML file
    if task_params_file_path is not None:
        task_params_file_path = Path(task_params_file_path)
        if not task_params_file_path.exists():
            raise FileNotFoundError(f"YAML file not found: '{task_params_file_path}'. Please provide a valid path.")
        arguments_metadata = load_dict_from_file(task_params_file_path)
        conversion_options["Behavior"].update(
            arguments_metadata=arguments_metadata,
        )

    converter = ArcBehaviorNWBConverter(source_data=source_data)

    # Load the editable metadata YAML first so timezone and other session-level
    # values are available before they are needed.
    editable_metadata_path = Path(__file__).parent / "metadata.yaml"
    editable_metadata = load_dict_from_file(editable_metadata_path)
    timezone_str = editable_metadata.get("Session", {}).get("timezone", "Europe/London")

    # Add datetime to conversion
    metadata = converter.get_metadata()
    session_start_time = metadata["NWBFile"]["session_start_time"]
    session_start_time = session_start_time.replace(tzinfo=ZoneInfo(timezone_str))
    metadata["NWBFile"].update(session_start_time=session_start_time)

    # Merge YAML metadata over interface defaults.
    # Optogenetics is extracted first and set directly: dict_deep_update treats duplicate
    # values inside lists as a set union and collapses them (e.g. [473.0, 473.0] → [473.0]).
    opto_meta = editable_metadata.pop("Optogenetics", None)
    metadata = dict_deep_update(metadata, editable_metadata)
    if opto_meta is not None:
        metadata["Optogenetics"] = opto_meta

    metadata["Subject"]["subject_id"] = subject_id
    metadata["NWBFile"]["session_id"] = session_id

    # Add per-subject metadata from rat_information.xlsx
    rat_info = _load_rat_info()
    if subject_id in rat_info.index:
        row = rat_info.loc[subject_id]
        dob = row["Date of Birth"]
        metadata["Subject"]["date_of_birth"] = dob.to_pydatetime().replace(tzinfo=ZoneInfo(timezone_str))
        sex_str = str(row["Sex"]).strip().lower()
        metadata["Subject"]["sex"] = _SEX_MAP.get(sex_str, "U")
    else:
        print(
            f"Warning: subject '{subject_id}' not found in rat_information.xlsx — subject metadata will be incomplete."
        )

    # Run conversion
    converter.run_conversion(
        metadata=metadata,
        nwbfile_path=nwbfile_path,
        conversion_options=conversion_options,
        overwrite=overwrite,
    )

    return nwbfile_path


if __name__ == "__main__":

    # Parameters for conversion
    behavior_file_path = '/Users/weian/data/Pagan/Protocol "TaskSwitch6"/data_@TaskSwitch6_Nuria_H7015_250516a.mat'
    nwb_folder_path = "/Volumes/T9/data/Pagan_latest_data_share/nwbfiles"

    # Path to the YAML file containing task parameters and their descriptions
    # The canonical copy lives in the conversion repo at arc_behavior/task_switch6_params.yaml
    yaml_file_path = _TASK_PARAMS_YAML_PATH

    stub_test = False
    overwrite = True

    session_to_nwb(
        file_path=behavior_file_path,
        nwb_folder_path=nwb_folder_path,
        task_params_file_path=yaml_file_path,
        stub_test=stub_test,
        overwrite=overwrite,
    )
