"""Primary script to run to convert an entire session for of data using the NWBConverter."""

from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import DirectoryPath, FilePath

from neuroconv.utils import dict_deep_update, load_dict_from_file
from pagan_lab_to_nwb.arc_behavior import ArcBehaviorNWBConverter


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
    nwb_folder_path = Path(nwb_folder_path)
    nwb_folder_path.mkdir(parents=True, exist_ok=True)

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

    # Add datetime to conversion
    metadata = converter.get_metadata()
    session_start_time = metadata["NWBFile"]["session_start_time"]
    session_start_time = session_start_time.replace(tzinfo=ZoneInfo("Europe/London"))
    metadata["NWBFile"].update(session_start_time=session_start_time)

    # Update default metadata with the editable in the corresponding yaml file
    editable_metadata_path = Path(__file__).parent / "metadata.yaml"
    editable_metadata = load_dict_from_file(editable_metadata_path)
    metadata = dict_deep_update(metadata, editable_metadata)

    file_path = Path(file_path)
    file_name = file_path.stem  # data_@TaskSwitch6_Nuria_H7015_250516a
    # extract data_@{protocol_name}_{experimenter}_{subject_id}_{session_id} pattern from file name
    file_name = file_name.replace("data_@", "")  # Remove 'data_@' prefix
    protocol_name, experimenter, subject_id, session_id = file_name.split("_")

    metadata["Subject"]["subject_id"] = subject_id
    metadata["NWBFile"]["session_id"] = session_id

    nwbfile_path = nwb_folder_path / f"sub-{subject_id}_ses-{session_id}.nwb"

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
    nwb_folder_path = "/Volumes/T9/data/Pagan/raw"

    # Path to the YAML file containing task parameters and their descriptions
    # This file should be generated from the MATLAB code files in the Protocol_code folder
    # See utils/notes.md for instructions on how to generate this file
    yaml_file_path = Path('/Users/weian/data/Pagan/Protocol "TaskSwitch6"/Protocol_code') / "task_switch6_params.yaml"

    stub_test = False
    overwrite = True

    session_to_nwb(
        file_path=behavior_file_path,
        nwb_folder_path=nwb_folder_path,
        task_params_file_path=yaml_file_path,
        stub_test=stub_test,
        overwrite=overwrite,
    )
