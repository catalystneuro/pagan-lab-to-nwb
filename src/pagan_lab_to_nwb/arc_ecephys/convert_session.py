"""Convert one arc_ecephys session (behavior + spikes + dati + video) to NWB."""

import re
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import DirectoryPath, FilePath

from neuroconv.utils import dict_deep_update, load_dict_from_file
from pagan_lab_to_nwb.arc_ecephys import ArcEcephysNWBConverter

_RAT_INFO_PATH = Path(__file__).parent.parent / "arc_behavior" / "rat_information.xlsx"
_SEX_MAP = {"male": "M", "female": "F"}

# Filename patterns
# behavior:  data_@{protocol}_{experimenter}_{subject}_{date}.mat
# spikes:    spikes_@{protocol}_{experimenter}_{subject}_{date}.mat
# dati:      dati_{protocol}_{experimenter}_{subject}_{date}.mat   (no @)
# video:     video_@{protocol}_{experimenter}_{subject}_{date}.mp4
_BCONTROL_RE = re.compile(
    r"^(?:data_@)?(?P<protocol>[^_]+)_(?P<experimenter>[^_]+)_(?P<subject_id>[^_]+)_(?P<date_str>.+)$"
)


def _load_rat_info() -> pd.DataFrame:
    df = pd.read_excel(_RAT_INFO_PATH)
    df["Rat"] = df["Rat"].str.strip()
    return df.set_index("Rat")


def session_to_nwb(
    behavior_file_path: FilePath,
    nwb_folder_path: DirectoryPath,
    spikes_file_path: FilePath | None = None,
    dati_file_path: FilePath | None = None,
    video_file_path: FilePath | None = None,
    video_time_offset: float | None = None,
    task_params_file_path: FilePath | None = None,
    stub_test: bool = False,
    overwrite: bool = True,
) -> Path:
    """Convert one arc_ecephys session to NWB.

    Parameters
    ----------
    behavior_file_path :
        Path to the BControl ``data_@*.mat`` file.
    nwb_folder_path :
        Output directory. NWB file is written to
        ``nwb_folder_path/sub-{subject_id}/sub-{subject_id}_ses-{session_id}.nwb``.
    spikes_file_path :
        Path to the ``spikes_@*.mat`` spike-sorted file (optional).
    dati_file_path :
        Path to the ``dati_*.mat`` processed trial/neural data file (optional).
    video_file_path :
        Path to the ``video_@*.mp4`` file (optional).
    task_params_file_path :
        YAML file with task argument descriptions (optional).
    stub_test :
        If True, limit data volume for fast testing.
    overwrite :
        Overwrite existing NWB file.

    Returns
    -------
    Path
        Path to the written NWB file.
    """
    behavior_file_path = Path(behavior_file_path)
    file_name = behavior_file_path.stem.replace("data_@", "")
    match = _BCONTROL_RE.match(file_name)
    if not match:
        raise ValueError(f"Filename does not match expected pattern: '{behavior_file_path.name}'")

    protocol_name = match.group("protocol")
    subject_id = match.group("subject_id")
    date_str = match.group("date_str")

    session_id = "-".join([protocol_name.replace("_", "-"), date_str])

    nwb_folder_path = Path(nwb_folder_path)
    subject_folder = nwb_folder_path / f"sub-{subject_id.replace('_', '-')}"
    subject_folder.mkdir(parents=True, exist_ok=True)
    nwbfile_path = subject_folder / f"sub-{subject_id}_ses-{session_id}.nwb"

    if nwbfile_path.exists() and not overwrite:
        print(f"Skipping (exists): '{nwbfile_path}'")
        return nwbfile_path

    # ---- Build source_data (only include interfaces with available files) ----
    source_data = dict(Behavior=dict(file_path=behavior_file_path))
    conversion_options = dict(Behavior=dict(stub_test=stub_test))

    if task_params_file_path is not None:
        task_params_file_path = Path(task_params_file_path)
        if not task_params_file_path.exists():
            raise FileNotFoundError(f"YAML not found: '{task_params_file_path}'")
        arguments_metadata = load_dict_from_file(task_params_file_path)
        conversion_options["Behavior"]["arguments_metadata"] = arguments_metadata

    if spikes_file_path is not None:
        source_data["SpikeSorting"] = dict(file_path=Path(spikes_file_path))
        conversion_options["SpikeSorting"] = dict(stub_test=stub_test, protocol=protocol_name)

    if dati_file_path is not None:
        source_data["ProcessedTrials"] = dict(file_path=Path(dati_file_path))
        conversion_options["ProcessedTrials"] = dict(stub_test=stub_test)

    if video_file_path is not None:
        source_data["Video"] = dict(file_path=video_file_path)
        conversion_options["Video"] = dict(stub_test=stub_test, protocol=protocol_name, time_offset=video_time_offset)

    converter = ArcEcephysNWBConverter(source_data=source_data)

    # ---- Metadata ----
    metadata = converter.get_metadata()
    session_start_time = metadata["NWBFile"]["session_start_time"]
    session_start_time = session_start_time.replace(tzinfo=ZoneInfo("Europe/London"))
    metadata["NWBFile"].update(session_start_time=session_start_time)

    editable_metadata_path = Path(__file__).parent / "metadata.yaml"
    editable_metadata = load_dict_from_file(editable_metadata_path)
    metadata = dict_deep_update(metadata, editable_metadata)

    metadata["Subject"]["subject_id"] = subject_id
    metadata["NWBFile"]["session_id"] = session_id

    # Per-subject metadata from rat_information.xlsx
    try:
        rat_info = _load_rat_info()
        if subject_id in rat_info.index:
            row = rat_info.loc[subject_id]
            dob = row["Date of Birth"]
            metadata["Subject"]["date_of_birth"] = dob.to_pydatetime().replace(tzinfo=ZoneInfo("Europe/London"))
            sex_str = str(row["Sex"]).strip().lower()
            metadata["Subject"]["sex"] = _SEX_MAP.get(sex_str, "U")
        else:
            print(f"Warning: subject '{subject_id}' not found in rat_information.xlsx.")
    except Exception as e:
        print(f"Warning: could not load rat_information.xlsx: {e}")

    # ---- Run conversion ----
    converter.run_conversion(
        metadata=metadata,
        nwbfile_path=nwbfile_path,
        conversion_options=conversion_options,
        overwrite=overwrite,
    )

    print(f"Written: '{nwbfile_path}'")
    return nwbfile_path


if __name__ == "__main__":
    EPHYS_DIR = Path("/Volumes/T9/data/Ephys_data_examples")
    VIDEO_DIR = Path("/Volumes/T9/data/Video_data_examples")
    OUTPUT_DIR = Path("/Users/weian/data/arc_ecephys_nwb")

    # --- Ephys session ---
    # Note: the example data has a date mismatch (behavior=180110a, spikes/dati=181010a)
    # See open_questions.md Q6. Using available files for prototype testing.
    print("Converting ephys session (full)...")
    session_to_nwb(
        behavior_file_path=EPHYS_DIR / "data_@TaskSwitch4_Marino_P100_181010a.mat",
        spikes_file_path=EPHYS_DIR / "spikes_@TaskSwitch4_Marino_P100_181010a.mat",
        dati_file_path=EPHYS_DIR / "dati_TaskSwitch4_Marino_P100_181010a.mat",
        nwb_folder_path=OUTPUT_DIR,
        stub_test=False,
        overwrite=True,
    )
    print("Ephys session done.")

    # --- Video session ---
    print("Converting video session (full)...")
    session_to_nwb(
        behavior_file_path=VIDEO_DIR / "data_@TaskSwitch6_Marino_P267_221211a.mat",
        video_file_path=VIDEO_DIR / "video_@TaskSwitch6_Marino_P267_221211a.mp4",
        nwb_folder_path=OUTPUT_DIR,
        stub_test=False,
        overwrite=True,
    )
    print("Video session done.")
