"""Insert a converted arc_ecephys NWB file into the Spyglass database.

Usage
-----
Run from the arc_ecephys directory (so dj_local_conf.json is found):

    cd src/pagan_lab_to_nwb/arc_ecephys
    python insert_session.py

Or call insert_session() directly from a notebook.

Import order matters: dj.config.load() MUST be called before any spyglass
import, because spyglass executes dj.schema() at module load time.

BControl tables
---------------
All arc_ecephys sessions include BControl behavior data.  After the core
Spyglass insertion this script also populates TaskRecordingTypes and
TaskRecording (from spyglass.common.common_task_rec), following the same
pattern used by arc_behavior/insert_session.py.

Spike sorting tables
--------------------
If the NWB file contains a units table (spike-sorted data from a spikes.mat
file), sgi.insert_sessions() automatically populates ImportedSpikeSorting and
SpikeSortingOutput.  insert_sorted_spikes() then adds SortedSpikesGroup (an
"all_units" group) and UnitAnnotation entries for trode_id (the tetrode each
unit was sorted from).
"""

from pathlib import Path

import datajoint as dj
import pandas as pd
from numpy.testing import assert_array_equal

_CONF_PATH = Path(__file__).parent / "dj_local_conf.json"
dj.config.load(str(_CONF_PATH))  # ← MUST be before spyglass imports
dj.conn(use_tls=False)

import spyglass.common as sgc
import spyglass.data_import as sgi
import spyglass.spikesorting.v1 as sgs
from spyglass.common import Nwbfile
from spyglass.common.common_task_rec import TaskRecording, TaskRecordingTypes
from spyglass.settings import raw_dir
from spyglass.spikesorting.analysis.v1.group import (
    SortedSpikesGroup,
    UnitSelectionParams,
)
from spyglass.spikesorting.analysis.v1.unit_annotation import UnitAnnotation
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename, get_nwb_file

from pagan_lab_to_nwb.spyglass_extensions.spyglass_processed_trials_table import (
    ProcessedTrials,
)

SPYGLASS_RAW_DIR = Path(raw_dir)


# ---------------------------------------------------------------------------
# Lookup table seeding (arc_ecephys-specific: probes, DataAcqDevice, camera)
# ---------------------------------------------------------------------------


def seed_lookup_tables():
    """Pre-insert all required Spyglass lookup table entries.

    Spyglass prompts interactively when it encounters an unknown device type
    or probe type. Pre-inserting these entries makes insertion non-interactive
    and reproducible.

    These values must match what is written in the NWB files by the interfaces.
    If device metadata changes, update metadata.yaml AND re-seed here.
    """
    sgc.DataAcquisitionDeviceSystem.insert1(
        {"data_acquisition_device_system": "SpikeGadgets"},
        skip_duplicates=True,
    )
    sgc.DataAcquisitionDeviceAmplifier.insert1(
        {"data_acquisition_device_amplifier": "Horizontal Headstage 128-Channel Datalogger"},
        skip_duplicates=True,
    )
    sgc.DataAcquisitionDevice.insert1(
        {
            "data_acquisition_device_name": "HH128",
            "data_acquisition_device_system": "SpikeGadgets",
            "data_acquisition_device_amplifier": "Horizontal Headstage 128-Channel Datalogger",
            "adc_circuit": "Horizontal Headstage 128-Channel Datalogger",
        },
        skip_duplicates=True,
    )
    # manufacturer='' matches what ndx-franklab-novela Probe writes when no
    # manufacturer is passed. Must match exactly or insertion raises PopulateException.
    sgc.ProbeType.insert1(
        {
            "probe_type": "tetrode",
            "probe_description": "32-tetrode array (SpikeGadgets, Pagan Lab)",
            "manufacturer": "",
            "num_shanks": 1,
        },
        skip_duplicates=True,
    )
    sgc.CameraDevice.insert1(
        {
            "camera_name": "top_camera",
            "meters_per_pixel": 0.001,
            "manufacturer": "unknown",
            "model": "unknown",
            "lens": "unknown",
            "camera_id": 1,
        },
        skip_duplicates=True,
    )


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def clean_db_entry(nwb_file_name: str):
    """Remove all Spyglass entries for a given NWB file.

    Use before re-inserting a session to avoid duplicate-key errors.
    Uses the copy filename (stem + '_') which is what Spyglass stores.
    """
    copy_file_name = get_nwb_copy_filename(nwb_file_name)
    nwb_dict = {"nwb_file_name": copy_file_name}

    # Delete Nwbfile — cascades to Session and all its descendants
    (sgc.Nwbfile & nwb_dict).delete(safemode=False)

    # Delete non-cascading lookup entries tied to this session
    probe_ids = (sgc.ElectrodeGroup & nwb_dict).fetch("probe_id", as_dict=True)
    (sgc.Probe & probe_ids).delete(safemode=False)
    (sgc.IntervalList & nwb_dict).delete(safemode=False)
    camera_names = (sgc.VideoFile & nwb_dict).fetch("camera_name", as_dict=True)
    (sgc.CameraDevice & camera_names).delete(safemode=False)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def print_tables(nwbfile_path: Path):
    """Print key Spyglass tables for a given session to tables.txt."""
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    nwb_dict = dict(nwb_file_name=nwb_copy_file_name)

    out_path = Path(__file__).parent / "tables.txt"
    with open(out_path, "w") as f:
        print("=== TaskRecordingTypes.StateTypes ===", file=f)
        state_types = (TaskRecordingTypes.StateTypes() & nwb_dict).fetch(as_dict=True)
        print(pd.DataFrame(state_types).set_index("id").to_markdown(), file=f)

        print("\n=== TaskRecordingTypes.EventTypes ===", file=f)
        event_types = (TaskRecordingTypes.EventTypes() & nwb_dict).fetch(as_dict=True)
        print(pd.DataFrame(event_types).set_index("id").to_markdown(), file=f)

        print("\n=== TaskRecordingTypes.ActionTypes ===", file=f)
        action_types = (TaskRecordingTypes.ActionTypes() & nwb_dict).fetch(as_dict=True)
        print(pd.DataFrame(action_types).set_index("id").to_markdown(), file=f)

        print("\n=== TaskRecordingTypes.Arguments ===", file=f)
        arguments = (TaskRecordingTypes.Arguments() & nwb_dict).fetch(as_dict=True)
        print(pd.DataFrame(arguments).to_markdown(), file=f)

        n_pt = len(ProcessedTrials() & nwb_dict)
        print(f"\n=== ProcessedTrials ({n_pt} rows) ===", file=f)
        if n_pt > 0:
            pt_rows = (ProcessedTrials() & nwb_dict).fetch(as_dict=True, limit=5)
            print(pd.DataFrame(pt_rows).to_markdown(), file=f)
            if n_pt > 5:
                print(f"  ... ({n_pt - 5} more rows)", file=f)
        else:
            print("  (no processed trials data for this session)", file=f)

    print(f"Table summary written to {out_path}")


def test_task_recording_types(nwbfile_path: Path):
    """Assert that TaskRecordingTypes matches the source NWB file."""
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    nwb_dict = dict(nwb_file_name=nwb_copy_file_name)

    nwbfile = get_nwb_file(str(nwbfile_path))
    task = nwbfile.lab_meta_data["task"]

    sgc_state_names = (TaskRecordingTypes.StateTypes() & nwb_dict).fetch("state_name")
    nwb_state_names = task.state_types["state_name"][:]
    assert_array_equal(nwb_state_names, sgc_state_names)

    sgc_event_names = (TaskRecordingTypes.EventTypes() & nwb_dict).fetch("event_name")
    nwb_event_names = task.event_types["event_name"][:]
    assert_array_equal(nwb_event_names, sgc_event_names)

    sgc_action_names = (TaskRecordingTypes.ActionTypes() & nwb_dict).fetch("action_name")
    nwb_action_names = task.action_types["action_name"][:]
    assert_array_equal(nwb_action_names, sgc_action_names)

    sgc_argument_names = (TaskRecordingTypes.Arguments() & nwb_dict).fetch("argument_name")
    nwb_argument_names = task.task_arguments["argument_name"][:]
    assert set(nwb_argument_names) == set(sgc_argument_names)

    print(f"test_task_recording_types passed for {nwbfile_path.name}")


def test_processed_trials(nwbfile_path: Path) -> None:
    """Assert ProcessedTrials is populated and fetch1_dataframe() returns the correct row count.

    Note: object_id comparison is intentionally skipped — Spyglass copies the NWB file
    with a ``_`` suffix and all object_ids are regenerated in the copy.
    """
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    nwb_dict = dict(nwb_file_name=nwb_copy_file_name)

    # Check against the NWB copy in the Spyglass raw dir (not the source file)
    nwb_copy_path = SPYGLASS_RAW_DIR / nwb_copy_file_name
    nwbf_copy = get_nwb_file(str(nwb_copy_path))
    try:
        pt_copy = nwbf_copy.processing["behavior"]["processed_trials"]
    except KeyError:
        print(f"test_processed_trials skipped (no dati data): {nwbfile_path.name}")
        return

    assert ProcessedTrials() & nwb_dict, f"ProcessedTrials has no entry for {nwb_copy_file_name}"

    # Verify fetch1_dataframe() resolves and returns the right trial count
    df = (ProcessedTrials() & nwb_dict).fetch1_dataframe()
    n_db = len(df)
    n_nwb = len(pt_copy)
    assert n_db == n_nwb, f"ProcessedTrials row count mismatch: fetch={n_db}, NWB copy={n_nwb}"

    print(f"test_processed_trials passed: {n_db} trials for {nwbfile_path.name}")


# ---------------------------------------------------------------------------
# Spike sorting tables
# ---------------------------------------------------------------------------


def insert_sorted_spikes(nwb_copy_file_name: str) -> bool:
    """Populate SortedSpikesGroup and UnitAnnotation for a session with units.

    sgi.insert_sessions() already populates ImportedSpikeSorting and
    SpikeSortingOutput when the NWB file has a units table.  This function
    adds the analysis-level grouping (SortedSpikesGroup) and annotates each
    unit with its tetrode ID (trode_id).

    Parameters
    ----------
    nwb_copy_file_name :
        The Spyglass copy filename (with trailing ``_``), e.g.
        ``sub-P100_ses-TaskSwitch4-180110a_.nwb``.

    Returns
    -------
    bool
        True if spike data was found and inserted, False if no units table.
    """
    nwb_dict = dict(nwb_file_name=nwb_copy_file_name)

    # Check that ImportedSpikeSorting was populated by insert_sessions()
    if not (sgs.ImportedSpikeSorting() & nwb_dict):
        print(f"  No ImportedSpikeSorting entry for {nwb_copy_file_name} — skipping spike tables.")
        return False

    merge_id = str((SpikeSortingOutput.ImportedSpikeSorting() & nwb_dict).fetch1("merge_id"))

    # SortedSpikesGroup — one group containing all sorted units
    UnitSelectionParams().insert_default()
    group_name = "all_units"
    if not (SortedSpikesGroup() & {**nwb_dict, "sorted_spikes_group_name": group_name}):
        SortedSpikesGroup().create_group(
            group_name=group_name,
            nwb_file_name=nwb_copy_file_name,
            keys=[{"spikesorting_merge_id": merge_id}],
        )

    group_key = (SortedSpikesGroup() & {**nwb_dict, "sorted_spikes_group_name": group_name}).fetch1("KEY")

    # UnitAnnotation — annotate each unit with its tetrode ID
    _, unit_ids = SortedSpikesGroup().fetch_spike_data(group_key, return_unit_ids=True)

    # Read trode_id directly from the NWB copy file (fetch_nwb() returns only
    # metadata keys for ImportedSpikeSorting, not the units table itself).
    import pynwb

    nwb_copy_path = SPYGLASS_RAW_DIR / nwb_copy_file_name
    with pynwb.NWBHDF5IO(str(nwb_copy_path), "r", load_namespaces=True) as io:
        nwbf = io.read()
        trode_ids = nwbf.units["trode_id"][:]

    n_annotated = 0
    for unit_key in unit_ids:
        unit_id = unit_key["unit_id"]
        annotation_key = {
            **unit_key,
            "annotation": "trode_id",
            "quantification": float(trode_ids[unit_id]),
        }
        UnitAnnotation().add_annotation(annotation_key, skip_duplicates=True)
        n_annotated += 1

    print(f"  SortedSpikesGroup '{group_name}' created with {n_annotated} units annotated.")
    return True


def test_sorted_spikes(nwb_copy_file_name: str) -> None:
    """Assert that SortedSpikesGroup spike times match the NWB units table."""
    import numpy as np
    import pynwb

    nwb_dict = dict(nwb_file_name=nwb_copy_file_name)
    group_key = (SortedSpikesGroup() & {**nwb_dict, "sorted_spikes_group_name": "all_units"}).fetch1("KEY")

    spikes_spyglass = SortedSpikesGroup().fetch_spike_data(group_key)

    nwb_copy_path = SPYGLASS_RAW_DIR / nwb_copy_file_name
    with pynwb.NWBHDF5IO(str(nwb_copy_path), "r") as io:
        nwbf = io.read()
        spikes_nwb = [nwbf.units["spike_times"][i] for i in range(len(nwbf.units))]

    for i, (nwb_unit, sglass_unit) in enumerate(zip(spikes_nwb, spikes_spyglass)):
        np.testing.assert_array_equal(nwb_unit, sglass_unit, err_msg=f"Mismatch at unit {i}")

    print(f"test_sorted_spikes passed for {nwb_copy_file_name} ({len(spikes_nwb)} units)")


# ---------------------------------------------------------------------------
# Main insertion
# ---------------------------------------------------------------------------


def insert_session(
    nwbfile_path: Path,
    rollback_on_fail: bool = True,
    raise_err: bool = True,
    clean_existing: bool = False,
):
    """Insert one arc_ecephys NWB session into the Spyglass database.

    Parameters
    ----------
    nwbfile_path :
        Full path to the NWB file — must already be in SPYGLASS_RAW_DIR.
    rollback_on_fail :
        Roll back the transaction on error (passed to insert_sessions).
    raise_err :
        Re-raise errors from insert_sessions.
    clean_existing :
        If True, delete any existing entries for this session before inserting.
        Set to True when re-inserting after a failed or partial run.
    """
    nwbfile_path = Path(nwbfile_path)
    if not nwbfile_path.exists():
        raise FileNotFoundError(f"NWB file not found: {nwbfile_path}")

    seed_lookup_tables()

    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    nwb_dict = dict(nwb_file_name=nwb_copy_file_name)

    if clean_existing:
        print(f"Cleaning existing entries for {nwbfile_path.name} ...")
        sgc_nwbfile = Nwbfile() & nwb_dict
        if sgc_nwbfile:
            sgc_nwbfile.delete(safemode=False)

    # Core Spyglass tables (Session, Subject, Electrode, Probe, VideoFile, etc.)
    print(f"Inserting: {nwbfile_path.name}")
    sgi.insert_sessions(str(nwbfile_path), rollback_on_fail=rollback_on_fail, raise_err=raise_err)

    # insert_sessions() silently skips if Nwbfile entry already exists (warns + continues).
    # Verify Session was actually populated to catch silent skips.
    if not (sgc.Session() & nwb_dict):
        raise RuntimeError(
            f"Session was not populated for {nwb_copy_file_name}. "
            "A Nwbfile entry may exist from a previous partial run. "
            "Re-run with clean_existing=True."
        )
    print("Core insertion complete.")

    # BControl tables (ndx-structured-behavior task recording)
    nwbfile = get_nwb_file(str(SPYGLASS_RAW_DIR / nwb_copy_file_name))

    rec_types = TaskRecordingTypes()
    if not rec_types & nwb_dict:
        rec_types.insert_from_nwbfile(nwb_copy_file_name, nwbfile)

    task_rec = TaskRecording()
    if not task_rec & nwb_dict:
        task_rec.insert_from_nwbfile(nwb_copy_file_name, nwbfile)

    print("BControl tables (TaskRecordingTypes, TaskRecording) populated.")

    # Dati processed_trials table (only for sessions with a dati file)
    if not (ProcessedTrials() & nwb_dict):
        ProcessedTrials().insert_from_nwbfile(nwb_copy_file_name, nwbfile)
    else:
        print("  ProcessedTrials: already populated, skipping.")

    # Spike sorting tables (only for sessions with a units table)
    insert_sorted_spikes(nwb_copy_file_name)

    print_tables(nwbfile_path)


if __name__ == "__main__":
    for nwb_file_name in [
        "sub-P100_ses-TaskSwitch4-181010a.nwb",
        "sub-P267_ses-TaskSwitch6-221211a.nwb",
    ]:
        nwbfile_path = SPYGLASS_RAW_DIR / nwb_file_name
        insert_session(nwbfile_path=nwbfile_path, clean_existing=True)
        test_task_recording_types(nwbfile_path=nwbfile_path)
        test_processed_trials(nwbfile_path=nwbfile_path)

    # Test spike sorting for the session that has units
    ephys_copy = get_nwb_copy_filename("sub-P100_ses-TaskSwitch4-181010a.nwb")
    test_sorted_spikes(ephys_copy)
