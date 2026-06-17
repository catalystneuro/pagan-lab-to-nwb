"""Insert a converted arc_behavior NWB file into the Spyglass database.

Usage
-----
Run directly:

    python src/pagan_lab_to_nwb/arc_behavior/insert_session.py

Or call insert_session() directly from a notebook.

Import order matters: dj.config.load() MUST be called before any spyglass
import, because spyglass executes dj.schema() at module load time.

BControl tables
---------------
Arc_behavior sessions contain BControl behavior data only (no ephys).
After the core Spyglass insertion this script also populates
TaskRecordingTypes and TaskRecording (from spyglass.common.common_task_rec).

Spyglass compatibility shim
---------------------------
Spyglass stores experiment_description in a VARCHAR(2000) column and
argument_description in a VARCHAR(255) column.  Our NWB files can exceed
these limits (rich YAML descriptions, experiment metadata).  Rather than
truncating the source NWB files or patching Spyglass, the function
`patch_for_spyglass()` does temporary data surgery on the copy that lands
in SPYGLASS_RAW_DIR, truncating only those two fields in-place via h5py.
The original file (and therefore the DANDI copy) is never touched.

This shim should be removed once Spyglass upstream increases its column sizes.
"""

from pathlib import Path

import datajoint as dj
import h5py
import pandas as pd
from numpy.testing import assert_array_equal

_CONF_PATH = Path(__file__).parent.parent / "arc_ecephys" / "dj_local_conf.json"
dj.config.load(str(_CONF_PATH))  # ← MUST be before spyglass imports
dj.conn(use_tls=False)

import numpy as np
import spyglass.common as sgc
import spyglass.data_import as sgi
from spyglass.common import Nwbfile
from spyglass.common.common_interval import IntervalList
from spyglass.common.common_optogenetics import (
    OpticalFiberDevice,
    OpticalFiberImplant,
    OptogeneticProtocol,
    Virus,
    VirusInjection,
)
from spyglass.common.common_task import Task, TaskEpoch
from spyglass.common.common_task_rec import TaskRecording, TaskRecordingTypes
from spyglass.settings import raw_dir
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename, get_nwb_file

SPYGLASS_RAW_DIR = Path(raw_dir)

# Spyglass column limits that our NWB content can exceed.
_EXPERIMENT_DESCRIPTION_LIMIT = 2000  # common_session.Session VARCHAR(2000)
_ARGUMENT_DESCRIPTION_LIMIT = 255  # common_task_rec.TaskRecordingTypes.Arguments VARCHAR(255)
_EXPRESSION_LIMIT = 2000  # common_task_rec.TaskRecordingTypes.Arguments VARCHAR(2000)


# ---------------------------------------------------------------------------
# Spyglass compatibility shim (h5py surgery on the raw-dir copy)
# ---------------------------------------------------------------------------


def _h5_truncate_scalar(f: h5py.File, path: str, max_len: int) -> None:
    """Truncate a scalar string HDF5 dataset to max_len characters, in-place."""
    if path not in f:
        return
    ds = f[path]
    val = ds[()]
    if isinstance(val, bytes):
        val = val.decode()
    if len(val) <= max_len:
        return
    print(f"  [spyglass-patch] {path}: {len(val)} → {max_len} chars")
    attrs = dict(ds.attrs)  # preserve HDMF/NWB type attributes
    del f[path]
    new_ds = f.create_dataset(path, data=val[:max_len], dtype=h5py.string_dtype())
    for k, v in attrs.items():
        new_ds.attrs[k] = v


def _h5_truncate_string_array(f: h5py.File, path: str, max_len: int) -> None:
    """Truncate a 1-D variable-length string HDF5 dataset, in-place."""
    if path not in f:
        return
    ds = f[path]
    raw = ds[()]  # numpy array of bytes objects
    truncated = []
    n_patched = 0
    for v in raw:
        s = v.decode() if isinstance(v, bytes) else v
        if len(s) > max_len:
            truncated.append(s[:max_len])
            n_patched += 1
        else:
            truncated.append(s)
    if n_patched == 0:
        return
    print(f"  [spyglass-patch] {path}: truncated {n_patched} entries to {max_len} chars")
    attrs = dict(ds.attrs)  # preserve HDMF/NWB type attributes
    del f[path]
    new_ds = f.create_dataset(
        path,
        data=[s.encode() for s in truncated],
        dtype=h5py.string_dtype(),
    )
    for k, v in attrs.items():
        new_ds.attrs[k] = v


def patch_for_spyglass(nwbfile_path: Path) -> None:
    """Truncate string fields in-place so they fit Spyglass column limits.

    Call this on the file in SPYGLASS_RAW_DIR *before* calling insert_sessions().
    The original source file (and any DANDI copy) is never modified.

    Fields patched
    --------------
    - ``general/experiment_description`` → VARCHAR(2000) limit
    - ``general/task/task_arguments/argument_description`` → VARCHAR(255) limit
    - ``general/task/task_arguments/expression`` → VARCHAR(2000) limit
      (BControl protocols embed full FSM code here; e.g. SessionDefinition_training_stages
      can be 40,000+ characters)
    """
    nwbfile_path = Path(nwbfile_path)
    with h5py.File(nwbfile_path, "a") as f:
        _h5_truncate_scalar(f, "general/experiment_description", _EXPERIMENT_DESCRIPTION_LIMIT)
        _h5_truncate_string_array(
            f,
            "general/task/task_arguments/argument_description",
            _ARGUMENT_DESCRIPTION_LIMIT,
        )
        _h5_truncate_string_array(
            f,
            "general/task/task_arguments/expression",
            _EXPRESSION_LIMIT,
        )


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


def test_optogenetics(nwbfile_path: Path):
    """Assert optogenetics tables were populated correctly for opto sessions.

    No-op for sessions without ``opto_epochs`` (non-opto sessions).

    Checks:
    - Hardware tables populated by ``insert_sessions()``: ``Virus`` (1 row),
      ``VirusInjection`` (2 rows — bilateral), ``OpticalFiberDevice`` (1 row),
      ``OpticalFiberImplant`` (2 rows — bilateral).
    - Post-hoc tables inserted by ``insert_optogenetics()``: ``IntervalList``,
      ``TaskEpoch`` (epoch=1), ``OptogeneticProtocol`` (epoch=1).
    - ``OptogeneticProtocol`` values round-trip against the source NWB file:
      ``pulse_length``, ``stimulus_power``, ``stimulus_object_id``.
    """
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    nwb_dict = dict(nwb_file_name=nwb_copy_file_name)

    nwbfile = get_nwb_file(str(nwbfile_path))
    opto_epochs = nwbfile.intervals.get("opto_epochs")
    if opto_epochs is None:
        return  # not an opto session

    df = opto_epochs.to_dataframe()

    # ── Hardware tables (populated by insert_sessions() from NWB objects) ──
    virus_rows = Virus() & {"virus_name": "aav_mdlx_chr2_mcherry"}
    assert len(virus_rows) == 1, f"Expected 1 Virus row, got {len(virus_rows)}"

    inj_rows = VirusInjection() & nwb_dict
    assert len(inj_rows) == 2, f"Expected 2 VirusInjection rows (bilateral), got {len(inj_rows)}"

    fiber_device_rows = OpticalFiberDevice() & {"fiber_name": "fof_fiber_model"}
    assert len(fiber_device_rows) == 1, f"Expected 1 OpticalFiberDevice row, got {len(fiber_device_rows)}"

    fiber_implant_rows = OpticalFiberImplant() & nwb_dict
    assert (
        len(fiber_implant_rows) == 2
    ), f"Expected 2 OpticalFiberImplant rows (bilateral), got {len(fiber_implant_rows)}"

    # ── Post-hoc tables (inserted by insert_optogenetics()) ──
    protocol_name = nwbfile.session_id.split("-")[0]

    interval_rows = IntervalList() & nwb_dict & {"interval_list_name": "01"}
    assert len(interval_rows) == 1, f"Expected IntervalList '01', got {len(interval_rows)} rows"

    epoch_rows = TaskEpoch() & nwb_dict & {"epoch": 1}
    assert len(epoch_rows) == 1, f"Expected TaskEpoch epoch=1, got {len(epoch_rows)} rows"
    assert epoch_rows.fetch1("task_name") == protocol_name, f"TaskEpoch.task_name mismatch: expected '{protocol_name}'"

    protocol_rows = OptogeneticProtocol() & nwb_dict & {"epoch": 1}
    assert len(protocol_rows) == 1, f"Expected OptogeneticProtocol epoch=1, got {len(protocol_rows)} rows"

    db = protocol_rows.fetch1()
    expected_pulse_length = float(df["pulse_length_in_ms"].max())
    assert (
        db["pulse_length"] == expected_pulse_length
    ), f"OptogeneticProtocol.pulse_length: expected {expected_pulse_length}, got {db['pulse_length']}"
    expected_power = float(df["power_in_mW"].max())
    assert (
        db["stimulus_power"] == expected_power
    ), f"OptogeneticProtocol.stimulus_power: expected {expected_power}, got {db['stimulus_power']}"
    assert db["stimulus_object_id"] == opto_epochs.object_id, f"OptogeneticProtocol.stimulus_object_id mismatch"

    print(f"test_optogenetics passed for {nwbfile_path.name}")


# ---------------------------------------------------------------------------
# Optogenetics insertion (post-hoc, Spyglass-only tables)
# ---------------------------------------------------------------------------

# Stimulation window types and their durations (from _bcontrol_metadata.yaml).
# Used in the session-level OptogeneticProtocol description.
_OPTO_WINDOW_TYPES = {
    "Full Trial": 1300.0,  # 0–1.3 s post-cpoke
    "First Half": 650.0,  # 0–0.65 s post-cpoke
    "Second Half": 650.0,  # 0.65–1.3 s post-cpoke
}


def insert_optogenetics(nwb_copy_file_name: str, nwbfile) -> None:
    """Populate Spyglass optogenetics tables post-hoc for opto sessions.

    Reads per-trial data from ``nwbfile.intervals["opto_epochs"]`` and inserts
    ``Task``, ``IntervalList``, ``TaskEpoch``, and ``OptogeneticProtocol``
    directly (bypassing ``make()``). These tables are Spyglass-specific and
    are not embedded in the NWB file itself.

    ``OptogeneticProtocol`` stores one row for the whole session (epoch 1).
    The session-level ``pulse_length`` is the maximum across trial types
    (1300 ms for Full Trial). Per-trial parameters are in
    ``nwbfile.intervals["opto_epochs"]``.

    Also verifies that ``Virus``, ``VirusInjection``, ``OpticalFiberDevice``,
    and ``OpticalFiberImplant`` were populated by ``insert_sessions()``.
    """
    opto_epochs = nwbfile.intervals.get("opto_epochs")
    if opto_epochs is None:
        return  # not an opto session

    nwb_dict = {"nwb_file_name": nwb_copy_file_name}
    df = opto_epochs.to_dataframe()

    # Full session time range from the trials table.
    trials_df = nwbfile.trials.to_dataframe() if nwbfile.trials is not None else df
    session_stop = float(trials_df["stop_time"].max())

    # Session-level aggregate values: use max across all opto trials.
    max_pulse_length = float(df["pulse_length_in_ms"].max())
    max_period = float(df["period_in_ms"].max())
    session_power = float(df["power_in_mW"].max())

    # Protocol name from session_id (e.g. "TaskSwitch6-190815a" → "TaskSwitch6").
    protocol_name = nwbfile.session_id.split("-")[0]

    window_doc = ", ".join(f"{wtype}={ms:.0f}ms" for wtype, ms in _OPTO_WINDOW_TYPES.items())
    # OptogeneticProtocol.description is varchar(255); keep well under that limit.
    description = (
        f"Bilateral FOF ChR2 via Cerebro. "
        f"Per-trial window types ({window_doc}; re cpoke). "
        f"Session pulse_length=max ({max_pulse_length:.0f} ms). "
        f"Per-trial detail: nwb.intervals['opto_epochs']."
    )[:255]

    # 1. IntervalList — session-level interval anchoring TaskEpoch.
    interval_list_name = "01"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_copy_file_name,
            "interval_list_name": interval_list_name,
            "valid_times": np.array([[0.0, session_stop]]),
        },
        skip_duplicates=True,
    )

    # 2. Task — protocol-level metadata (shared across sessions).
    Task.insert1(
        {
            "task_name": protocol_name,
            "task_description": "Auditory decision-making task-switching paradigm (BControl)",
            "task_type": "auditory decision-making",
            "task_subtype": "task-switching",
        },
        skip_duplicates=True,
    )

    # 3. TaskEpoch — one epoch per session (arc_behavior has no multi-epoch structure).
    if not (TaskEpoch() & nwb_dict & {"epoch": 1}):
        TaskEpoch.insert1(
            {
                "nwb_file_name": nwb_copy_file_name,
                "epoch": 1,
                "task_name": protocol_name,
                "interval_list_name": interval_list_name,
                "task_environment": "behavioral_box",
                "camera_names": [],
            },
            allow_direct_insert=True,
        )

    # 4. OptogeneticProtocol — one row per epoch (session).
    if not (OptogeneticProtocol() & nwb_dict & {"epoch": 1}):
        OptogeneticProtocol.insert1(
            {
                "nwb_file_name": nwb_copy_file_name,
                "epoch": 1,
                "description": description,
                "pulse_length": max_pulse_length,
                "pulses_per_train": 1,
                "period": max_period,
                "intertrain_interval": 0.0,
                "stimulus_power": session_power,
                # object_id of the opto_epochs TimeIntervals table in the NWB file,
                # used as the stimulus reference (Spyglass stores this as a UUID string).
                "stimulus_object_id": opto_epochs.object_id,
            }
        )

    print("Optogenetics tables populated: IntervalList, Task, TaskEpoch, OptogeneticProtocol.")

    # Verify that insert_sessions() populated the hardware tables from the NWB objects.
    # If these are empty it means the Tier 1 field fixes in _optogenetics.py are missing.
    warnings = []
    if not (Virus() & {"virus_name": "aav_mdlx_chr2_mcherry"}):
        warnings.append("Virus")
    if not (VirusInjection() & nwb_dict):
        warnings.append("VirusInjection")
    if not (OpticalFiberDevice() & {"fiber_name": "fof_fiber_model"}):
        warnings.append("OpticalFiberDevice")
    if not (OpticalFiberImplant() & nwb_dict):
        warnings.append("OpticalFiberImplant")
    if warnings:
        print(
            f"  WARNING: {', '.join(warnings)} inserted 0 rows — "
            "Tier 1 field fixes in _optogenetics.py + _bcontrol_metadata.yaml are still needed."
        )


# ---------------------------------------------------------------------------
# Main insertion
# ---------------------------------------------------------------------------


def insert_session(
    nwbfile_path: Path,
    rollback_on_fail: bool = True,
    raise_err: bool = True,
    clean_existing: bool = False,
):
    """Insert one arc_behavior NWB session into the Spyglass database.

    Parameters
    ----------
    nwbfile_path :
        Full path to the NWB file.  The file MUST be at the **root** of
        SPYGLASS_RAW_DIR (not in a subject subfolder).  Spyglass copies
        the file to a ``_``-suffixed name in that same directory and creates
        an HDF5 external link back to the source.  The link is stored as a
        path relative to SPYGLASS_RAW_DIR, so if the source is in a subfolder
        the link resolves to the wrong location and the copy reads stale data.
        Copy with::

            cp /path/to/sub-ID/sub-ID_ses-<session>.nwb  \\
               /Volumes/T9/data/Pagan/raw/sub-ID_ses-<session>.nwb
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

    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    nwb_dict = dict(nwb_file_name=nwb_copy_file_name)

    if clean_existing:
        print(f"Cleaning existing entries for {nwbfile_path.name} ...")
        sgc_nwbfile = Nwbfile() & nwb_dict
        if sgc_nwbfile:
            sgc_nwbfile.delete(safemode=False)

    # Apply Spyglass compatibility patches to the file in SPYGLASS_RAW_DIR.
    print(f"Patching for Spyglass compatibility: {nwbfile_path.name}")
    patch_for_spyglass(nwbfile_path)

    # Core Spyglass tables (Session, Subject, etc.)
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

    # Optogenetics tables (only for sessions with active laser stimulation).
    insert_optogenetics(nwb_copy_file_name, nwbfile)
    test_optogenetics(nwbfile_path)

    print_tables(nwbfile_path)


if __name__ == "__main__":
    # Representative sessions across protocol families. Update these filenames
    # to match sessions that have been converted and copied to SPYGLASS_RAW_DIR
    # before running.
    for nwb_file_name in [
        "sub-P007_ses-PBups-150427a.nwb",
        "sub-P116_ses-ProAnti3Marino-170625a.nwb",
        "sub-H113_ses-TaskSwitch4-170630a.nwb",
        "sub-P131_ses-TaskSwitch6-190815a.nwb",
    ]:
        nwbfile_path = SPYGLASS_RAW_DIR / nwb_file_name
        insert_session(nwbfile_path=nwbfile_path, clean_existing=True)
        test_task_recording_types(nwbfile_path=nwbfile_path)
        test_optogenetics(nwbfile_path=nwbfile_path)
