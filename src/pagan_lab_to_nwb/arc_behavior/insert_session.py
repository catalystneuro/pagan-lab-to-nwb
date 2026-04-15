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

import spyglass.common as sgc
import spyglass.data_import as sgi
from spyglass.common import Nwbfile
from spyglass.common.common_task_rec import TaskRecording, TaskRecordingTypes
from spyglass.settings import raw_dir
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename, get_nwb_file

SPYGLASS_RAW_DIR = Path(raw_dir)

# Spyglass column limits that our NWB content can exceed.
_EXPERIMENT_DESCRIPTION_LIMIT = 2000  # common_session.Session VARCHAR(2000)
_ARGUMENT_DESCRIPTION_LIMIT = 255  # common_task_rec.TaskRecordingTypes.Arguments VARCHAR(255)


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
    """
    nwbfile_path = Path(nwbfile_path)
    with h5py.File(nwbfile_path, "a") as f:
        _h5_truncate_scalar(f, "general/experiment_description", _EXPERIMENT_DESCRIPTION_LIMIT)
        _h5_truncate_string_array(
            f,
            "general/task/task_arguments/argument_description",
            _ARGUMENT_DESCRIPTION_LIMIT,
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
    print_tables(nwbfile_path)


if __name__ == "__main__":
    # Update these filenames to match sessions that have been converted and
    # copied to SPYGLASS_RAW_DIR before running.
    for nwb_file_name in [
        "sub-P131_ses-TaskSwitch6-190815a.nwb",
    ]:
        nwbfile_path = SPYGLASS_RAW_DIR / nwb_file_name
        insert_session(nwbfile_path=nwbfile_path, clean_existing=True)
        test_task_recording_types(nwbfile_path=nwbfile_path)
