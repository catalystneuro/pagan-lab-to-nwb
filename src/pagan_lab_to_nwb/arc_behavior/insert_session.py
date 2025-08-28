from pathlib import Path

import datajoint as dj
import pandas as pd
from numpy.testing import assert_array_equal

dj_local_conf_path = "/Users/weian/catalystneuro/pagan-lab-to-nwb/src/pagan_lab_to_nwb/spyglass_mock/dj_local_conf.json"
dj.config.load(dj_local_conf_path)  # load config for database connection info

dj.conn(use_tls=False)

from spyglass.common import Nwbfile
from spyglass.common.common_task_rec import TaskRecording, TaskRecordingTypes
from spyglass.data_import import insert_sessions
from spyglass.settings import raw_dir
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename, get_nwb_file


def insert_session(nwbfile_path: Path, rollback_on_fail: bool = True, raise_err: bool = True):
    """Insert session data from an NWB file into the Spyglass database.

    This function extracts session information from an NWB file and inserts it into
    the Spyglass database using the Nwbfile table. It first gets the NWB copy file name
    and then uses the sgi.insert_sessions() method to insert the data.

    Parameters
    ----------
    nwbfile_path : Path
        Path to the NWB file containing the session information to be inserted.
    rollback_on_fail : bool, optional
        If True, roll back the transaction if any error occurs during insertion.
        Defaults to True.
    raise_err : bool, optional
        If True, raise an error if the insertion fails. Defaults to True.
    """
    if not nwbfile_path.exists():
        raise FileNotFoundError(f"NWB file does not exist: {nwbfile_path}.")

    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    nwb_dict = dict(nwb_file_name=nwb_copy_file_name)
    sgc_nwbfile = Nwbfile() & nwb_dict
    if sgc_nwbfile:
        sgc_nwbfile.delete()  # Delete existing Nwbfile entry if it exists

    # populate all common tables
    insert_sessions(str(nwbfile_path), rollback_on_fail=rollback_on_fail, raise_err=raise_err)

    nwbfile = get_nwb_file(data_path)

    rec_types = TaskRecordingTypes()
    if not rec_types & nwb_dict:
        rec_types.insert_from_nwbfile(nwb_copy_file_name, nwbfile)

    task_rec = TaskRecording()
    if not task_rec & nwb_dict:
        task_rec.insert_from_nwbfile(nwb_copy_file_name, nwbfile)


def print_tables(nwbfile_path: Path):
    """Print the contents of the TaskRecordingTypes tables for the given NWB file."""

    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    nwb_dict = dict(nwb_file_name=nwb_copy_file_name)
    with open("tables.txt", "w") as f:
        print("=== TaskRecordingTypes.StateTypes ===", file=f)
        state_types = (TaskRecordingTypes.StateTypes() & nwb_dict).fetch(as_dict=True)
        state_types_df = pd.DataFrame(state_types).set_index("id")
        print(state_types_df.to_markdown(), file=f)
        print("\n=== TaskRecordingTypes.EventTypes ===", file=f)
        event_types = (TaskRecordingTypes.EventTypes() & nwb_dict).fetch(as_dict=True)
        event_types_df = pd.DataFrame(event_types).set_index("id")
        print(event_types_df.to_markdown(), file=f)
        print("\n=== TaskRecordingTypes.ActionTypes ===", file=f)
        action_types = (TaskRecordingTypes.ActionTypes() & nwb_dict).fetch(as_dict=True)
        action_types_df = pd.DataFrame(action_types).set_index("id")
        print(action_types_df.to_markdown(), file=f)
        print("\n=== TaskRecordingTypes.Arguments ===", file=f)
        arguments = (TaskRecordingTypes.Arguments() & nwb_dict).fetch(as_dict=True)
        arguments_df = pd.DataFrame(arguments)
        print(arguments_df.to_markdown(), file=f)


def test_task_recording_types(nwbfile_path: Path):
    """Test the TaskRecordingTypes against the NWB file."""

    nwbfile = get_nwb_file(data_path)
    task = nwbfile.lab_meta_data["task"]
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    nwb_dict = dict(nwb_file_name=nwb_copy_file_name)

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


if __name__ == "__main__":

    nwb_file_name = "sub-P131_ses-190815a.nwb"
    data_path = Path(raw_dir) / nwb_file_name

    insert_session(nwbfile_path=data_path)
    print_tables(nwbfile_path=data_path)
    test_task_recording_types(nwbfile_path=data_path)
