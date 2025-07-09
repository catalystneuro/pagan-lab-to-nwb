"""Module for inserting data from NWB files into a Spyglass database.

This module provides functions to insert data from NWB files into a Spyglass database.
It specifically handles the insertion of state information from behavioral experiments.

The module connects to a DataJoint database using configuration from a local file
and provides functionality to:
1. Insert session data using Spyglass's built-in tools
2. Insert custom state information using the StatesTable

Example usage is provided in the __main__ section, which demonstrates how to:
- Check if an NWB file exists
- Clear existing database tables if needed
- Insert session and state data
- Query and display the inserted data
"""

import sys
from pathlib import Path

import datajoint as dj
from numpy.testing import assert_array_equal
from pynwb import NWBHDF5IO

dj_local_conf_path = "/Users/weian/catalystneuro/pagan-lab-to-nwb/src/pagan_lab_to_nwb/spyglass_mock/dj_local_conf.json"
dj.config.load(dj_local_conf_path)  # load config for database connection info

dj.conn(use_tls=False)

# spyglass.common has the most frequently used tables
import spyglass.common as sgc  # this import connects to the database

# spyglass.data_import has tools for inserting NWB files into the database
import spyglass.data_import as sgi
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

# Custom Table Imports
sys.path.append("/Users/weian/catalystneuro/pagan-lab-to-nwb/src/pagan_lab_to_nwb/spyglass_extensions")
from pagan_lab_to_nwb.spyglass_extensions.states import StatesTable
from pagan_lab_to_nwb.spyglass_extensions.trials import TrialsTable


def insert_session(nwbfile_path: Path):
    """Insert session data from an NWB file into the Spyglass database.

    This function extracts session information from an NWB file and inserts it into
    the Spyglass database using the Nwbfile table. It first gets the NWB copy file name
    and then uses the sgi.insert_sessions() method to insert the data.

    Parameters
    ----------
    nwbfile_path : Path
        Path to the NWB file containing the session information to be inserted.
    """
    if not nwbfile_path.exists():
        raise FileNotFoundError(f"NWB file does not exist: {nwbfile_path}.")

    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    # this removes all tables from the database
    sgc_nwbfile = sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name}
    sgc_nwbfile.delete()

    sgi.insert_sessions(str(nwbfile_path), rollback_on_fail=True, raise_err=True)
    insert_states(nwbfile_path=nwbfile_path)
    insert_trials(nwbfile_path=nwbfile_path)


def insert_states(nwbfile_path: Path):
    """Insert states from the NWB file into the Spyglass database.

    This function extracts state information from an NWB file and inserts it into
    the Spyglass database using the StatesTable. It first gets the NWB copy file name
    and then uses the StatesTable.make() method to insert the data.

    Parameters
    ----------
    nwbfile_path : Path
        Path to the NWB file containing the state information to be inserted.
    """
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    StatesTable().make(key={"nwb_file_name": nwb_copy_file_name})


def insert_trials(nwbfile_path: Path):
    """Insert trials from the NWB file into the Spyglass database.

    This function extracts trial information from an NWB file and inserts it into
    the Spyglass database using the TrialsTable. It first gets the NWB copy file name
    and then uses the TrialsTable.make() method to insert the data.

    Parameters
    ----------
    nwbfile_path : Path
        Path to the NWB file containing the trial information to be inserted.
    """
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    TrialsTable().make(key={"nwb_file_name": nwb_copy_file_name})


def test_states(nwbfile_path: Path):
    """Test function to insert states from an NWB file into the Spyglass database."""
    state_type = "state_0"
    with NWBHDF5IO(nwbfile_path, "r") as io:
        nwbfile = io.read()
        states = nwbfile.acquisition["task_recording"].states.to_dataframe()
        states["state_name"] = states["state_type"].apply(lambda row: row["state_name"].iloc[0])
        intervals_from_nwb = states[states["state_name"] == state_type][["start_time", "stop_time"]].to_numpy()
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    intervals_from_spyglass = (StatesTable() & {"nwb_file_name": nwb_copy_file_name, "state_type": state_type}).fetch1(
        "valid_times"
    )
    assert_array_equal(intervals_from_nwb, intervals_from_spyglass)


def test_trials(nwbfile_path: Path):
    """Test function to insert trials from an NWB file into the Spyglass database."""
    with NWBHDF5IO(nwbfile_path, "r") as io:
        nwbfile = io.read()
        trials = nwbfile.trials.to_dataframe()
        trials = trials[["start_time", "stop_time"]].to_numpy()
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    for i in range(len(trials)):
        trials_from_spyglass = (TrialsTable() & {"nwb_file_name": nwb_copy_file_name, "trial_id": i}).fetch1(
            "valid_times"
        )
        assert_array_equal(trials[i], trials_from_spyglass.squeeze())


if __name__ == "__main__":

    nwbfile_path = Path("/Volumes/T9/data/Pagan/raw/sub-H7015_ses-250516a.nwb")
    insert_session(nwbfile_path=nwbfile_path)

    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    print("=== Session ===")
    print(sgc.Session & {"nwb_file_name": nwb_copy_file_name})
    print("=== NWB File ===")
    print(sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name})
    print("=== StatesTable ===")
    print(StatesTable() & {"nwb_file_name": nwb_copy_file_name})
    print("=== TrialsTable ===")
    print(TrialsTable() & {"nwb_file_name": nwb_copy_file_name})

    test_states(nwbfile_path=nwbfile_path)
    test_trials(nwbfile_path=nwbfile_path)
