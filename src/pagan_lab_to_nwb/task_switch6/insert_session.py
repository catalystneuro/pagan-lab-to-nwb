"""Ingest behavior data from an NWB file into a spyglass database."""

import sys
from pathlib import Path

import datajoint as dj

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
from src.pagan_lab_to_nwb.spyglass_extensions.states import StatesTable


def insert_states(nwbfile_path: Path):
    """Insert states from the NWB file into the Spyglass database."""
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    StatesTable().make(key={"nwb_file_name": nwb_copy_file_name})


if __name__ == "__main__":
    # Check if the NWB file exists before inserting
    nwbfile_path = Path("/Volumes/T9/data/Pagan/raw/test_with_spyglass3.nwb")
    if not nwbfile_path.exists():
        raise FileNotFoundError(f"NWB file does not exist: {nwbfile_path}")

    # this removes all tables from the database
    sgc_nwbfile = sgc.Nwbfile()
    sgc_nwbfile.delete()

    # Insert the session and states into the SpyGlass database
    sgi.insert_sessions(str(nwbfile_path), rollback_on_fail=True, raise_err=True)
    insert_states(nwbfile_path=nwbfile_path)

    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    print("=== Session ===")
    print(sgc.Session & {"nwb_file_name": nwb_copy_file_name})
    print("=== NWB File ===")
    print(sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name})
    print("=== StatesTable ===")
    print(StatesTable())

    # fetch times
    times = (StatesTable() & {"nwb_file_name": nwb_copy_file_name, "state_type": "state_0"}).fetch1("valid_times")

    # os.remove(nwbfile_path.parent / nwb_copy_file_name)  # Clean up the NWB file after testing
