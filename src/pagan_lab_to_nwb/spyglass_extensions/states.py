"""SpyGlass extension for storing and retrieving state information from NWB files.

This module defines a custom Spyglass table for storing state information extracted from NWB files.
States represent time intervals during behavioral experiments that can be used for analysis.
Each state has a type identifier and start/end times.

The module provides functionality to:
1. Extract state information from NWB files
2. Store the state information in a structured database table
3. Enable querying of state information for analysis

This extension integrates with the Spyglass ecosystem to provide a standardized way
to access and analyze state information across different experiments.
"""

import datajoint as dj
import numpy as np
from spyglass.common.common_nwbfile import Nwbfile
from spyglass.utils import SpyglassMixin
from spyglass.utils.nwb_helper_fn import get_nwb_file

schema = dj.schema("states")


@schema
class StatesTable(SpyglassMixin, dj.Imported):
    """Table for storing state information from NWB files.

    This table stores time intervals representing different states during behavioral experiments.
    Each state has a type identifier and a set of start/end times. The states are extracted from
    the NWB file's acquisition data, specifically from the 'task_recording' group's 'states' table.

    The primary key of this table consists of:
    1. A foreign key to the Session table (nwb_file_name)
    2. The state type identifier (state_type)

    Attributes
    ----------
    definition : str
        DataJoint table definition
    """

    definition = """
    # Time intervals used for analysis
    -> Session
    state_type: varchar(170)  # descriptive name of this interval
    ---
    valid_times: longblob  # numpy array with start/end times for each interval
    pipeline = "": varchar(64)  # type of interval list
    """

    def make(self, key):
        """Populate the table with state information from an NWB file.

        This method extracts state information from an NWB file and inserts it into the StatesTable.
        It reads the states from the 'task_recording' acquisition group in the NWB file,
        processes them to extract state types and time intervals, and inserts them into the table.

        The method groups states by their state_name and creates a record for each unique state type,
        with an array of start/end times for all intervals of that state type.

        Parameters
        ----------
        key : dict
            Dictionary containing the 'nwb_file_name' key, which specifies the NWB file to process.

        Notes
        -----
        This method assumes that:
        - The NWB file exists and is accessible
        - The NWB file contains a 'task_recording' acquisition with a 'states' table
        - The 'states' table has 'state_type', 'start_time', and 'stop_time' columns
        - The 'state_type' column contains DynamicTableRegion references to state names

        The method handles the conversion of DynamicTableRegion references to string values
        and groups the states by their names for efficient storage.
        """

        nwb_file_name = key["nwb_file_name"]
        nwb_file_abspath = Nwbfile().get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)

        states = nwbf.acquisition["task_recording"].states.to_dataframe()
        # convert DynamicTableRegion column to string column
        states["state_name"] = states["state_type"].apply(lambda row: row["state_name"].iloc[0])

        inserts = (
            states.groupby("state_name")
            .apply(
                lambda df: {
                    "nwb_file_name": nwb_file_name,
                    "state_type": df.name,
                    "valid_times": np.array(df[["start_time", "stop_time"]]),
                },
                include_groups=False,
            )
            .tolist()
        )

        self.insert(inserts, allow_direct_insert=True, skip_duplicates=True)
