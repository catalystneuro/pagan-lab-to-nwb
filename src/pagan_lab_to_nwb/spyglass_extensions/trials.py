"""SpyGlass extension for storing and retrieving trial information from NWB files.

This module defines a custom Spyglass table for storing trial information extracted from NWB files.
Trials represent discrete experimental epochs, each with a start and stop time, that are commonly used for behavioral and neurophysiological analyses.

The module provides functionality to:
1. Extract trial information from NWB files
2. Store the trial information in a structured database table
3. Enable querying of trial information for analysis

This extension integrates with the Spyglass ecosystem to provide a standardized way
to access and analyze trial information across different experiments.
"""

import datajoint as dj
import numpy as np
from spyglass.common.common_nwbfile import Nwbfile
from spyglass.utils import SpyglassMixin
from spyglass.utils.nwb_helper_fn import get_nwb_file

schema = dj.schema("common_trial")


@schema
class TrialsTable(SpyglassMixin, dj.Imported):
    """Table for storing trial information from NWB files.

    This table stores time intervals representing individual trials during behavioral or neurophysiological experiments.
    Each trial is identified by a trial number and has associated start and stop times. The trials are extracted from
    the NWB file's trials table.

    The primary key of this table consists of:
    1. A foreign key to the Session table (nwb_file_name)
    2. The trial number (trial)

    Attributes
    ----------
    definition : str
        DataJoint table definition
    """

    definition = """
    # Time intervals used for analysis
    -> Session
    trial_id: int  # trial number
    ---
    valid_times: longblob  # numpy array with start/stop times for each trial
    pipeline = "": varchar(64)  # type of interval list
    """

    def make(self, key):
        """Populate the table with trial information from an NWB file.

        This method extracts trial information from an NWB file and inserts it into the TrialsTable.
        It reads the trials from the NWB file's trials table, processes each trial to extract the trial number,
        start time, and stop time, and inserts them into the table.

        Parameters
        ----------
        key : dict
            Dictionary containing the 'nwb_file_name' key, which specifies the NWB file to process.

        Notes
        -----
        This method assumes that:
        - The NWB file exists and is accessible
        - The NWB file contains a 'trials' table with 'start_time' and 'stop_time' columns
        - Each row in the 'trials' table represents a single trial
        """

        nwb_file_name = key["nwb_file_name"]
        nwb_file_abspath = Nwbfile().get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)

        if nwbf.trials is None:
            return

        trials = nwbf.trials.to_dataframe()

        inserts = trials.apply(
            lambda row: {
                "nwb_file_name": nwb_file_name,
                "trial_id": row.name,
                "valid_times": np.asarray([[row.start_time, row.stop_time]]),
            },
            axis=1,
        ).tolist()

        self.insert(inserts, allow_direct_insert=True, skip_duplicates=True)
