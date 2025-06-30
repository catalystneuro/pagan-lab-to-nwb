"""SpyGlass extension for"""

import datajoint as dj
import numpy as np
from spyglass.common.common_nwbfile import Nwbfile
from spyglass.utils import SpyglassMixin
from spyglass.utils.nwb_helper_fn import get_nwb_file

schema = dj.schema("states_table")


@schema
class StatesTable(SpyglassMixin, dj.Imported):
    """ """

    definition = """
    # Time intervals used for analysis
    -> Session
    state_type: varchar(170)  # descriptive name of this interval list
    ---
    valid_times: longblob  # numpy array with start/end times for each interval
    pipeline = "": varchar(64)  # type of interval list
    """

    def make(self, key):
        """ """

        nwb_file_name = key["nwb_file_name"]
        nwb_file_abspath = Nwbfile().get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)

        states = nwbf.acquisition["task_recording"].states.to_dataframe().head(10)
        # convert DynamicTableRegion column to string column
        states["state_name"] = states["state_type"].apply(lambda row: row["state_name"].iloc[0])

        inserts = (
            states.groupby("state_name")  # TODO: state_type is a dataframe so use .apply to extract the name
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
