"""Custom Spyglass/DataJoint table for ProcessedTrialsInterface processed_trials data.

Follows the kind-lab-to-nwb pattern: stores the NWB object_id of the
processed_trials TimeIntervals table rather than duplicating all columns
into DataJoint. The actual per-trial data is accessed by reading the NWB
file via fetch_nwb().

Schema prefix ``behavior`` is in Spyglass's SHARED_MODULES, so the
``behavior_pagan`` schema passes the SpyglassMixin prefix check without
requiring any edits to the Spyglass fork.

Import order constraint
-----------------------
dj.config.load() must be called before this module is imported — the schema
decorator executes dj.schema() at import time (same rule as all Spyglass tables).
insert_session.py handles this: it calls dj.config.load() before importing
spyglass and before importing this module.
"""

import datajoint as dj
import pynwb
from spyglass.common.common_nwbfile import Nwbfile
from spyglass.utils import SpyglassMixin

schema = dj.schema("behavior_pagan")


@schema
class ProcessedTrials(SpyglassMixin, dj.Manual):
    """Pointer to the processed_trials TimeIntervals table from ProcessedTrialsInterface.

    Stores the NWB object_id of nwbfile.processing["behavior"]["processed_trials"].
    Use fetch_nwb() to resolve the object_id back to the TimeIntervals object,
    or call fetch1_dataframe() to get a pandas DataFrame directly.

    Only populated for sessions that include a dati_*.mat file.
    """

    definition = """
    -> Nwbfile
    ---
    processed_trials_object_id = NULL: varchar(40)  # NWB object ID for the processed_trials TimeIntervals
    """
    _nwb_table = Nwbfile

    def insert_from_nwbfile(self, nwb_file_name: str, nwbf: pynwb.NWBFile) -> None:
        """Insert the processed_trials object_id pointer from an NWB file.

        Parameters
        ----------
        nwb_file_name :
            The Spyglass copy filename (with trailing ``_``).
        nwbf :
            The open NWB file object.
        """
        try:
            pt = nwbf.processing["behavior"]["processed_trials"]
        except KeyError:
            print(f"  ProcessedTrials: no processed_trials in {nwb_file_name} — skipping.")
            return

        self.insert1(
            {"nwb_file_name": nwb_file_name, "processed_trials_object_id": pt.object_id},
            skip_duplicates=True,
        )
        print(f"  ProcessedTrials: inserted (object_id={pt.object_id}) for {nwb_file_name}")

    def fetch1_dataframe(self):
        """Return the processed_trials TimeIntervals as a pandas DataFrame.

        Reads the NWB file to resolve the stored object_id.
        Requires a single-entry restriction, e.g.::

            (ProcessedTrials() & nwb_dict).fetch1_dataframe()
        """
        self.ensure_single_entry()
        nwb_data = self.fetch_nwb()[0]
        return nwb_data["processed_trials"]
