"""NWBConverter for the arc_ecephys pipeline (behavior + spikes + processed trials + video)."""

from neuroconv import NWBConverter
from pagan_lab_to_nwb.interfaces import BControlBehaviorInterface
from pagan_lab_to_nwb.interfaces.processed_trials_interface import (
    ProcessedTrialsInterface,
)
from pagan_lab_to_nwb.interfaces.spike_sorting_mat_interface import (
    SpikeSortingMatInterface,
)
from pagan_lab_to_nwb.interfaces.spyglass_spikegadgets_recording_interface import (
    SpyglassSpikeGadgetsRecordingInterface,
)
from pagan_lab_to_nwb.interfaces.spyglass_video_interface import SpyglassVideoInterface


class ArcEcephysNWBConverter(NWBConverter):
    """Converter for sessions with behavior + spike-sorted ephys + optional video.

    Data streams:

      Behavior          — creates nwbfile.trials
      SpikeGadgets      — (optional) writes raw ElectricalSeriesRaw, creates
                        electrode table, DataAcqDevice + Probe + NwbElectrodeGroup
                        hierarchy (must be listed before SpikeSorting)
      Video             — adds CameraDevice + ImageSeries in behavior module
                        (Spyglass VideoFile.make() requires CameraDevice)
      ProcessedTrials   — adds processed_trials TimeIntervals to behavior module
      SpikeSorting      — creates nwbfile.units; reuses electrode table if
                        SpikeGadgets is also present, otherwise creates its own

    All interfaces are optional at run-time: omit a key from ``source_data``
    to skip that interface for sessions where the corresponding file is absent.

    Spyglass compatibility
    ----------------------
    - Uses ``DataAcqDevice`` (ndx-franklab-novela) instead of plain ``Device``
    - Uses ``NwbElectrodeGroup`` + ``Probe`` hierarchy for Spyglass Electrode tables
    - Adds Spyglass-required electrode columns: probe_shank, probe_electrode,
      bad_channel, ref_elect_id
    - Uses ``CameraDevice`` (ndx-franklab-novela) for VideoFile.make()
    - SpikeGadgets interface adds ``ElectricalSeriesRaw`` to acquisition;
      Spyglass populates the ``Raw`` table from this automatically on insertion
    - See ``metadata.yaml`` and ``open_questions.md`` for placeholder values to
      confirm with the lab before production ingestion
    """

    data_interface_classes = dict(
        Behavior=BControlBehaviorInterface,
        SpikeGadgets=SpyglassSpikeGadgetsRecordingInterface,
        SpikeSorting=SpikeSortingMatInterface,
        ProcessedTrials=ProcessedTrialsInterface,
        Video=SpyglassVideoInterface,
    )
