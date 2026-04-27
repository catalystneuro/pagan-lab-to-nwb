"""NWBConverter for the arc_ecephys pipeline (behavior + spikes + processed trials + video)."""

from neuroconv import NWBConverter
from pagan_lab_to_nwb.interfaces import BControlBehaviorInterface
from pagan_lab_to_nwb.interfaces._spikes_mat import SpikesMatInterface
from pagan_lab_to_nwb.interfaces.processed_trials_interface import (
    ProcessedTrialsInterface,
)
from pagan_lab_to_nwb.interfaces.spyglass_video_interface import SpyglassVideoInterface


class ArcEcephysNWBConverter(NWBConverter):
    """Converter for sessions with behavior + spike-sorted ephys + optional video.

    Data streams and interface order (order matters — each interface may depend on
    objects created by a preceding one):

      1. BControlBehavior  — creates nwbfile.trials
      2. SpikesMat         — creates nwbfile.units, electrodes, Probe + DataAcqDevice
                             hierarchy, and behavior processing module
      3. ProcessedTrials   — adds processed_trials TimeIntervals to behavior module
      4. Video             — adds CameraDevice + ImageSeries in behavior module
                             (Spyglass VideoFile.make() requires CameraDevice)

    All four interfaces are optional at run-time: omit a key from ``source_data``
    to skip that interface for sessions where the corresponding file is absent.

    Spyglass compatibility
    ----------------------
    - Uses ``DataAcqDevice`` (ndx-franklab-novela) instead of plain ``Device``
    - Uses ``NwbElectrodeGroup`` + ``Probe`` hierarchy for Spyglass Electrode tables
    - Adds Spyglass-required electrode columns: probe_shank, probe_electrode,
      bad_channel, ref_elect_id
    - Uses ``CameraDevice`` (ndx-franklab-novela) for VideoFile.make()
    - See ``metadata.yaml`` and ``open_questions.md`` for placeholder values to
      confirm with the lab before production ingestion
    """

    data_interface_classes = dict(
        Behavior=BControlBehaviorInterface,
        Video=SpyglassVideoInterface,
        ProcessedTrials=ProcessedTrialsInterface,
        SpikesMat=SpikesMatInterface,
    )
