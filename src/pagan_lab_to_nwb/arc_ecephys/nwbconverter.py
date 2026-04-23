"""NWBConverter for the arc_ecephys pipeline (behavior + spikes + dati + video)."""

from neuroconv import NWBConverter
from pagan_lab_to_nwb.interfaces import BControlBehaviorInterface
from pagan_lab_to_nwb.interfaces._dati_mat import DatiMatInterface
from pagan_lab_to_nwb.interfaces._spikes_mat import SpikesMatInterface
from pagan_lab_to_nwb.interfaces.spyglass_video_interface import SpyglassVideoInterface


class ArcEcephysNWBConverter(NWBConverter):
    """Converter for sessions with behavior + spike-sorted ephys + optional video.

    Data streams and interface order (order matters — each interface may depend on
    objects created by a preceding one):

      1. BControlBehavior  — creates nwbfile.trials (required by DatiMat)
      2. SpikesMat         — creates nwbfile.units, electrodes, Probe + DataAcqDevice
                             hierarchy, and behavior processing module
      3. DatiMat           — adds columns to nwbfile.trials (requires trials to exist)
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
        DatiMat=DatiMatInterface,
        SpikesMat=SpikesMatInterface,
    )
