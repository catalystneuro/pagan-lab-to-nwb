"""Primary NWBConverter class for this dataset."""

from neuroconv import NWBConverter
from neuroconv.datainterfaces import (
    PhySortingInterface,
    SpikeGLXRecordingInterface,
)
from pagan_lab_to_nwb.arc_ecephys import ArcEcephysBehaviorInterface


class ArcEcephysNWBConverter(NWBConverter):
    """Primary conversion class for my extracellular electrophysiology dataset."""

    data_interface_classes = dict(
        Recording=SpikeGLXRecordingInterface,
        Sorting=PhySortingInterface,
        Behavior=ArcEcephysBehaviorInterface,
    )
