"""Primary NWBConverter class for this dataset."""

from neuroconv import NWBConverter
from pagan_lab_to_nwb.interfaces import BControlBehaviorInterface


class ArcBehaviorNWBConverter(NWBConverter):
    """Primary conversion class for TaskSwitch6 behavioral protocol."""

    data_interface_classes = dict(
        Behavior=BControlBehaviorInterface,
    )
