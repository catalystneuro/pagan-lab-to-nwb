"""Spyglass-compatible video interface using ndx-franklab-novela CameraDevice.

Differences from NeuroConv's VideoInterface:
- Uses ``CameraDevice`` (ndx-franklab-novela) instead of a plain ``Device``.
- Places ``ImageSeries`` in the ``behavior`` processing module wrapped in
  ``BehavioralEvents`` (required by Spyglass ``VideoFile.make()``).
- Accepts real per-frame timestamps; falls back to uniform timestamps from a
  nominal frame rate when real sync data is unavailable (see open_questions.md Q3 / Q10).
- ``camera_name`` must match an entry in the Spyglass ``sgc.CameraDevice`` table
  (see open_questions.md Q11).
"""

from pathlib import Path

from hdmf.common import DynamicTable
from ndx_franklab_novela import CameraDevice
from pydantic import validate_call
from pynwb.file import NWBFile
from pynwb.image import ImageSeries

from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.datainterfaces.behavior.video.video_utils import get_video_timestamps
from neuroconv.tools import get_module
from neuroconv.utils import DeepDict, dict_deep_update, load_dict_from_file

# Nominal frame rate used to generate placeholder timestamps when real sync
# data is unavailable.  Replace with actual frame timestamps once confirmed
# (see open_questions.md Q3 / Q10).
_NOMINAL_FPS = 19.98


class SpyglassVideoInterface(BaseDataInterface):
    """Spyglass-compatible behavioral video interface.

    Parameters
    ----------
    file_path :
        Path to the video file (e.g. ``video_@*.mp4``).
    verbose :
        Print progress messages.
    """

    display_name = "Spyglass Behavioral Video"
    keywords = ("video", "behavior", "camera", "spyglass")
    associated_suffixes = (".mp4", ".avi", ".mov")
    info = "Spyglass-compatible video interface using CameraDevice (ndx-franklab-novela)."

    @validate_call
    def __init__(self, file_path: Path, verbose: bool = False):
        super().__init__(file_path=file_path)
        self.verbose = verbose

    def get_metadata(self) -> DeepDict:
        metadata = super().get_metadata()

        editable_metadata_path = Path(__file__).parent.parent / "metadata" / "_video_metadata.yaml"
        editable_metadata = load_dict_from_file(editable_metadata_path)
        metadata = dict_deep_update(metadata, editable_metadata)

        return metadata

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: dict,
        protocol: str = "Spyglass Behavioral Video",
        time_offset: float | None = None,
        stub_test: bool = False,
    ) -> None:
        video_meta = metadata["Video"]
        camera_meta = video_meta["CameraDevice"]

        # ---- CameraDevice (Spyglass: required on ImageSeries) ----
        # Note: ndx-franklab-novela >= 0.2.4 treats 'model' as a DeviceModel reference.
        # Use 'model_name' (deprecated str field) for the model string, or omit and
        # include model info in the description.
        if camera_meta["name"] not in nwbfile.devices:
            camera_device = CameraDevice(
                name=camera_meta["name"],
                meters_per_pixel=float(camera_meta["meters_per_pixel"]),
                lens=camera_meta.get("lens", "unknown"),
                camera_name=camera_meta["camera_name"],
                description=(f"Behavioral camera. Model: {camera_meta.get('model', 'unknown')}. "),
            )
            nwbfile.add_device(camera_device)
        else:
            camera_device = nwbfile.devices[camera_meta["name"]]

        video_file_path = self.source_data["file_path"]
        timestamps = get_video_timestamps(file_path=video_file_path)
        if time_offset is not None:
            timestamps = timestamps + time_offset

        nwbfile.add_epoch_column(name="task_name", description="Name of the task associated with the epoch.")
        nwbfile.add_epoch(start_time=timestamps[0], stop_time=timestamps[-1], tags=["01"], task_name=protocol)

        # ---- ImageSeries in behavior processing module ----
        # Spyglass VideoFile.make() finds ImageSeries via nwbf.objects.values()
        # and requires each ImageSeries to have a device attribute (CameraDevice).
        image_series = ImageSeries(
            name=f"Video {Path(video_file_path).stem}",
            description=video_meta.get("description", "Behavioral video recording"),
            unit="n.a.",
            external_file=[video_file_path],
            format="external",
            timestamps=timestamps,
            device=camera_device,
        )

        nwbfile.add_acquisition(image_series)

        # Add a custom processing module for tasks
        # This is necessary for the video data to be compatible with spyglass.
        tasks_module = get_module(nwbfile, name="tasks", description="tasks module")

        task_table = DynamicTable(
            name=protocol,
            description=f"{protocol} behavioral task",
        )
        task_table.add_column(name="task_name", description="Name of the task")
        task_table.add_column(name="task_description", description="Description of the task")
        task_table.add_column(name="task_type", description="Type of task (required by Spyglass Task table)")
        task_table.add_column(name="task_subtype", description="Subtype of task (required by Spyglass Task table)")
        task_table.add_column(name="task_environment", description="Recording environment")
        task_table.add_column(name="camera_id", description="Camera IDs used during this task")
        task_table.add_column(name="task_epochs", description="Epoch indices for this task")
        task_table.add_row(
            task_name=protocol,
            task_description="Auditory decision-making task-switching paradigm (BControl)",
            task_type="auditory decision-making",
            task_subtype="task-switching",
            task_environment="behavioral_box",
            camera_id=[1],
            task_epochs=[1],
        )
        tasks_module.add(task_table)

        if self.verbose:
            print(f"Added video '{Path(self.source_data['file_path']).name}' ")
