"""Spyglass-compatible video interface using ndx-franklab-novela CameraDevice."""

from pathlib import Path

import numpy as np
from ndx_franklab_novela import CameraDevice
from numpy.typing import NDArray
from pydantic import validate_call
from pynwb.file import NWBFile
from pynwb.image import ImageSeries

from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.datainterfaces.behavior.video.video_utils import get_video_timestamps
from neuroconv.utils import DeepDict, dict_deep_update, load_dict_from_file

from ._spyglass_tasks import add_spyglass_task_table


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
        self._aligned_timestamps: NDArray[np.float64] | None = None

    def set_aligned_timestamps(self, timestamps: NDArray[np.float64]) -> None:
        """Set per-frame timestamps in seconds (session clock).

        Call this before conversion when you have a hardware sync signal or any
        other source of per-frame timing.  The array must have one entry per
        video frame and be expressed in the same time base as
        ``nwbfile.session_start_time`` (i.e. seconds since session start).

        If not called, timestamps are derived from the nominal frame rate and
        optionally shifted by ``time_offset`` in ``add_to_nwbfile()``.
        """
        self._aligned_timestamps = np.asarray(timestamps, dtype=np.float64)

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
        # ndx-franklab-novela >= 0.2.4 requires 'model' to be a DeviceModel reference
        # (Spyglass's CameraDevice ingestion reads camera_device.model.name /
        # camera_device.model.manufacturer).
        if camera_meta["name"] not in nwbfile.devices:
            model_name = camera_meta.get("model", "unknown")
            manufacturer = camera_meta.get("manufacturer", "unknown")
            device_model = nwbfile.create_device_model(
                name=model_name,
                manufacturer=manufacturer,
                description=f"Behavioral camera model: {model_name}.",
            )
            camera_device = CameraDevice(
                name=camera_meta["name"],
                meters_per_pixel=float(camera_meta["meters_per_pixel"]),
                lens=camera_meta.get("lens", "unknown"),
                camera_name=camera_meta["camera_name"],
                model=device_model,
                description=f"Behavioral camera. Model: {model_name}.",
            )
            nwbfile.add_device(camera_device)
        else:
            camera_device = nwbfile.devices[camera_meta["name"]]

        video_file_path = self.source_data["file_path"]
        if self._aligned_timestamps is not None:
            timestamps = self._aligned_timestamps
        else:
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

        # Spyglass TaskEpoch.make() requires processing["tasks"][protocol] to exist.
        # No-op if the SpikeSorting interface already created it for this protocol
        # (e.g. an ephys+video session); camera_id then stays empty for that table.
        add_spyglass_task_table(nwbfile, protocol=protocol, camera_id=[1])

        if self.verbose:
            print(f"Added video '{Path(self.source_data['file_path']).name}' ")
