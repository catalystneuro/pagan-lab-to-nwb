"""Spyglass-compatible SpikeGadgets recording interface for Pagan Lab.

Inherits from NeuroConv's SpikeGadgetsRecordingInterface for extractor setup,
then overrides add_to_nwbfile to create ndx-franklab-novela device types
(DataAcqDevice, Probe, NwbElectrodeGroup) required by Spyglass ingestion.

The electrode group naming (``tetrode{N}``) matches the SpikeSortingMatInterface
convention so both interfaces can coexist in the same converter:
when SpikeGadgets is listed first, it builds the full electrode table;
SpikeSortingMatInterface then detects the pre-existing rows and maps units
into the correct electrode indices instead of adding duplicate rows.

Note: NeuroConv 0.9.x + SpikeInterface 0.99.x have an incompatible import
(SortingAnalyzer missing from SI 0.99).  get_metadata() and add_to_nwbfile()
are fully overridden here to avoid the broken neuroconv.tools.spikeinterface
path.  Device metadata is loaded from _spike_sorting_mat_metadata.yaml so
both interfaces share the same DataAcqDevice / Probe definitions.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.data_utils import DataChunkIterator
from ndx_franklab_novela import (
    DataAcqDevice,
    NwbElectrodeGroup,
    Probe,
    Shank,
    ShanksElectrode,
)
from pydantic import FilePath
from pynwb.ecephys import ElectricalSeries
from pynwb.file import NWBFile

from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.datainterfaces import SpikeGadgetsRecordingInterface
from neuroconv.utils import DeepDict, dict_deep_update, load_dict_from_file


def _get_spikegadgets_header(file_path: str | Path) -> str:
    """Read the XML header from a SpikeGadgets .rec file.

    Reads bytes until </Configuration> and returns the decoded header string.

    Raises
    ------
    ValueError
        If the header does not contain ``</Configuration>``.
    """
    header_size = None
    with open(file_path, mode="rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if b"</Configuration>" in line:
                header_size = f.tell()
                break
        if header_size is None:
            raise ValueError(f"SpikeGadgets: XML header in '{file_path}' does not contain " "'</Configuration>'.")
        f.seek(0)
        return f.read(header_size).decode("utf-8")


class SpyglassSpikeGadgetsRecordingInterface(SpikeGadgetsRecordingInterface):
    """Spyglass-compatible SpikeGadgets raw recording interface for Pagan Lab.

    Reads a SpikeGadgets .rec file and writes an ``ElectricalSeriesRaw`` to
    ``nwbfile.acquisition`` using the ndx-franklab-novela device hierarchy
    (DataAcqDevice → Probe → NwbElectrodeGroup) required by Spyglass ingestion.

    Electrode table
    ---------------
    One row per recorded channel, ordered by the SpikeGadgets hardware channel
    index.  Required Spyglass columns (probe_shank, probe_electrode, bad_channel,
    ref_elect_id) are added automatically.  Electrode groups follow the naming
    convention ``tetrode{N}`` (1-indexed tetrode ID from the .rec header) to
    match the existing SpikeSortingMatInterface convention.

    Combined use with SpikeSortingMatInterface
    ------------------------------------------
    List this interface FIRST in ``data_interface_classes`` so the electrode
    table is fully populated before SpikeSortingMatInterface runs.  That interface
    detects the pre-existing rows and maps sorted units into the correct electrode
    indices rather than adding duplicate rows.

    Spyglass insertion
    ------------------
    No changes to insert_session.py are required.  ``sgi.insert_sessions()``
    automatically populates the ``Raw`` table from the ``ElectricalSeriesRaw``
    object in acquisition, and the existing ``seed_lookup_tables()`` already
    registers the DataAcquisitionDevice and ProbeType entries that Spyglass
    needs.
    """

    display_name = "Pagan Lab SpikeGadgets Recording"
    keywords = ("extracellular electrophysiology", "raw recording", "tetrode", "SpikeGadgets")
    associated_suffixes = (".rec",)
    info = "Interface for raw SpikeGadgets tetrode recordings (Pagan Lab, Spyglass-compatible)."

    def __init__(self, file_path: FilePath, verbose: bool = False, **kwargs):
        """
        Parameters
        ----------
        file_path :
            Path to the SpikeGadgets ``.rec`` file.
        verbose :
            Print summary after conversion.
        """
        super().__init__(file_path=file_path, verbose=verbose, **kwargs)
        self._verbose = verbose

        # Parse .rec header: map hardware channel IDs → tetrode IDs and
        # intra-tetrode channel indices (0-indexed).
        header_txt = _get_spikegadgets_header(file_path)
        root = ElementTree.fromstring(header_txt)
        sconf = root.find("SpikeConfiguration")

        self._hwchan_to_ntrode: dict[str, int] = {}
        self._hwchan_to_trode_ch: dict[str, int] = {}

        if sconf is not None:
            for tetrode in sconf.findall("SpikeNTrode"):
                ntrode_id = int(tetrode.attrib["id"])
                for ch_idx, electrode in enumerate(tetrode.findall("SpikeChannel")):
                    hw_chan = electrode.attrib["hwChan"]
                    self._hwchan_to_ntrode[hw_chan] = ntrode_id
                    self._hwchan_to_trode_ch[hw_chan] = ch_idx

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_metadata(self) -> DeepDict:
        # Bypass the broken neuroconv.tools.spikeinterface import path.
        # Call BaseDataInterface.get_metadata directly (NWBFile watermark only).
        metadata = BaseDataInterface.get_metadata(self)

        # Load device/probe metadata shared with SpikeSortingMatInterface
        editable_metadata_path = Path(__file__).parent.parent / "metadata" / "_spike_sorting_mat_metadata.yaml"
        editable_metadata = load_dict_from_file(editable_metadata_path)
        metadata = dict_deep_update(metadata, editable_metadata)
        return metadata

    # ------------------------------------------------------------------
    # NWB conversion
    # ------------------------------------------------------------------

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: dict,
        stub_test: bool = False,
        **kwargs,
    ) -> None:
        """Add raw SpikeGadgets recording to the NWB file.

        Creates (or reuses) ndx-franklab-novela DataAcqDevice, Probe, and
        NwbElectrodeGroup entries, then writes an ``ElectricalSeriesRaw``
        to ``nwbfile.acquisition``.

        Parameters
        ----------
        nwbfile :
            Target NWB file.
        metadata :
            Metadata dict (must contain ``Ecephys.DataAcqDevice`` and
            ``Ecephys.Probe`` — provided by ``get_metadata()``).
        stub_test :
            If True, limit data to the first second of recording for fast testing.
        """
        daq_meta = metadata["Ecephys"]["DataAcqDevice"]
        probe_meta = metadata["Ecephys"]["Probe"]

        # --- DataAcqDevice (ndx-franklab-novela) ---
        if daq_meta["name"] not in nwbfile.devices:
            nwbfile.add_device(
                DataAcqDevice(
                    name=daq_meta["name"],
                    system=daq_meta["system"],
                    amplifier=daq_meta["amplifier"],
                    adc_circuit=daq_meta["adc_circuit"],
                    description=daq_meta.get("description", ""),
                    manufacturer=daq_meta.get("manufacturer", ""),
                )
            )

        # --- Probe (ndx-franklab-novela) ---
        if probe_meta["name"] not in nwbfile.devices:
            n_ch_per_tetrode = 4
            shanks_electrodes = [
                ShanksElectrode(name=str(ch), rel_x=0.0, rel_y=0.0, rel_z=0.0) for ch in range(n_ch_per_tetrode)
            ]
            nwbfile.add_device(
                Probe(
                    name=probe_meta["name"],
                    id=0,
                    probe_type=probe_meta["probe_type"],
                    units=probe_meta["units"],
                    probe_description=probe_meta["probe_description"],
                    contact_side_numbering=False,
                    contact_size=probe_meta.get("contact_size"),
                    shanks=[Shank(name="0", shanks_electrodes=shanks_electrodes)],
                )
            )
        probe = nwbfile.devices[probe_meta["name"]]

        # --- Channel → tetrode mapping ---
        recording = self.recording_extractor
        channel_ids = recording.get_channel_ids()
        channel_names = recording.get_property("channel_name", ids=channel_ids)
        if channel_names is None:
            channel_names = [str(cid) for cid in channel_ids]

        channel_ntrodes: list[int] = []
        channel_trode_chs: list[int] = []
        for cname in channel_names:
            hw_chan = str(cname).split("hwChan")[-1]
            channel_ntrodes.append(self._hwchan_to_ntrode.get(hw_chan, 1))
            channel_trode_chs.append(self._hwchan_to_trode_ch.get(hw_chan, 0))

        # --- NwbElectrodeGroup — one per unique tetrode ---
        for ntrode_id in sorted(set(channel_ntrodes)):
            group_name = f"tetrode{ntrode_id}"
            if group_name not in nwbfile.electrode_groups:
                nwbfile.add_electrode_group(
                    NwbElectrodeGroup(
                        name=group_name,
                        description=f"Tetrode {ntrode_id}",
                        location="unknown",  # placeholder — see open_questions.md Q2
                        device=probe,
                        targeted_location="unknown",
                        targeted_x=float("nan"),
                        targeted_y=float("nan"),
                        targeted_z=float("nan"),
                        units="mm",
                    )
                )

        # --- Electrode table columns (Spyglass-required) ---
        spyglass_cols = [
            ("probe_shank", "Shank index on the probe (0-indexed)"),
            ("probe_electrode", "Electrode index within the shank (0-indexed)"),
            ("bad_channel", "Whether this channel is flagged as bad"),
            ("ref_elect_id", "Reference electrode ID (-1 if no reference)"),
        ]
        existing_cols = list(nwbfile.electrodes.colnames) if nwbfile.electrodes else []
        for col_name, col_desc in spyglass_cols:
            if col_name not in existing_cols:
                nwbfile.add_electrode_column(name=col_name, description=col_desc)

        # --- Electrode rows ---
        # Only add rows if the table is empty.  SpikeSortingMatInterface may
        # have run first and pre-populated it; in that case we skip adding rows
        # (the existing rows cover the same channels) and reference them below.
        electrode_start_idx = len(nwbfile.electrodes) if nwbfile.electrodes else 0
        n_channels = len(channel_ids)

        if electrode_start_idx == 0:
            for i in range(n_channels):
                nwbfile.add_electrode(
                    group=nwbfile.electrode_groups[f"tetrode{channel_ntrodes[i]}"],
                    location="unknown",
                    filtering="none",
                    probe_shank=0,
                    probe_electrode=channel_trode_chs[i],
                    bad_channel=False,
                    ref_elect_id=-1,
                )
            electrode_indices = list(range(n_channels))
        else:
            # Pre-existing rows: reference all of them.
            electrode_indices = list(range(len(nwbfile.electrodes)))
            if self._verbose and len(electrode_indices) != n_channels:
                print(
                    f"[SpikeGadgets] Warning: electrode table has "
                    f"{len(electrode_indices)} rows but recording has {n_channels} "
                    "channels.  Electrode reference may be imprecise.  For best "
                    "results, list SpikeGadgets before SpikeSortingMat in the "
                    "converter's data_interface_classes."
                )

        # --- ElectricalSeriesRaw ---
        if stub_test:
            n_stub = int(recording.get_sampling_frequency() * 1.0)  # 1 s
            recording = recording.frame_slice(start_frame=0, end_frame=n_stub)

        fs = recording.get_sampling_frequency()
        n_frames = recording.get_num_frames()

        electrode_table_region = nwbfile.create_electrode_table_region(
            region=electrode_indices,
            description="All SpikeGadgets recorded channels.",
        )

        # Gain: SpikeGadgets stores int16 ADC counts; gains are in µV/count
        gains = recording.get_channel_gains()
        gain_v = float(gains[0]) * 1e-6 if gains is not None else 1e-6

        # Spyglass requires explicit timestamps (always_write_timestamps=True)
        timestamps = np.arange(n_frames, dtype=np.float64) / fs

        # Memory-efficient chunk iterator — avoids loading the full recording
        _chunk_size = 10_000

        def _data_gen():
            for start in range(0, n_frames, _chunk_size):
                end = min(start + _chunk_size, n_frames)
                yield recording.get_traces(start_frame=start, end_frame=end, return_scaled=False)

        data_iter = DataChunkIterator(
            data=_data_gen(),
            iter_axis=0,
            maxshape=(n_frames, n_channels),
            dtype=np.dtype("int16"),
        )

        electrical_series = ElectricalSeries(
            name="ElectricalSeriesRaw",
            data=H5DataIO(data=data_iter, compression="gzip", compression_opts=4),
            electrodes=electrode_table_region,
            timestamps=H5DataIO(data=timestamps, compression="gzip", compression_opts=4),
            description="Raw broadband signal from SpikeGadgets tetrode array.",
            conversion=gain_v,
        )

        nwbfile.add_acquisition(electrical_series)

        # --- Epoch (Spyglass IntervalList / TaskEpoch) ---
        duration = n_frames / fs
        if nwbfile.epochs is None or len(nwbfile.epochs) == 0:
            nwbfile.add_epoch(start_time=0.0, stop_time=duration, tags=["01"])

        # --- Behavior processing module (required by Spyglass) ---
        if "behavior" not in nwbfile.processing:
            nwbfile.create_processing_module(name="behavior", description="Behavioral data")

        if self._verbose:
            print(
                f"[SpikeGadgets] ElectricalSeriesRaw: "
                f"{n_channels} ch × {n_frames} samples "
                f"({n_frames / fs:.1f} s @ {fs:.0f} Hz)  "
                f"gain={gain_v * 1e6:.4f} µV/count"
            )
