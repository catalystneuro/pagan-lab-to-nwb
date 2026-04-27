"""Interface for reading spike-sorted .mat files (spikes_@*.mat) into NWB Units table.

Uses ndx-franklab-novela types (DataAcqDevice, Probe, Shank, ShanksElectrode,
NwbElectrodeGroup) for Spyglass ingestion compatibility.
"""

from pathlib import Path

import numpy as np
from hdmf.common import DynamicTable
from ndx_franklab_novela import (
    DataAcqDevice,
    NwbElectrodeGroup,
    Probe,
    Shank,
    ShanksElectrode,
)
from pydantic import validate_call
from pynwb.file import NWBFile
from pynwb.misc import Units

from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.tools import get_module
from neuroconv.utils import (
    DeepDict,
    dict_deep_update,
    get_base_schema,
    load_dict_from_file,
)


class SpikeSortingMatInterface(BaseDataInterface):
    """Interface for spike-sorted MATLAB files produced by the Pagan Lab pipeline.

    Reads ``spikes_@*.mat`` files (HDF5 / MATLAB v7.3 format) containing:
    - spike timestamps for each sorted unit
    - mean waveform and waveform SD per unit (61 samples × 4 tetrode channels)
    - tetrode assignment per unit
    - usable recording window (``goodp``)

    Writes to NWB:
    - ``nwbfile.electrodes``  — one row per channel (4 ch × n_tetrodes)
    - ``nwbfile.electrode_groups``  — one group per unique tetrode
    - ``nwbfile.units``  — spike times, waveforms, and tetrode ID per unit
    """

    display_name = "Pagan Lab Spike Sorting MAT"
    keywords = ("extracellular electrophysiology", "spike sorting", "tetrode", "units")
    associated_suffixes = (".mat",)
    info = "Interface for spike-sorted tetrode data from the Pagan Lab MATLAB pipeline."

    @validate_call
    def __init__(self, file_path: Path, verbose: bool = False):
        super().__init__(file_path=file_path)
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open(self):
        """Open the HDF5 file and cache the handle."""
        if hasattr(self, "_h5"):
            return
        import h5py

        self._h5 = h5py.File(self.source_data["file_path"], "r")

    def _close(self):
        if hasattr(self, "_h5"):
            self._h5.close()
            del self._h5

    def _read_scalar_str(self, key: str) -> str:
        """Read a UTF-16LE encoded scalar string dataset."""
        raw = self._h5[key][:]
        return "".join(chr(c) for c in raw.flatten())

    def _deref_array(self, ref) -> np.ndarray:
        """Dereference an HDF5 object reference and return a numpy array."""
        return self._h5[ref][:]

    def _read_units(self, stub_test: bool = False) -> dict:
        """Read all unit data from the HDF5 file.

        Returns a dict with keys:
            n_units, trodes, goodp, spike_times, waveform_mean, waveform_sd
        """
        self._open()
        f = self._h5

        n_units = f["spikes"].shape[0]
        if stub_test:
            n_units = min(n_units, 20)

        trodes = f["trode"][:n_units].flatten().astype(int)
        goodp = f["goodp"][:].flatten()  # [recording_start_sec, recording_end_sec]

        spike_times = []
        waveform_mean = []
        waveform_sd = []

        for i in range(n_units):
            # Spike times
            ref = f["spikes"][i, 0]
            st = self._deref_array(ref).flatten()
            spike_times.append(st)

            # Mean waveform — shape (n_samples, n_channels) or transposed
            wref = f["wave"][i, 0]
            wm = self._deref_array(wref)
            if wm.ndim == 2 and wm.shape[0] < wm.shape[1]:
                wm = wm.T  # ensure (n_samples, n_channels)
            waveform_mean.append(wm.astype(np.float32))

            # Waveform SD
            wsref = f["wavestd"][i, 0]
            ws = self._deref_array(wsref)
            if ws.ndim == 2 and ws.shape[0] < ws.shape[1]:
                ws = ws.T
            waveform_sd.append(ws.astype(np.float32))

        return dict(
            n_units=n_units,
            trodes=trodes,
            goodp=goodp,
            spike_times=spike_times,
            waveform_mean=waveform_mean,
            waveform_sd=waveform_sd,
        )

    # ------------------------------------------------------------------
    # NeuroConv interface methods
    # ------------------------------------------------------------------

    def get_metadata_schema(self) -> dict:
        metadata_schema = super().get_metadata_schema()
        metadata_schema["properties"]["Ecephys"] = get_base_schema(tag="Ecephys")
        metadata_schema["properties"]["Ecephys"].update(
            required=["DataAcqDevice", "Probe"],
            properties=dict(
                Units=dict(
                    type="object",
                    properties=dict(
                        description=dict(type="string"),
                    ),
                ),
                DataAcqDevice=dict(
                    type="object",
                    properties=dict(
                        name=dict(type="string"),
                        system=dict(type="string"),
                        amplifier=dict(type="string"),
                        adc_circuit=dict(type="string"),
                        description=dict(type="string"),
                        manufacturer=dict(type="string"),
                    ),
                ),
                Probe=dict(
                    type="object",
                    properties=dict(
                        name=dict(type="string"),
                        probe_type=dict(type="string"),
                        units=dict(type="string"),
                        probe_description=dict(type="string"),
                        contact_size=dict(type="number"),
                    ),
                ),
            ),
        )
        return metadata_schema

    def get_metadata(self) -> DeepDict:
        metadata = super().get_metadata()
        editable_metadata_path = Path(__file__).parent.parent / "metadata" / "_spike_sorting_mat_metadata.yaml"
        editable_metadata = load_dict_from_file(editable_metadata_path)
        metadata = dict_deep_update(metadata, editable_metadata)
        return metadata

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: dict,
        protocol: str = "Ecephys",
        stub_test: bool = False,
    ) -> None:
        unit_data = self._read_units(stub_test=stub_test)
        unique_trodes = sorted(set(unit_data["trodes"].tolist()))

        # ---- DataAcqDevice (Spyglass: DataAcquisitionDevice.make()) ----
        daq_meta = metadata["Ecephys"]["DataAcqDevice"]
        if daq_meta["name"] not in nwbfile.devices:
            data_acq_device = DataAcqDevice(
                name=daq_meta["name"],
                system=daq_meta["system"],
                amplifier=daq_meta["amplifier"],
                adc_circuit=daq_meta["adc_circuit"],
                description=daq_meta.get("description", ""),
                manufacturer=daq_meta.get("manufacturer", ""),
            )
            nwbfile.add_device(data_acq_device)
        else:
            data_acq_device = nwbfile.devices[daq_meta["name"]]

        # ---- Probe hierarchy (Spyglass: Probe.make()) ----
        # One Probe device for the whole tetrode array; each tetrode is a Shank.
        probe_meta = metadata["Ecephys"]["Probe"]
        if probe_meta["name"] not in nwbfile.devices:
            # One shank (shank 0) with 4 ShanksElectrodes (the 4 wires of a tetrode).
            # All tetrodes on the array share this physical contact geometry.
            # probe_shank=0 and probe_electrode=0..3 in the session Electrode table
            # must match this hierarchy exactly for Spyglass FK constraints.
            n_channels = 4
            shanks_electrodes = [
                ShanksElectrode(name=str(ch), rel_x=0.0, rel_y=0.0, rel_z=0.0) for ch in range(n_channels)
            ]
            shanks = [Shank(name="0", shanks_electrodes=shanks_electrodes)]
            probe = Probe(
                name=probe_meta["name"],
                id=0,
                probe_type=probe_meta["probe_type"],
                units=probe_meta["units"],
                probe_description=probe_meta["probe_description"],
                contact_side_numbering=False,
                contact_size=probe_meta.get("contact_size"),  # None → NULL; float("nan") breaks DJ queries
                shanks=shanks,
            )
            nwbfile.add_device(probe)
        else:
            probe = nwbfile.devices[probe_meta["name"]]

        # ---- NwbElectrodeGroups — one per tetrode (Spyglass: ElectrodeGroup.make()) ----
        # TODO (open_questions Q2): replace "unknown" with actual brain region per tetrode
        for trode_id in unique_trodes:
            group_name = f"tetrode{trode_id}"
            if group_name not in nwbfile.electrode_groups:
                group = NwbElectrodeGroup(
                    name=group_name,
                    description=f"Tetrode {trode_id}",
                    location="unknown",  # placeholder — see open_questions.md Q2
                    device=probe,  # must point to Probe, not DataAcqDevice
                    targeted_location="unknown",  # placeholder — see open_questions.md Q2
                    targeted_x=float("nan"),
                    targeted_y=float("nan"),
                    targeted_z=float("nan"),
                    units="mm",
                )
                nwbfile.add_electrode_group(group)

        # ---- Electrodes table — 4 channels per tetrode ----
        # Add Spyglass-required columns BEFORE adding any electrode rows.
        n_channels_per_tetrode = 4
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

        for trode_id in unique_trodes:
            group = nwbfile.electrode_groups[f"tetrode{trode_id}"]
            for ch in range(n_channels_per_tetrode):
                nwbfile.add_electrode(
                    group=group,
                    location="unknown",  # placeholder — see open_questions.md Q2
                    filtering="none",
                    probe_shank=0,  # tetrodes have 1 shank → shank 0
                    probe_electrode=ch,  # 0–3 within tetrode
                    bad_channel=False,
                    ref_elect_id=-1,  # unknown — update when lab confirms
                )

        # Ensure behavior processing module exists (required by Spyglass ingestion)
        if "behavior" not in nwbfile.processing:
            nwbfile.create_processing_module(name="behavior", description="Behavioral data")

        # Build a lookup: trode_id → index of its first electrode row
        trode_to_electrode_idx = {trode_id: i * n_channels_per_tetrode for i, trode_id in enumerate(unique_trodes)}

        # ---- Units table ----
        units_description = metadata.get("Ecephys", {}).get("Units", {}).get("description", "Spike-sorted units.")
        if nwbfile.units is None:
            nwbfile.units = Units(name="units", description=units_description)

        nwbfile.add_unit_column(name="trode_id", description="Tetrode ID (1-indexed)")
        nwbfile.add_unit_column(
            name="waveform_mean",
            description=(
                "Mean spike waveform across all spikes for this unit. " "Shape: (n_samples=61, n_channels=4) per unit."
            ),
        )
        nwbfile.add_unit_column(
            name="waveform_sd",
            description=(
                "Standard deviation of spike waveforms across all spikes. "
                "Shape: (n_samples=61, n_channels=4) per unit."
            ),
        )

        goodp = unit_data["goodp"]
        if goodp is not None and len(goodp) >= 2:
            from pynwb.epoch import TimeIntervals

            valid_times = TimeIntervals(
                name="goodp",
                description=(
                    "Usable recording window from the spike-sorting pipeline "
                    "(goodp: [recording_start_sec, recording_end_sec])."
                ),
            )
            valid_times.add_interval(start_time=float(goodp[0]), stop_time=float(goodp[1]))
            nwbfile.add_time_intervals(valid_times)

        for i in range(unit_data["n_units"]):
            trode_id = int(unit_data["trodes"][i])
            electrode_idx = trode_to_electrode_idx.get(trode_id, 0)
            electrode_indices = [electrode_idx + ch for ch in range(n_channels_per_tetrode)]

            st = unit_data["spike_times"][i]

            nwbfile.add_unit(
                spike_times=st,
                electrodes=electrode_indices,
                trode_id=trode_id,
                waveform_mean=unit_data["waveform_mean"][i],
                waveform_sd=unit_data["waveform_sd"][i],
            )

        # Add a custom processing module for tasks
        tasks_module = get_module(nwbfile, name="tasks", description="tasks module")
        if protocol not in tasks_module.data_interfaces:
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
                camera_id=np.array([], dtype=np.int32),
                task_epochs=[1],
            )
            tasks_module.add(task_table)

        # Max spike time across all units — actual end of ephys recording.
        # goodp is intentionally not used here: it marks the validated subset of the
        # recording, not the full session duration.
        max_spike = max(
            float(nwbfile.units["spike_times"][i][-1])
            for i in range(len(nwbfile.units))
            if len(nwbfile.units["spike_times"][i]) > 0
        )
        nwbfile.add_epoch(
            start_time=0.0,
            stop_time=max_spike,
            tags=["01"],
        )

        if self.verbose:
            total_spikes = sum(len(st) for st in unit_data["spike_times"][: unit_data["n_units"]])
            print(
                f"Added {unit_data['n_units']} units across "
                f"{len(unique_trodes)} tetrodes "
                f"({total_spikes} total spikes). "
                f"goodp IntervalList: {goodp[0]:.1f}–{goodp[1]:.1f} s"
            )

        self._close()
