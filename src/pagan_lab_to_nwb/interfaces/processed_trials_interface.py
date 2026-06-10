"""Interface for writing processed trial data from dati_*.mat files to NWB TimeIntervals."""

import json
from pathlib import Path
from warnings import warn

import numpy as np
from pydantic import validate_call
from pynwb import TimeSeries
from pynwb.file import NWBFile

from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.tools import get_module
from neuroconv.utils import (
    DeepDict,
    dict_deep_update,
    get_base_schema,
    load_dict_from_file,
)

# Row-index → NWB column name for the 7-row ``tim`` matrix (confirmed by lab, 2026-04-21).
# Order is fixed by the mat file format; descriptions live in _processed_trials_metadata.yaml.
_TIM_COLUMN_MAP = {
    0: "previous_trial_end",
    1: "trial_ready",  # used as start_time for processed trial intervals (10 ms offset from BControl clock)
    2: "cue_start",
    3: "poke_in",
    4: "poke_out",
    5: "choice_time",
    6: "trial_end",  # used as stop_time for processed trial intervals (10 ms offset from BControl clock)
}


class ProcessedTrialsInterface(BaseDataInterface):
    """Interface for the Pagan Lab processed trial data file (``dati_*.mat``).

    Reads the MATLAB v5 file containing:
    - per-trial behavioral variables (choice, hits, task, side, gdir, gfreq, nta, stim)
    - 7 key event timestamps per trial (``tim``)
    - optional rrr4 PSTH tensor (n_units × n_trials × n_bins)

    Writes to NWB:
    - ``processing["behavior"]["processed_trials"]`` — a ``TimeIntervals`` table with
      all behavioral variables and a ``timeseries`` column linking each trial to its
      PSTH row in ``processing["ecephys"]["rrr4_psth"]`` (when rrr4 is present).
    - ``processing["ecephys"]["rrr4_psth"]`` — a ``TimeSeries`` with shape
      ``(n_trials, n_units, n_bins)``, aligned to cue onset, values in spikes/s.
    """

    display_name = "Pagan Lab Processed Trials"
    keywords = ("behavior", "trials")
    associated_suffixes = (".mat",)
    info = "Interface for processed trial data from dati_*.mat files (Pagan Lab pipeline)."

    @validate_call
    def __init__(self, file_path: Path, verbose: bool = False):
        super().__init__(file_path=file_path)
        self.verbose = verbose

    def _read_file(self):
        if hasattr(self, "_dati"):
            return
        from pymatreader import read_mat

        self._dati = read_mat(str(self.source_data["file_path"]))

    def get_metadata_schema(self) -> dict:
        metadata_schema = super().get_metadata_schema()
        metadata_schema["properties"]["Behavior"] = get_base_schema(tag="Behavior")
        metadata_schema["properties"]["Behavior"].update(
            required=["TimeIntervals"],
            properties=dict(TimeIntervals=dict(type="object", properties=dict(description={"type": "string"}))),
        )
        return metadata_schema

    def get_metadata(self) -> DeepDict:
        metadata = super().get_metadata()
        editable_metadata_path = Path(__file__).parent.parent / "metadata" / "_processed_trials_metadata.yaml"
        editable_metadata = load_dict_from_file(editable_metadata_path)
        metadata = dict_deep_update(metadata, editable_metadata)
        return metadata

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: dict,
        stub_test: bool = False,
    ) -> None:
        self._read_file()
        d = self._dati

        if "tim" not in d:
            warn("ProcessedTrialsInterface: 'tim' not found in dati file. Skipping.")
            return

        tim = np.asarray(d["tim"])  # expected shape: (7, n_trials)
        if tim.ndim != 2 or tim.shape[0] < 7:
            warn(f"ProcessedTrialsInterface: unexpected 'tim' shape {tim.shape}. Skipping.")
            return

        n = tim.shape[1]

        def _to_float_list(arr):
            arr = np.asarray(arr, dtype=float).flatten()
            if len(arr) != n:
                warn(f"ProcessedTrialsInterface: field length {len(arr)} != {n}. Truncating/padding.")
            if len(arr) >= n:
                return arr[:n].tolist()
            padded = np.full(n, np.nan)
            padded[: len(arr)] = arr
            return padded.tolist()

        def _to_str_list(s):
            chars = list(str(s)) if isinstance(s, str) else [str(x) for x in np.asarray(s).flatten()]
            if len(chars) != n:
                warn(f"ProcessedTrialsInterface: string field length {len(chars)} != {n}. Truncating/padding.")
            if len(chars) >= n:
                return chars[:n]
            return chars + [""] * (n - len(chars))

        # ---- rrr4 PSTH — create TimeSeries before intervals so we can pass timeseries refs ----
        # The TimeSeries must exist before add_interval() is called because PyNWB creates
        # the indexed timeseries column on the first add_interval() call that includes it.
        psth_ts = None
        if "rrr4" in d and "centers4" in d:
            rrr4 = np.asarray(d["rrr4"], dtype=np.float32)  # (n_units, n_trials, n_bins)
            centers4 = np.asarray(d["centers4"], dtype=np.float64)

            if rrr4.ndim == 3 and len(centers4) >= 2 and rrr4.shape[1] == n:
                n_units_rrr4 = rrr4.shape[0]
                n_bins = rrr4.shape[2]
                bin_width_ms = float(np.round((centers4[1] - centers4[0]) * 1000))

                # Clamp to len(nwbfile.units) in case stub_test reduced the unit count
                if nwbfile.units is not None and len(nwbfile.units) > 0:
                    n_units_used = min(n_units_rrr4, len(nwbfile.units))
                    if n_units_used < n_units_rrr4:
                        warn(f"ProcessedTrialsInterface: clamping rrr4 from {n_units_rrr4} to {n_units_used} units.")
                        rrr4 = rrr4[:n_units_used, :, :]

                    # Transpose (n_units, n_trials, n_bins) → (n_trials, n_units, n_bins)
                    # so axis 0 is the time/trial axis required by TimeSeries.
                    psth_data = np.transpose(rrr4, (1, 0, 2))

                    # Timestamps: trial_ready (tim row 1) — always defined and monotonically
                    # increasing. PSTH bins are relative to cue_start within each trial.
                    trial_ready_ts = np.array(_to_float_list(tim[1, :]), dtype=np.float64)

                    psth_ts = TimeSeries(
                        name="rrr4_psth",
                        description=(
                            f"Peri-stimulus time histogram (PSTH) of spike rates aligned to "
                            f"auditory cue onset. "
                            f"Shape per sample: (n_units={rrr4.shape[0]}, n_bins={n_bins}). "
                            f"Timestamps are trial_ready (start_time) for each trial; "
                            f"PSTH bins are relative to cue_start within each trial. "
                            f"Bin width: {bin_width_ms:.0f} ms. "
                            f"Time range: {float(centers4[0]):.2f} to {float(centers4[-1]):.2f} s "
                            f"relative to cue onset. "
                            f"Axis 1 (units) corresponds to nwbfile.units[0:{rrr4.shape[0]}]. "
                            "Values in spikes/s. "
                            "Produced by the Pagan Lab spike-sorting pipeline (rrr4 variable)."
                        ),
                        data=psth_data,
                        timestamps=trial_ready_ts,
                        unit="spikes/s",
                    )
                    ecephys = get_module(nwbfile, name="ecephys", description="Processed electrophysiology data.")
                    ecephys.add(psth_ts)
                else:
                    warn("ProcessedTrialsInterface: nwbfile.units is empty — skipping rrr4 PSTH.")
            else:
                warn("ProcessedTrialsInterface: rrr4/centers4 shape mismatch or trial count mismatch. Skipping PSTH.")

        # ---- Build processed_trials TimeIntervals ----
        from pynwb.epoch import TimeIntervals

        table_meta = metadata.get("Behavior", {}).get("TimeIntervals", {})
        cols_meta = table_meta.get("columns", {})

        def _desc(col_name):
            return cols_meta.get(col_name, {}).get("description", "")

        dati_trials = TimeIntervals(
            name=table_meta.get("name", "processed_trials"),
            description=table_meta.get("description", ""),
        )

        # start_time = trial_ready (tim row 1), stop_time = trial_end (tim row 6).
        # Each interval optionally carries a timeseries reference to its PSTH row.
        start_times = _to_float_list(tim[1, :])
        stop_times = _to_float_list(tim[6, :])
        if psth_ts is not None:
            for st, sp in zip(start_times, stop_times):
                # PyNWB computes idx_start and count from psth_ts.timestamps vs (st, sp).
                # Since psth_ts.timestamps = trial_ready_ts and st = trial_ready[i],
                # idx_start = i and count = 1 for every trial.
                dati_trials.add_interval(start_time=st, stop_time=sp, timeseries=[psth_ts])
        else:
            for st, sp in zip(start_times, stop_times):
                dati_trials.add_interval(start_time=st, stop_time=sp, check_ragged=False)

        # ---- Per-trial behavioral columns (all dati variables) ----
        if "choice" in d:
            dati_trials.add_column(name="choice", description=_desc("choice"), data=_to_float_list(d["choice"]))

        if "hits" in d:
            dati_trials.add_column(name="hits", description=_desc("hits"), data=_to_float_list(d["hits"]))

        if "nta" in d:
            dati_trials.add_column(name="nta", description=_desc("nta"), data=_to_float_list(d["nta"]))

        if "side" in d:
            _side_map = {"l": "Left", "r": "Right"}
            dati_trials.add_column(
                name="correct_side",
                description=_desc("correct_side"),
                data=[_side_map.get(c, c) for c in _to_str_list(d["side"])],
            )

        if "task" in d:
            _task_map = {"d": "Direction", "f": "Frequency"}
            dati_trials.add_column(
                name="task_context",
                description=_desc("task_context"),
                data=[_task_map.get(c, c) for c in _to_str_list(d["task"])],
            )

        if "gdir" in d:
            dati_trials.add_column(name="gdir", description=_desc("gdir"), data=_to_float_list(d["gdir"]))

        if "gfreq" in d:
            dati_trials.add_column(name="gfreq", description=_desc("gfreq"), data=_to_float_list(d["gfreq"]))

        # ---- tim: all 7 event timestamps per trial ----
        for row_idx, col_name in _TIM_COLUMN_MAP.items():
            if row_idx >= tim.shape[0]:
                break
            if col_name in ["trial_ready", "trial_end"]:
                continue  # already used as start_time / stop_time
            dati_trials.add_column(name=col_name, description=_desc(col_name), data=_to_float_list(tim[row_idx, :]))

        # ---- stim: per-trial stimulus pulse dicts — JSON serialised ----
        if "stim" in d:
            stim = d["stim"] if isinstance(d["stim"], list) else list(d["stim"])
            if len(stim) > n:
                stim = stim[:n]
            elif len(stim) < n:
                warn(f"ProcessedTrialsInterface: 'stim' length {len(stim)} != {n}. Padding.")
                stim = stim + [{}] * (n - len(stim))
            stim_json = []
            for entry in stim:
                try:
                    stim_json.append(json.dumps(entry, default=str))
                except Exception:
                    stim_json.append("{}")
            dati_trials.add_column(name="stim_params", description=_desc("stim_params"), data=stim_json)

        # ---- Add to behavior processing module ----
        behavior = get_module(
            nwbfile=nwbfile,
            name="behavior",
            description="Contains processed trial-by-trial data derived from BControl.",
        )
        behavior.add(dati_trials)

        if self.verbose:
            psth_note = f", rrr4_psth TimeSeries ({psth_ts.data.shape})" if psth_ts is not None else ""
            print(f"Added TimeIntervals with {n} trials to behavior processing module{psth_note}.")
