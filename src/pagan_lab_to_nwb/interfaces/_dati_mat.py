"""Interface for reading processed trial/neural data from dati_*.mat files."""

import json
from pathlib import Path
from warnings import warn

import numpy as np
from pydantic import validate_call
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
# Order is fixed by the mat file format; descriptions live in metadata.yaml.
_TIM_COLUMN_MAP = {
    0: "previous_trial_end",
    1: "trial_ready",  # used as start_time for processed trial intervals (10 ms offset from BControl clock)
    2: "cue_start",
    3: "poke_in",
    4: "poke_out",
    5: "choice_time",
    6: "trial_end",  # used as stop_time for processed trial intervals (10 ms offset from BControl clock)
}


class DatiMatInterface(BaseDataInterface):
    """Interface for the Pagan Lab processed trial/neural data file (``dati_*.mat``).

    Reads the MATLAB v5 file containing:
    - per-trial behavioral variables (choice, hits, task, side, gdir, gfreq, nta, stim)
    - 7 key event timestamps per trial (``tim``)

    Writes to NWB:
    - ``processing["behavior"]["processed_trials"]`` — a ``TimeIntervals`` table containing
      all dati behavioral variables (choice, hits, nta, correct_side, task_context,
      gdir, gfreq, all 7 tim event timestamps, stim_params).  start_time/stop_time
      are set from tim rows 1 and 6 (dati clock; ~10 ms offset from BControl clock).

    """

    display_name = "Pagan Lab Dati MAT"
    keywords = ("behavior", "trials")
    associated_suffixes = (".mat",)
    info = "Interface for processed trial and neural response data (dati_*.mat)."

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

        editable_metadata_path = Path(__file__).parent.parent / "metadata" / "_dati_mat_metadata.yaml"
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
            warn("DatiMatInterface: 'tim' not found in dati file. Skipping.")
            return

        tim = np.asarray(d["tim"])  # expected shape: (7, n_trials)
        if tim.ndim != 2 or tim.shape[0] < 7:
            warn(f"DatiMatInterface: unexpected 'tim' shape {tim.shape}. Skipping.")
            return

        n = tim.shape[1]

        def _to_float_list(arr):
            arr = np.asarray(arr, dtype=float).flatten()
            if len(arr) != n:
                warn(f"DatiMatInterface: field length {len(arr)} != {n}. Truncating/padding.")
            if len(arr) >= n:
                return arr[:n].tolist()
            padded = np.full(n, np.nan)
            padded[: len(arr)] = arr
            return padded.tolist()

        def _to_str_list(s):
            chars = list(str(s)) if isinstance(s, str) else [str(x) for x in np.asarray(s).flatten()]
            if len(chars) != n:
                warn(f"DatiMatInterface: string field length {len(chars)} != {n}. Truncating/padding.")
            if len(chars) >= n:
                return chars[:n]
            return chars + [""] * (n - len(chars))

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
        start_times = _to_float_list(tim[1, :])
        stop_times = _to_float_list(tim[6, :])
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
                continue  # these are already used as start_time, stop_time
            dati_trials.add_column(name=col_name, description=_desc(col_name), data=_to_float_list(tim[row_idx, :]))

        # ---- stim: per-trial stimulus pulse dicts — JSON serialised ----
        if "stim" in d:
            stim = d["stim"] if isinstance(d["stim"], list) else list(d["stim"])
            if len(stim) > n:
                stim = stim[:n]
            elif len(stim) < n:
                warn(f"DatiMatInterface: 'stim' length {len(stim)} != {n}. Padding.")
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
            print(f"Added TimeIntervals with {n} trials to behavior processing module.")
