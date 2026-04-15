"""Functions for adding StimulusSection data to the trials table."""

from warnings import warn

import numpy as np
import pandas as pd
from pynwb.file import NWBFile


def add_stimulus_to_trials(
    nwbfile: NWBFile,
    saved_history: dict,
    metadata: dict,
    stimulus_field_name: str = "StimulusSection_ThisStimulus",
) -> None:
    """Add per-trial stimulus parameters from ``saved_history`` to ``nwbfile.trials``.

    This is a no-op for non-TaskSwitch sessions that lack ``StimulusSection_ThisStimulus``.

    Scalar stimulus parameters are added as plain trial columns.
    Pulse-time lists (left_hi, right_hi, left_lo, right_lo) are added as ragged columns
    storing times relative to cpoke onset.  A companion ``cpoke_start_time`` column
    (absolute, NaN when the rat did not poke) allows conversion to session time:
    ``abs_time = cpoke_start_time + pulse_time``.
    """
    if stimulus_field_name not in saved_history:
        warn(f"Stimulus field '{stimulus_field_name}' not found in saved_history. Skipping.")
        return

    stimulus_data = saved_history[stimulus_field_name]
    trials = nwbfile.trials
    if not isinstance(stimulus_data, list):
        stimulus_data = [stimulus_data]
    if len(stimulus_data) >= len(trials):
        stimulus_data = stimulus_data[: len(trials)]
    else:
        raise ValueError(
            f"Stimulus entries ({len(stimulus_data)}) < trials ({len(trials)}). " "Cannot align stimulus to trials."
        )

    stimulus_data = pd.DataFrame(stimulus_data)

    if "freqs" in stimulus_data.columns:
        stimulus_data["freq_lo"] = stimulus_data["freqs"].apply(
            lambda x: x[0] if isinstance(x, (list, np.ndarray)) else np.nan
        )
        stimulus_data["freq_hi"] = stimulus_data["freqs"].apply(
            lambda x: x[1] if isinstance(x, (list, np.ndarray)) else np.nan
        )
        stimulus_data.drop(columns=["freqs"], inplace=True)

    # ── Ragged pulse-time columns ──────────────────────────────────────────────
    # Times are stored relative to cpoke onset (centre-port entry), exactly as
    # they come from BControl.  Add cpoke_start_time to convert to absolute time.
    # For error/invalid trials (rat never poked) cpoke_start_time is NaN; relative
    # pulse times are still stored because the stimulus was generated.
    # Descriptions are read from metadata.yaml (Behavior.TrialsTable.pulse_columns).
    pulse_col_meta = metadata.get("Behavior", {}).get("TrialsTable", {}).get("pulse_columns", {})
    pulse_column_descriptions = {
        col: pulse_col_meta.get(col, "no description") for col in ("left_hi", "right_hi", "left_lo", "right_lo")
    }

    n_trials = len(trials)
    parsed_events_all = saved_history.get("ProtocolsSection_parsed_events", [])

    # Extract cpoke start time for each trial (used to align pulse times)
    cpoke_start_times = []
    for i in range(n_trials):
        try:
            cpoke_state = parsed_events_all[i].get("states", {}).get("cpoke", [])
            if len(cpoke_state) == 0:
                cpoke_start_times.append(None)
            else:
                cpoke_start_times.append(float(np.asarray(cpoke_state).flat[0]))
        except (IndexError, KeyError, TypeError, AttributeError):
            cpoke_start_times.append(None)

    nwbfile.trials.add_column(
        name="cpoke_start_time",
        description=pulse_col_meta.get("cpoke_start_time", "no description"),
        data=[t if t is not None else float("nan") for t in cpoke_start_times],
    )

    for stimulus_type, description in pulse_column_descriptions.items():
        if stimulus_type not in stimulus_data.columns:
            continue

        per_trial_times = []
        for trial_idx in range(n_trials):
            raw_values = stimulus_data[stimulus_type].values[trial_idx]
            if isinstance(raw_values, float):
                raw_values = [] if np.isnan(raw_values) else [raw_values]
            elif isinstance(raw_values, (int, np.integer)):
                raw_values = [float(raw_values)]
            elif isinstance(raw_values, np.ndarray):
                raw_values = raw_values.tolist()
            elif not isinstance(raw_values, list):
                raw_values = list(raw_values) if raw_values else []
            # Filter out zeros/falsy values (MATLAB zero-padding)
            per_trial_times.append([float(v) for v in raw_values if v])

        flat_times = np.array([t for times in per_trial_times for t in times], dtype=float)
        cumsum = np.cumsum([len(t) for t in per_trial_times])
        if len(flat_times) == 0:
            # All trials have zero pulses of this type; HDMF rejects an empty data
            # array paired with a non-empty index, so skip this column entirely.
            continue
        nwbfile.trials.add_column(
            name=stimulus_type,
            description=description,
            data=flat_times,
            index=cumsum,
        )

    # ── Scalar stimulus columns ────────────────────────────────────────────────
    stimulus_parameters_metadata = metadata["Behavior"]["TrialsTable"]["columns"]
    for col_meta in stimulus_parameters_metadata:
        stimulus_name = col_meta["name"]
        if stimulus_name not in stimulus_data.columns:
            warn(f"Stimulus parameter '{stimulus_name}' not found in stimulus data. Skipping.")
            continue
        stimulus_values = stimulus_data[stimulus_name].values.tolist()
        if any(isinstance(v, dict) for v in stimulus_values):
            warn(f"Stimulus parameter '{stimulus_name}' has unsupported dict values. Skipping.")
            continue
        if any(isinstance(v, np.ndarray) for v in stimulus_values):
            stimulus_values = [val.tolist() if isinstance(val, np.ndarray) else val for val in stimulus_values]
        nwbfile.trials.add_column(
            name=stimulus_name,
            description=col_meta["description"],
            data=stimulus_values,
        )

    print(f"Added {len(stimulus_data.columns)} stimulus parameters to trials.")
