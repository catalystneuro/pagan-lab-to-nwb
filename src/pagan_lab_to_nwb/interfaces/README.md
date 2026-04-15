# interfaces

This package contains the NWB conversion interface for BControl `.mat` files.

## File overview

| File | Role |
|---|---|
| `bcontroldatainterface.py` | **Entry point.** `BControlBehaviorInterface` class — reads the `.mat` file, extracts metadata, and orchestrates the conversion by delegating to the modules below. |
| `_task_recording.py` | Builds `StateTypesTable`, `StatesTable`, `EventTypesTable`, `EventsTable`, `ActionTypesTable`, `ActionsTable`, and attaches a `Task` + `TaskRecording` to the NWB file. Used for all protocols. |
| `_trials.py` | Builds `TrialsTable` (linking each trial to its states/events/actions) and populates `TaskArgumentsTable` with session-level parameters from `saved`. Used for all protocols. |
| `_stimulus.py` | Adds per-trial stimulus columns to `TrialsTable`: scalar parameters (gamma_dir, freq_lo, …) and ragged pulse-time columns (left_hi, right_hi, left_lo, right_lo). **No-op** for non-TaskSwitch sessions that lack `StimulusSection_ThisStimulus`. |
| `_optogenetics.py` | Adds optogenetics data: per-trial opto columns on `TrialsTable`, `OptogeneticEpochsTable`, `OptogeneticSeries` (left/right), and rich `ndx-optogenetics` metadata (virus, fiber model, injection coordinates). **No-op** for sessions without active laser stimulation. |

## Design rationale

`BControlBehaviorInterface` is the single public entry point. Its `add_to_nwbfile` method
calls the free functions in each module in order:

```
add_task_recording_to_nwbfile  (_task_recording.py)  ← always runs
add_trials_to_nwbfile          (_trials.py)           ← always runs
add_stimulus_to_trials         (_stimulus.py)         ← no-op for PBups/ProAnti3
add_task_arguments             (_trials.py)           ← always runs
add_optogenetic_series_to_nwbfile (_optogenetics.py)  ← no-op for non-opto sessions
```

Stimulus and optogenetics are **independent** optional dimensions (a session can be any
combination), so they live in separate modules rather than subclasses. Adding a new optional
data channel means creating a new module and adding one line to `add_to_nwbfile`.

Each module function takes plain data arguments (`parsed_events`, `saved_history`, etc.)
rather than `self`, making them independently testable without constructing a full interface.

## Adding a new data channel

1. Create `_myfeature.py` with a function `add_myfeature_to_nwbfile(nwbfile, saved_history, ...)`.
2. Make it a no-op (early return) when the relevant BControl keys are absent.
3. Import and call it in `BControlBehaviorInterface.add_to_nwbfile`.
