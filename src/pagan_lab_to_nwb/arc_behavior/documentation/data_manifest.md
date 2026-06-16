# Data Manifest: BControl → NWB Conversion

This document describes exactly where every field from a BControl `.mat` file ends up in the
converted NWB file, how `ndx-structured-behavior` types are used, and how to read data back.

---

## 1. BControl Source Data Structure

A BControl `.mat` file contains two top-level structures.

### `saved` (session-level parameters)

A flat dict of all task parameters and session metadata at the time the session ended.
Keys follow the pattern `{SectionName}_{field_name}` (e.g., `ProtocolsSection_n_completed_trials`).

Key categories:

| Key prefix | What it contains |
|---|---|
| `SavingSection_*` | File save path and timestamp (`SavingSection_SaveTime`) |
| `ProtocolsSection_*` | Trial counts, protocol title, parsed events flag |
| `CommentsSection_*` | Per-animal session log and overall comments |
| `SessionDefinition_*` | Block/stage structure for PBups protocol (not in TaskSwitch) |
| `StimulusSection_*` | Stimulus parameters (scalar, session-level) |
| Everything else | Task parameters: timing, thresholds, reward amounts, etc. |

### `saved_history` (trial-by-trial data)

A dict where each key is a parameter name and the value is a list with one entry per trial.

Key fields:

| Key | What it contains |
|---|---|
| `ProtocolsSection_parsed_events` | List of trial dicts — the core behavioral data |
| `StimulusSection_ThisStimulus` | List of per-trial stimulus parameter dicts (TaskSwitch only) |
| `{ProtocolParam}_*` | Any parameter that changes trial-by-trial |

#### `parsed_events` structure (per trial)

```python
{
  "states": {
      "state_0":   np.ndarray,  # [[nan, start], [stop, nan]] or [start, stop]
      "wait_poke": np.ndarray,  # [start, stop] or [[s1,e1],[s2,e2],...]
      "cpoke":     np.ndarray,  # start of fixation (used to align pulse times)
      "reward":    np.ndarray,
      ...
  },
  "pokes": {
      "C":                np.ndarray,  # [[enter, exit], ...] for centre port
      "L":                np.ndarray,  # left port pokes
      "R":                np.ndarray,  # right port pokes
      "starting_state":   dict,        # {port: state_name_on_entry}
  },
  "waves": {
      "base_sound":       np.ndarray,  # [[start, stop], ...] for each wave
      "starting_state":   dict,        # {wave: state_name_on_start}
      ...
  },
}
```

#### `StimulusSection_ThisStimulus` structure (per trial, TaskSwitch only)

```python
{
  "gamma_dir":    float,   # log ratio right/left pulse rates
  "gamma_freq":   float,   # log ratio hi/lo frequency pulse rates
  "duration":     float,   # stimulus duration in seconds
  "freq_lo":      float,   # low-frequency tone in Hz
  "freq_hi":      float,   # high-frequency tone in Hz
  "vol":          float,   # overall volume multiplier
  "vol_low":      float,   # low-frequency volume multiplier
  "vol_hi":       float,   # high-frequency volume multiplier
  "bup_width":    float,   # single pulse duration in ms
  "bup_ramp":     float,   # pulse ramp duration in ms
  "crosstalk_dir": float,  # spatial crosstalk ratio (0=both, 1=right only)
  "crosstalk_freq": float, # frequency crosstalk ratio (sometimes a dict — skipped)
  "left_hi":      list,    # pulse times relative to cpoke start, left+high
  "right_hi":     list,    # pulse times relative to cpoke start, right+high
  "left_lo":      list,    # pulse times relative to cpoke start, left+low
  "right_lo":     list,    # pulse times relative to cpoke start, right+low
  "freqs":        list,    # [freq_lo, freq_hi] — split into freq_lo/freq_hi columns
}
```

---

## 2. NWB File Layout

```
NWBFile
├── session_description   ← from metadata.yaml
├── session_start_time    ← parsed from SavingSection_SaveTime + protocol title
├── identifier            ← auto-generated UUID
├── session_id            ← from filename (date_str part)
├── notes                 ← CommentsSection_comments + CommentsSection_overall_comments
├── experimenter          ← from metadata.yaml
├── institution           ← from metadata.yaml ("University of Edinburgh")
├── lab                   ← from metadata.yaml ("Pagan")
├── experiment_description← from metadata.yaml
├── keywords              ← from metadata.yaml
│
├── subject (Subject)
│   ├── subject_id        ← from filename (subject_id part)
│   ├── species           ← "Rattus norvegicus" (metadata.yaml)
│   ├── sex               ← "M" (metadata.yaml)
│   ├── age               ← "P6M/P2Y" (metadata.yaml)
│   └── description       ← from metadata.yaml
│
├── devices
│   ├── BControl (Device)
│   │   └── manufacturer  ← "Example Manufacturer" (default)
│   ├── Cerebro (ExcitationSource)                ← opto sessions only; ndx-optogenetics
│   │   ├── manufacturer  ← "Karpova Lab"
│   │   └── power_in_W    ← 0.025
│   ├── cerebro_model (ExcitationSourceModel)     ← opto sessions only; ndx-optogenetics
│   │   ├── source_type   ← "Solid-State Laser"
│   │   ├── excitation_mode ← "one-photon"
│   │   └── wavelength_range_in_nm ← [450.0, 450.0]
│   ├── optical_fiber_left (OpticalFiber)         ← opto sessions only; ndx-ophys-devices
│   │   └── fiber_insertion (FiberInsertion)
│   │       ├── insertion_position_ap_in_mm ← 2.0
│   │       └── insertion_position_ml_in_mm ← -1.3
│   ├── optical_fiber_right (OpticalFiber)        ← opto sessions only; ndx-ophys-devices
│   │   └── fiber_insertion (FiberInsertion)
│   │       ├── insertion_position_ap_in_mm ← 2.0
│   │       └── insertion_position_ml_in_mm ← 1.3
│   └── fof_fiber_model (OpticalFiberModel)       ← opto sessions only; ndx-ophys-devices
│       ├── numerical_aperture   ← 0.37
│       └── core_diameter_in_um  ← 400.0
│
├── lab_meta_data
│   ├── task (Task)                          ← ndx-structured-behavior
│       ├── state_types (StateTypesTable)
│       │   └── state_name [VectorData/text]  ← all state names from first trial
│       ├── event_types (EventTypesTable)
│       │   └── event_name [VectorData/text]  ← all poke/event names from first trial
│       ├── action_types (ActionTypesTable)
│       │   └── action_name [VectorData/text] ← all wave names from first trial
│       └── task_arguments (TaskArgumentsTable)
│           ├── argument_name    [VectorData/text]
│           ├── argument_description [VectorData/text]
│           ├── expression       [VectorData/text]
│           ├── expression_type  [VectorData/text]  ← "integer","float","string","array","json"
│           └── output_type      [VectorData/text]  ← "numeric","text"
│   └── optogenetic_experiment_metadata (OptogeneticExperimentMetadata)  ← opto sessions only; ndx-optogenetics
│       ├── optogenetic_sites_table (OptogeneticSitesTable)
│       │   ├── row 0  optical_fiber=optical_fiber_left,  excitation_source=Cerebro, effector=ChR2
│       │   └── row 1  optical_fiber=optical_fiber_right, excitation_source=Cerebro, effector=ChR2
│       ├── optogenetic_effectors (OptogeneticEffectors)
│       │   └── ChR2 (Effector)                         ← Channelrhodopsin-2 opsin
│       ├── optogenetic_viruses (OptogeneticViruses)
│       │   └── aav_mdlx_chr2_mcherry (ViralVector)     ← AAV2/5-mDlx-ChR2-mCherry
│       └── optogenetic_virus_injections (OptogeneticVirusInjections)
│           ├── injection_fof_left (ViralVectorInjection)   ← FOF, +2 mm AP, -1.3 mm ML
│           └── injection_fof_right (ViralVectorInjection)  ← FOF, +2 mm AP, +1.3 mm ML
│
├── acquisition
│   └── task_recording (TaskRecording)        ← ndx-structured-behavior
│       ├── states (StatesTable)
│       │   ├── start_time  [VectorData/float64]  ← state enter time (seconds)
│       │   ├── stop_time   [VectorData/float64]  ← state exit time (seconds)
│       │   └── state_type  [DynamicTableRegion]  → StateTypesTable row index
│       ├── events (EventsTable)
│       │   ├── timestamp   [VectorData/float32]  ← poke enter time (seconds)
│       │   ├── duration    [VectorData/float32]  ← poke duration (exit - enter)
│       │   ├── value       [VectorData/text]     ← state name when poke started
│       │   └── event_type  [DynamicTableRegion]  → EventTypesTable row index
│       └── actions (ActionsTable)
│           ├── timestamp   [VectorData/float32]  ← wave onset time (seconds)
│           ├── duration    [VectorData/float32]  ← wave duration (offset - onset)
│           ├── value       [VectorData/text]     ← state name when wave started
│           └── action_type [DynamicTableRegion]  → ActionTypesTable row index
│
├── intervals
│   └── opto_epochs (OptogeneticEpochsTable)      ← opto sessions only; ndx-optogenetics
│       ├── start_time                ← cpoke_start + window_start (seconds)
│       ├── stop_time                 ← cpoke_start + window_stop (seconds)
│       ├── stimulation_on            ← True for all rows
│       ├── pulse_length_in_ms        ← window duration: 1300, 650, or 650 ms
│       ├── period_in_ms              ← same as pulse_length_in_ms (single pulse)
│       ├── number_pulses_per_pulse_train ← 1
│       ├── number_trains             ← 1
│       ├── intertrain_interval_in_ms ← NaN (not applicable for single pulse)
│       ├── power_in_mW               ← 25.0
│       ├── wavelength_in_nm          ← 450.0
│       └── optogenetic_sites [DynamicTableRegion] → OptogeneticSitesTable rows
│                             0 = optical_fiber_left, 1 = optical_fiber_right
│
└── trials (TrialsTable)                      ← ndx-structured-behavior
    ├── start_time     [VectorData/float64]   ← state_0 enter time
    ├── stop_time      [VectorData/float64]   ← state_0 exit time
    ├── states         [DynamicTableRegion]   → StatesTable rows within this trial
    ├── states_index   [VectorIndex]          ← ragged index for states
    ├── events         [DynamicTableRegion]   → EventsTable rows within this trial
    ├── events_index   [VectorIndex]          ← ragged index for events
    ├── actions        [DynamicTableRegion]   → ActionsTable rows within this trial
    ├── actions_index  [VectorIndex]          ← ragged index for actions
    │
    │   ── Stimulus scalar columns (TaskSwitch only) ──
    ├── gamma_dir      [VectorData/float64]   ← log ratio right/left pulse rates
    ├── gamma_freq     [VectorData/float64]   ← log ratio hi/lo pulse rates
    ├── duration       [VectorData/float64]   ← stimulus duration (seconds)
    ├── freq_lo        [VectorData/float64]   ← low-frequency tone (Hz)
    ├── freq_hi        [VectorData/float64]   ← high-frequency tone (Hz)
    ├── vol            [VectorData/float64]   ← overall volume multiplier
    ├── vol_low        [VectorData/float64]   ← low-freq volume multiplier
    ├── vol_hi         [VectorData/float64]   ← high-freq volume multiplier
    ├── bup_width      [VectorData/float64]   ← pulse duration (ms)
    ├── bup_ramp       [VectorData/float64]   ← pulse ramp duration (ms)
    ├── crosstalk_dir  [VectorData/float64]   ← spatial crosstalk (0–1)
    ├── crosstalk_freq [VectorData/float64]   ← frequency crosstalk (skipped if dict)
    │
    │   ── Pulse time ragged columns (TaskSwitch only) ──
    ├── cpoke_start_time [VectorData/float64]  ← absolute cpoke onset (NaN if rat didn't poke)
    ├── left_hi        [VectorData/float64 + VectorIndex]  ← relative pulse times (seconds from cpoke)
    ├── right_hi       [VectorData/float64 + VectorIndex]  ← relative pulse times (seconds from cpoke)
    ├── left_lo        [VectorData/float64 + VectorIndex]  ← relative pulse times (seconds from cpoke)
    ├── right_lo       [VectorData/float64 + VectorIndex]  ← relative pulse times (seconds from cpoke)
    │
    │   ── Optogenetics columns (opto sessions only) ──
    ├── OptoSection_opto_connected [VectorData/int]    ← 1 = Cerebro connected, 0 = not
    ├── OptoSection_opto_type      [VectorData/text]   ← 'Full Trial' / 'First Half' / 'Second Half'
    │   NOTE: opto_left_power / opto_right_power are NOT stored here — see §10 for rationale
    │         and how to recover per-trial hemisphere stimulation from opto_epochs.
    │
    └── {ProtocolParam}_* [VectorData]        ← per-trial task parameters from saved_history
```

---

## 3. BControl Field → NWB Location Mapping

### Session-level metadata (`saved`)

| BControl field | NWB location | Notes |
|---|---|---|
| `SavingSection_SaveTime` | `NWBFile.session_start_time` (date part) | Combined with "Started at" from protocol title |
| `ProtocolsSection_prot_title` | `NWBFile.session_start_time` (time part) | Regex `Started at HH:MM`; session excluded (ValueError) if not found |
| `CommentsSection_comments` | `NWBFile.notes` | Joined as newline-separated string |
| `CommentsSection_overall_comments` | `NWBFile.notes` (appended) | Prefixed with "Overall comments:" |

### Session-level parameters (`saved`, scalar types)

These become rows in `task.task_arguments` (TaskArgumentsTable):

| Python type | `expression_type` | `output_type` | Example fields |
|---|---|---|---|
| `int` | `"integer"` | `"numeric"` | `ProtocolsSection_n_completed_trials`, `RewardsSection_*` |
| `float` | `"float"` | `"numeric"` | `SoundSection_volume`, timing thresholds |
| `str` | `"string"` | `"text"` | `SavingSection_experimenter`, protocol name |
| `dict` | `"json"` | `"text"` | Any dict-valued parameter (JSON-dumped) |

### Session-level parameters (`saved`, array types)

The routing depends on array length relative to trial count:

| Condition | NWB location | Notes |
|---|---|---|
| `len == n_completed_trials` | `trials.{argument_name}` column | Per-trial column |
| `len == n_started_trials` and 1 incomplete | `trials.{argument_name}` column | Last entry trimmed |
| `len == n_completed_trials + 2` | `trials.{argument_name}` column | Warns, trims 2 extra |
| `len == n_completed_trials - 1` | `trials.{argument_name}` column | NaN appended |
| `dtype == object` (inhomogeneous/nested) | `task.task_arguments` | JSON-dumped, expression_type="json" |
| All-NaN array | Skipped | |

### Per-trial data (`saved_history`)

| BControl field | NWB location | Notes |
|---|---|---|
| `ProtocolsSection_parsed_events[i]["states"]` | `task_recording.states` (StatesTable) | All states sorted by start_time |
| `ProtocolsSection_parsed_events[i]["pokes"]` | `task_recording.events` (EventsTable) | Port pokes, sorted by timestamp |
| `ProtocolsSection_parsed_events[i]["waves"]` | `task_recording.actions` (ActionsTable) | Sounds/outputs, sorted by timestamp |
| `StimulusSection_ThisStimulus[i]` scalar params | `trials.*` columns | See stimulus scalar columns above |
| `StimulusSection_ThisStimulus[i].left_hi` | `trials.left_hi` (ragged) | Relative times in seconds from cpoke onset; add `cpoke_start_time` for absolute |
| `StimulusSection_ThisStimulus[i].right_hi` | `trials.right_hi` (ragged) | Same |
| `StimulusSection_ThisStimulus[i].left_lo` | `trials.left_lo` (ragged) | Same |
| `StimulusSection_ThisStimulus[i].right_lo` | `trials.right_lo` (ragged) | Same |
| `parsed_events[i].states.cpoke[0]` | `trials.cpoke_start_time` | Absolute cpoke onset (seconds); NaN for trials with no cpoke |
| `StimulusSection_ThisStimulus[i].freqs` | `trials.freq_lo`, `trials.freq_hi` | Split: freqs[0] → freq_lo, freqs[1] → freq_hi |

### Excluded fields

| BControl field | Reason for exclusion |
|---|---|
| `raw_events` | Raw (unparsed) event stream; redundant with parsed_events |
| `prepare_next_trial_set` | Internal BControl callback handle |
| `parsed_events` | Duplicate of `ProtocolsSection_parsed_events` |
| `current_assembler` | Internal assembler object reference |
| `my_gui`, `my_xyfig` | GUI handles |
| `PokesPlotSection`, `pokesplot` | GUI visualization state |
| `ThisStimulus` | Duplicate of `StimulusSection_ThisStimulus` |
| `SessionDefinition_my_gui_info` | GUI layout state |
| `SessionDefinition_last_change_tracker` | 1000-element zero array (GUI state) |
| `SessionDefinition_last_callback_placeholder` | GUI internal |
| `SessionDefinition_session_show` | GUI display flag |
| `SessionDefinition_display_type` | GUI display type |
| `SessionDefinition_show_*` | GUI show/hide flags |
| `SessionDefinition_btnbk_hdr` | GUI button header |
| `SessionDefinition_listblock`, `_paramblock` | GUI list widgets |
| `SessionDefinition_var_hdr`, `_var_names` | GUI variable headers |
| `SessionDefinition_list_hdr`, `_subparam_hdr` | GUI list headers |
| `SessionDefinition_last_change`, `_old_style_parsing_flag` | Internal BControl bookkeeping |
| `SoundInterface_*` (~150 keys) | Feedback-sound configuration (hit/error/timeout/violation tones); not scientifically relevant per lab |

### Protocol-specific fields (`SessionDefinition_*`, PBups only)

Present only in PBups files. Scientific fields are stored as task arguments:

| BControl field | NWB location | Type | Notes |
|---|---|---|---|
| `SessionDefinition_active_stage` | `task.task_arguments` | integer | Current training stage number |
| `SessionDefinition_nopoke_trials` | `task.task_arguments` | integer | Count of no-poke trials |
| `SessionDefinition_poke_trials` | `task.task_arguments` | integer | Count of poke trials |
| `SessionDefinition_training_stages` | `task.task_arguments` | json | Ragged nested list of stage defs (JSON-dumped) |
| Other `SessionDefinition_*` scalars | `task.task_arguments` | integer/float/string | Stored normally |

---

## 4. ndx-structured-behavior Schema Details

### StateTypesTable

One row per unique state name (populated from first trial's state dict).
All states with `np.ndarray` values are included; non-array entries are skipped.

| Column | Type | Description |
|---|---|---|
| `id` | int | Auto-incrementing row ID |
| `state_name` | text | Name of the state (e.g., `"state_0"`, `"cpoke"`, `"reward"`) |

### StatesTable (extends `TimeIntervals`)

One row per state occurrence across all trials, sorted by `start_time`.
States with NaN start times are excluded.

| Column | Type | Description |
|---|---|---|
| `id` | int | Auto-incrementing row ID |
| `start_time` | float64 | State enter time in seconds from session start |
| `stop_time` | float64 | State exit time in seconds from session start |
| `state_type` | DynamicTableRegion | Index into StateTypesTable |

Special handling for `state_0` (starting state): NaN-filtered; expected shape `(2,)` after filtering.
Other states: shape `(2,)` = single interval; shape `(N, 2)` = N intervals; shape `(2, 2)` with first row NaN = one valid interval.

### EventTypesTable

One row per unique event (poke) type, populated from first trial's pokes dict.

| Column | Type | Description |
|---|---|---|
| `id` | int | Auto-incrementing row ID |
| `event_name` | text | Port name (e.g., `"C"`, `"L"`, `"R"`) |

### EventsTable

One row per poke occurrence across all trials, sorted by `timestamp`.
Pokes with NaN start times are excluded.

| Column | Type | Description |
|---|---|---|
| `id` | int | Auto-incrementing row ID |
| `timestamp` | float32 | Poke enter time (seconds) |
| `duration` | float32 | Poke duration = exit_time - enter_time (seconds) |
| `value` | text | State name active when the poke started (`pokes["starting_state"][port]`) |
| `event_type` | DynamicTableRegion | Index into EventTypesTable |

### ActionTypesTable

One row per unique action (wave) type, populated from first trial's waves dict.

| Column | Type | Description |
|---|---|---|
| `id` | int | Auto-incrementing row ID |
| `action_name` | text | Wave name (e.g., `"base_sound"`, `"go_sound"`) |

### ActionsTable

One row per wave occurrence across all trials, sorted by `timestamp`.

| Column | Type | Description |
|---|---|---|
| `id` | int | Auto-incrementing row ID |
| `timestamp` | float32 | Wave onset time (seconds) |
| `duration` | float32 | Wave duration = offset - onset (seconds) |
| `value` | text | State name active when the wave started (`waves["starting_state"][wave]`) |
| `action_type` | DynamicTableRegion | Index into ActionTypesTable |

### TrialsTable (extends `TimeIntervals`)

One row per completed trial.

| Column | Type | Description |
|---|---|---|
| `id` | int | Auto-incrementing row ID |
| `start_time` | float64 | Trial start = state_0 enter time (seconds) |
| `stop_time` | float64 | Trial stop = state_0 exit time (seconds) |
| `states` | DynamicTableRegion + VectorIndex | Rows in StatesTable for this trial (by start/stop overlap) |
| `events` | DynamicTableRegion + VectorIndex | Rows in EventsTable for this trial (by timestamp in [start, stop]) |
| `actions` | DynamicTableRegion + VectorIndex | Rows in ActionsTable for this trial (by timestamp in [start, stop]) |
| `left_hi` | float64 + VectorIndex | Absolute times of left-speaker high-freq pulses |
| `right_hi` | float64 + VectorIndex | Absolute times of right-speaker high-freq pulses |
| `left_lo` | float64 + VectorIndex | Absolute times of left-speaker low-freq pulses |
| `right_lo` | float64 + VectorIndex | Absolute times of right-speaker low-freq pulses |
| `gamma_dir` | float64 | Log ratio right/left pulse rates |
| `gamma_freq` | float64 | Log ratio hi/lo pulse rates |
| `duration` | float64 | Stimulus duration (seconds) |
| `freq_lo` | float64 | Low-frequency tone (Hz) |
| `freq_hi` | float64 | High-frequency tone (Hz) |
| `vol` | float64 | Overall volume multiplier |
| `vol_low` | float64 | Low-freq volume multiplier |
| `vol_hi` | float64 | High-freq volume multiplier |
| `bup_width` | float64 | Pulse duration (ms) |
| `bup_ramp` | float64 | Pulse ramp duration (ms) |
| `crosstalk_dir` | float64 | Spatial crosstalk (0=equal, 1=right only) |
| `crosstalk_freq` | float64 | Frequency crosstalk |
| `{param}_*` | varies | Per-trial task parameters from `saved` arrays |

**Note:** Stimulus columns are absent in PBups files and present only in TaskSwitch files.

### TaskArgumentsTable

One row per session-level (scalar or short-array) parameter.

| Column | Type | Description |
|---|---|---|
| `id` | int | Auto-incrementing row ID |
| `argument_name` | text | BControl key (e.g., `"SoundSection_volume"`) |
| `argument_description` | text | From task_params.yaml if provided; else "no description" |
| `expression` | text | String representation of the value |
| `expression_type` | text | `"integer"`, `"float"`, `"string"`, `"array"`, or `"json"` |
| `output_type` | text | `"numeric"` or `"text"` |

---

## 5. Protocol-Specific Notes

### TaskSwitch2 / TaskSwitch4 / TaskSwitch6

- `StimulusSection_ThisStimulus` is present in `saved_history` → stimulus columns are added to `trials`.
- Per-trial pulse times (`left_hi`, `right_hi`, `left_lo`, `right_lo`) are stored as ragged columns.
- Pulse times are stored as **relative times in seconds from cpoke onset** (exactly as in the BControl file) for all trials.  To convert to absolute session time: `abs_time = cpoke_start_time + pulse_time`.
- `cpoke_start_time` is a dedicated trial column (float64, NaN when the rat did not poke). For error/invalid trials the relative pulse times are still stored because the stimulus was generated and the data is scientifically valuable.
- `crosstalk_freq` is sometimes a dict `{'lpos': 359}` rather than a scalar → skipped with a warning.
- `freqs` column (if present) is split into `freq_lo` and `freq_hi` before adding to trials.

### PBups

- No `StimulusSection_ThisStimulus` → no stimulus columns added.
- `SessionDefinition_*` fields are present; GUI-state fields are excluded (see exclusion list above).
- Scientific `SessionDefinition_*` fields (stage number, trial counts, training stages) are stored in `task.task_arguments`.

### Session start time extraction (all protocols)

**Decision (confirmed with Marino, 2026-03-13):** Sessions whose `ProtocolsSection_prot_title`
does not contain the substring `"Started at"` are excluded from the conversion. These are old
BControl files from early training sessions that lack reliable start-time metadata and are not
scientifically relevant. No fallback to `"Ended at"` or `SavingSection_SaveTime` is used.

```
protocol_title = saved["ProtocolsSection_prot_title"]
                  ↓
Search "Started at HH:MM"   →  success
                  ↓ (not found)
raise ValueError / skip session  ← no fallback; session is excluded
```

`dataset_to_nwb` catches the `ValueError` and writes it to the error log (`conversion_errors.log`),
so skipped sessions are traceable.

If "Started at HH:MM" is found:

```
date from SavingSection_SaveTime + "Started at" time → datetime.strptime → session_start_time
                  ↓
replace(tzinfo=ZoneInfo("Europe/London"))   ← hard-coded timezone
```

---

## 6. Reading Data Back from NWB

```python
from pynwb import NWBHDF5IO

with NWBHDF5IO("sub-H7015_ses-250516a.nwb") as io:
    nwb = io.read()

    # --- Session metadata ---
    print(nwb.session_start_time)
    print(nwb.notes)                    # experimenter comments

    # --- Subject ---
    print(nwb.subject.subject_id)

    # --- Trials table ---
    trials = nwb.trials.to_dataframe()
    print(trials.head())
    print(trials.columns.tolist())

    # --- Stimulus scalar columns ---
    print(trials[["gamma_dir", "gamma_freq", "duration", "freq_lo", "freq_hi"]])

    # --- Ragged pulse-time columns ---
    # Each entry is a list of pulse times (seconds from session start)
    left_hi_col = nwb.trials["left_hi"]
    left_hi_trial_0 = left_hi_col[0]   # list of floats for trial 0

    # --- States table ---
    task_recording = nwb.acquisition["task_recording"]
    states = task_recording.states.to_dataframe()
    print(states[["start_time", "stop_time"]].head())

    # To resolve state_type index back to name:
    state_types = nwb.lab_meta_data["task"].state_types.to_dataframe()
    states["state_name"] = states["state_type"].apply(lambda i: state_types.loc[i, "state_name"])

    # --- Events (pokes) table ---
    events = task_recording.events.to_dataframe()
    event_types = nwb.lab_meta_data["task"].event_types.to_dataframe()
    events["port"] = events["event_type"].apply(lambda i: event_types.loc[i, "event_name"])

    # --- Actions (waves/sounds) table ---
    actions = task_recording.actions.to_dataframe()
    action_types = nwb.lab_meta_data["task"].action_types.to_dataframe()
    actions["wave"] = actions["action_type"].apply(lambda i: action_types.loc[i, "action_name"])

    # --- Task arguments ---
    task_args = nwb.lab_meta_data["task"].task_arguments.to_dataframe()
    print(task_args[["argument_name", "expression", "expression_type"]])

    # Parse a JSON argument (e.g., SessionDefinition_training_stages):
    import json
    row = task_args[task_args["argument_name"] == "SessionDefinition_training_stages"].iloc[0]
    training_stages = json.loads(row["expression"])

    # --- States for a specific trial ---
    trial_idx = 5
    states_for_trial = nwb.trials["states"][trial_idx]  # list of StatesTable row indices
    trial_states = states.iloc[states_for_trial]
    print(trial_states)
```

---

## 7. Conversion Entry Points

### Single session

```python
from pagan_lab_to_nwb.arc_behavior.convert_session import session_to_nwb

nwb_path = session_to_nwb(
    file_path="data_@TaskSwitch6_Nuria_H7015_250516a.mat",
    nwb_folder_path="/path/to/output",
    task_params_file_path="task_switch6_params.yaml",  # optional YAML with param descriptions
    stub_test=False,
    overwrite=True,
)
```

### All sessions

```python
from pagan_lab_to_nwb.arc_behavior.convert_all_sessions import dataset_to_nwb

dataset_to_nwb(
    data_folder_path="/path/to/data",
    nwb_folder_path="/path/to/output",
    task_params_file_path="task_switch6_params.yaml",
    stub_test=False,
    overwrite=False,
)
```

### Task parameter YAML

The canonical YAML is checked into the repo at `arc_behavior/task_switch6_params.yaml`.
It is auto-loaded by `convert_session.py` and `convert_all_sessions.py`; no path argument is needed.

To regenerate it from MATLAB protocol code (e.g. after BControl updates), run:
```bash
python src/pagan_lab_to_nwb/arc_behavior/parse_mat_code.py
```
This reads `*.m` files from the local Protocol_code folder and writes the output back to
`arc_behavior/task_switch6_params.yaml`. Human-curated descriptions (e.g. for `_history`
columns) must be re-applied afterwards. See `utils/notes.md` for the full workflow.

---

## 8. Key Implementation Files

| File | Purpose |
|---|---|
| `interfaces/bcontroldatainterface.py` | Main data interface; all BControl→NWB conversion logic |
| `arc_behavior/convert_session.py` | Single-session entry point; handles filename parsing, timezone, metadata merging |
| `arc_behavior/convert_all_sessions.py` | Batch conversion; error handling, progress tracking |
| `arc_behavior/metadata.yaml` | Editable session/subject metadata and table descriptions |
| `arc_behavior/utils/utils.py` | `get_description_from_arguments_metadata` for YAML-based parameter docs |
| `arc_behavior/utils/notes.md` | Instructions for generating `task_params.yaml` from MATLAB code |

---

## 9. ndx-structured-behavior: DynamicTableRegion Pattern

The type tables (`StateTypesTable`, `EventTypesTable`, `ActionTypesTable`) act as vocabularies —
like foreign-key lookup tables. The data tables reference them via integer indices stored in
`DynamicTableRegion` columns.

```
StateTypesTable         StatesTable
┌────┬───────────┐     ┌────┬────────────┬───────────┬────────────┐
│ id │ state_name│     │ id │ start_time │ stop_time │ state_type │
├────┼───────────┤     ├────┼────────────┼───────────┼────────────┤
│  0 │ state_0   │◄────│  0 │  0.000     │  0.003    │     0      │
│  1 │ wait_poke │◄────│  1 │  0.003     │  0.412    │     1      │
│  2 │ cpoke     │◄────│  2 │  0.412     │  1.715    │     2      │
│  3 │ reward    │◄────│  3 │  1.715     │  1.935    │     3      │
└────┴───────────┘     └────┴────────────┴───────────┴────────────┘
```

`TrialsTable.states` is also a `DynamicTableRegion` pointing into `StatesTable`, with a `VectorIndex`
making it ragged (each trial references a variable number of state rows).

```
TrialsTable                    StatesTable
┌────┬────────────┬──────────┬─────────────────┐
│ id │ start_time │ stop_time│ states (region) │
├────┼────────────┼──────────┼─────────────────┤
│  0 │  0.000     │  1.935   │ [0, 1, 2, 3]    │──► rows 0-3 of StatesTable
│  1 │  1.935     │  3.780   │ [4, 5, 6, 7, 8] │──► rows 4-8 of StatesTable
└────┴────────────┴──────────┴─────────────────┘
```

---

## References

- [ndx-structured-behavior](https://github.com/rly/ndx-structured-behavior)
- [NeuroConv](https://github.com/catalystneuro/neuroconv)
- [NWB (Neurodata Without Borders)](https://www.nwb.org/)

---

## 10. Design Decisions & Resolved Questions

### Optogenetics: power unit conversion (RESOLVED 2026-03-16)

**Answer from Marino:** The Cerebro internal unit does not have a fixed linear conversion to
mW — the threshold value varied session-to-session because the laser was glued to the fiber
optic implant and the exact output could not be measured precisely. The experimenters
identified a session-specific threshold for 25 mW output.

**Rule implemented:**
- Raw Cerebro value **>= 800** → laser was on → **25 mW (0.025 W)**
- Raw Cerebro value **< 800** → laser was off → **0 mW (0 W)**

**Implementation in `interfaces/_optogenetics.py`:**
- Per-trial stimulation timing and hemisphere are stored in `intervals["opto_epochs"]`
  (`OptogeneticEpochsTable`, ndx-optogenetics). One row per stimulated trial.
- Rich hardware metadata (fiber specs, virus, injection coordinates) is stored in
  `lab_meta_data["optogenetic_experiment_metadata"]` (`OptogeneticExperimentMetadata`).

---

### Optogenetics: why `opto_left/right_power` are NOT trials columns (DECIDED 2026-03-16)

**Decision:** `OptoSection_opto_left_power` and `OptoSection_opto_right_power` (raw Cerebro
internal units) are **not** stored as trials table columns.

**Rationale:** Given the binary conversion rule (raw >= 800 → 25 mW, else 0 mW), the raw
values carry no information beyond what is already in `intervals["opto_epochs"]`:

| Information | Trials columns | `opto_epochs` (OptogeneticEpochsTable) |
|---|---|---|
| Was the laser on this trial? | `opto_connected` ✓ | derivable from row presence |
| Which hemisphere was stimulated? | (removed) | ✓ via `optogenetic_sites` region |
| Power in physical units | (removed) | ✓ `power_in_mW` = 25.0 |
| Stimulation window type | `opto_type` ✓ | implicit in `start_time`/`stop_time` |
| Control trials (no stimulation) | `opto_connected` ✓ | **not encoded** (no row) |

`opto_epochs` already encodes, for each stimulated trial, exactly which hemisphere(s) fired
and at what power. Adding the raw Cerebro values alongside would be redundant and potentially
confusing (two representations of the same binary fact in different units).

**What IS kept in the trials table:**
- `OptoSection_opto_connected` — the only record of which trials were control (Cerebro
  disconnected) vs. stimulated within an opto session. Not derivable from the time-series.
- `OptoSection_opto_type` — the stimulation window label (`'Full Trial'`, `'First Half'`,
  `'Second Half'`) as an explicit per-trial string, convenient for groupby/filtering.

---

### How to recover per-trial hemisphere stimulation from `opto_epochs`

`intervals["opto_epochs"]` (`OptogeneticEpochsTable`, ndx-optogenetics) has one row per
stimulated trial. The `optogenetic_sites` column is a `DynamicTableRegion` pointing into
`OptogeneticSitesTable` (row 0 = left fiber, row 1 = right fiber).

```python
import numpy as np
import pandas as pd
import ndx_structured_behavior  # must import before NWBHDF5IO.read()
from pynwb import NWBHDF5IO

with NWBHDF5IO("sub-P131_ses-TaskSwitch6-190815a.nwb", "r") as io:
    nwb = io.read()

    trials = nwb.trials.to_dataframe()[["start_time", "stop_time",
                                        "OptoSection_opto_connected",
                                        "OptoSection_opto_type"]]

    # ── Per-trial hemisphere stimulation from OptogeneticEpochsTable ─────────
    opto_epochs = nwb.intervals["opto_epochs"].to_dataframe()

    # optogenetic_sites is a list of site indices per epoch:
    #   [0] = left FOF only, [1] = right FOF only, [0, 1] = both hemispheres
    opto_epochs["stim_left"]  = opto_epochs["optogenetic_sites"].apply(lambda s: 0 in s)
    opto_epochs["stim_right"] = opto_epochs["optogenetic_sites"].apply(lambda s: 1 in s)

    # Join epochs back to trials by matching epoch start_time to the trial interval
    trials["stim_left"]  = False
    trials["stim_right"] = False
    for _, epoch in opto_epochs.iterrows():
        in_trial = ((trials["start_time"] <= epoch["start_time"]) &
                    (epoch["start_time"]   < trials["stop_time"]))
        trials.loc[in_trial, "stim_left"]  |= epoch["stim_left"]
        trials.loc[in_trial, "stim_right"] |= epoch["stim_right"]

    # ── Result ───────────────────────────────────────────────────────────────
    # trials["stim_left"]  → True if left FOF was stimulated on this trial
    # trials["stim_right"] → True if right FOF was stimulated on this trial
    # Trials with opto_connected == 0 have no epoch row → stim_left/right stay False
    print(trials[["OptoSection_opto_connected", "OptoSection_opto_type",
                  "stim_left", "stim_right"]].value_counts())
```

**Notes:**
- Trials with `opto_connected == 0` have no row in `opto_epochs` and will have
  `stim_left = stim_right = False` by construction.
- Stimulation duration per trial: `epoch["stop_time"] - epoch["start_time"]` (seconds),
  or read `pulse_length_in_ms` directly from the epoch row.
- Rich stimulation metadata per epoch (wavelength, power, pulse count) is in the
  `opto_epochs` columns — no need to inspect device objects for these values.
