# arc_ecephys — Data Manifest

Field-by-field map from each source file to its NWB destination.
See `README.md` for the overall pipeline description.

---

## BControl behavior (`data_@*.mat`) → `BControlBehaviorInterface`

### Task structure (`lab_meta_data["task"]`, ndx-structured-behavior)

| Source field | NWB destination | Notes |
|---|---|---|
| `saved_history["ProtocolsSection_parsed_events"][i]["states"]` | `StatesTable` | One row per (trial, state); `start_time`/`stop_time` in seconds |
| `saved_history["ProtocolsSection_parsed_events"][i]["pokes"]` | `EventsTable` | Port poke events (C/L/R); one row per event |
| `saved_history["ProtocolsSection_parsed_events"][i]["waves"]` | `ActionsTable` | Reward + timer actions |
| Protocol state names | `StateTypesTable` | Unique FSM state names |
| Protocol event names | `EventTypesTable` | Unique port event names (C, L, R) |
| Protocol action names | `ActionTypesTable` | Unique action names |
| `saved["*Section_*"]` parameters | `TaskArgumentsTable` | Session-level protocol parameters |

### Trials (`nwbfile.trials`)

| Source field | NWB column | Notes |
|---|---|---|
| `parsed_events[i]["states"]["state_0"][0][1]` | `start_time` | Trial start (state_0 entry) |
| `parsed_events[i]["states"]["state_0"][1][0]` | `stop_time` | Trial end (state_0 exit) |
| `saved_history["StimulusSection_ThisStimulus"][i]["gamma_dir"]` | `gamma_dir` | Direction evidence log-ratio |
| `saved_history["StimulusSection_ThisStimulus"][i]["gamma_freq"]` | `gamma_freq` | Frequency evidence log-ratio |
| `saved_history["StimulusSection_ThisStimulus"][i]["duration"]` | `duration` | Stimulus duration (s) |
| `saved_history["StimulusSection_ThisStimulus"][i]["freq_lo/hi"]` | `freq_lo`, `freq_hi` | Pulse frequencies (Hz) |
| `saved_history["StimulusSection_ThisStimulus"][i]["left_hi/lo"]` | `left_hi`, `left_lo` | Pulse times relative to cpoke (s), ragged |
| `saved_history["StimulusSection_ThisStimulus"][i]["right_hi/lo"]` | `right_hi`, `right_lo` | Pulse times relative to cpoke (s), ragged |
| Parsed cpoke state entry time | `cpoke_start_time` | Absolute session time of cpoke onset (NaN if no poke) |
| Other `StimulusSection_ThisStimulus` scalars | `crosstalk_dir`, `crosstalk_freq`, `bup_width`, `bup_ramp`, `vol_low`, `vol_hi`, `vol` | Stimulus parameters |
| `saved_history["HistorySection_*"]` | `HistorySection_*` columns | Trial history (hits, sides, contexts, quadrants) |

### Session metadata (`nwbfile`)

| Source field | NWB field | Notes |
|---|---|---|
| `saved["*prot_title*"]` (regex `Started at HH:MM`) | `session_start_time` | Combined with `SavingSection_SaveTime` for date |
| `saved["CommentsSection_comments"]` | `notes` | Experimenter session notes |

---

## Raw ephys recording (`*.rec`) → `SpyglassSpikeGadgetsRecordingInterface`

> **Status: implemented but untested** — no `.rec` files are available yet (see
> `open_questions.md` Q5). Optional; only runs when `spikegadgets_file_path` is passed to
> `session_to_nwb()`. Must run before `SpikeSortingMatInterface` so the electrode table is
> shared between the two.

Source format: SpikeGadgets `.rec` (XML `<Configuration>` header + binary trace data), read
via NeuroConv's `SpikeGadgetsRecordingInterface`.

### Devices (`nwbfile.devices`)

| Source | NWB object | Metadata source |
|---|---|---|
| — | `DataAcqDevice("HH128")` | `metadata/_spike_sorting_mat_metadata.yaml` → `Ecephys.DataAcqDevice` (shared with `SpikeSortingMatInterface`) |
| — | `Probe("tetrode_array")` → `Shank("0")` → `ShanksElectrode("0".."3")` | `metadata/_spike_sorting_mat_metadata.yaml` → `Ecephys.Probe` (shared) |

### Electrode groups and electrodes

| Source | NWB destination | Notes |
|---|---|---|
| `.rec` header `<SpikeNTrode id=...>` → `<SpikeChannel hwChan=...>` | `NwbElectrodeGroup("tetrode{N}")` per `SpikeNTrode/@id` | One group per tetrode; `location="unknown"` **⚠ placeholder** |
| Hardware-channel → (tetrode, intra-tetrode index) map parsed from the header | `nwbfile.electrodes` rows (one per recorded channel) | `probe_shank=0`, `probe_electrode=0..3`, `bad_channel=False`, `ref_elect_id=-1` |

### Acquisition (`nwbfile.acquisition`)

| Source | NWB destination | Notes |
|---|---|---|
| Raw int16 traces, all channels | `ElectricalSeriesRaw` (`ElectricalSeries`) | `conversion` = channel gain (µV/count × 1e-6); chunked + gzip via `DataChunkIterator`/`H5DataIO`; explicit `timestamps` (Spyglass requires `always_write_timestamps=True`) |
| Recording duration | `nwbfile.epochs` `[0.0, duration]`, `tags=["01"]` | Only added if `nwbfile.epochs` is still empty (shared with `SpikeSortingMatInterface`) |

**Interaction with `SpikeSortingMatInterface`:** when both interfaces run, `SpikeGadgets` runs
first and creates the electrode table; `SpikeSorting` detects the pre-existing rows (matched
via `tetrode{N}` electrode-group naming) and maps sorted units onto them instead of adding
duplicate rows.

---

## Spike sorting (`spikes_@*.mat`) → `SpikeSortingMatInterface`

Source format: MATLAB v7.3 / HDF5. All datasets accessed via h5py.

### Devices (`nwbfile.devices`)

| Source | NWB object | Metadata source |
|---|---|---|
| — | `DataAcqDevice("HH128")` | `metadata/_spike_sorting_mat_metadata.yaml` → `Ecephys.DataAcqDevice` |
| — | `Probe("tetrode_array")` → `Shank("0")` → `ShanksElectrode("0".."3")` | `metadata/_spike_sorting_mat_metadata.yaml` → `Ecephys.Probe` |

### Electrode groups and electrodes

| Source | NWB destination | Notes |
|---|---|---|
| `f["trode"][:]` unique values | `NwbElectrodeGroup("tetrode{N}")` | One group per tetrode; `location="unknown"` **⚠ placeholder** |
| 4 channels per tetrode | `nwbfile.electrodes` rows | `probe_shank=0`, `probe_electrode=0..3`, `bad_channel=False`, `ref_elect_id=-1` |

### Units (`nwbfile.units`)

| HDF5 dataset | NWB column | Shape / notes |
|---|---|---|
| `f["spikes"][i, 0]` (object ref) | `spike_times` | Ragged array of spike timestamps (s) per unit |
| `f["wave"][i, 0]` (object ref) | `waveform_mean` | `(61, 4)` float32 — mean waveform, 61 samples × 4 channels |
| `f["wavestd"][i, 0]` (object ref) | `waveform_sd` | `(61, 4)` float32 — SD of waveforms |
| `f["trode"][i]` | `trode_id` | Tetrode ID (1-indexed) |

### Time intervals

| Source | NWB destination | Notes |
|---|---|---|
| `f["goodp"][:]` | `nwbfile.time_intervals["goodp"]` | `[start_sec, end_sec]` — usable recording window from spike sorter |
| `max(spike_times[-1])` | `nwbfile.epochs` `[0.0, max_spike]` | One epoch per session; `tags=["01"]` |

---

## Processed trials (`dati_*.mat`) → `ProcessedTrialsInterface`

Source format: MATLAB v5. Read with `pymatreader`.

Writes to: `processing["behavior"]["processed_trials"]` (PyNWB `TimeIntervals`).

`start_time` = `tim[1]` (trial_ready), `stop_time` = `tim[6]` (trial_end).
Timestamps use the dati clock, which has an ~10 ms offset from the BControl clock.
Use `cue_start` for cross-stream alignment.

### Behavioral columns

| MATLAB variable | NWB column | Type | Notes |
|---|---|---|---|
| `d["choice"]` | `choice` | float | 0.0 = left, 1.0 = right, NaN = unrecorded |
| `d["hits"]` | `hits` | float | 1.0 = correct, 0.0 = incorrect |
| `d["nta"]` | `nta` | float | Position within contextual block (1-indexed) |
| `d["side"]` (`"l"`/`"r"`) | `correct_side` | str | `"Left"` or `"Right"` |
| `d["task"]` (`"d"`/`"f"`) | `task_context` | str | `"Direction"` or `"Frequency"` |
| `d["gdir"]` | `gdir` | float | Direction evidence log-ratio |
| `d["gfreq"]` | `gfreq` | float | Frequency evidence log-ratio |
| `d["stim"]` (list of dicts) | `stim_params` | str | JSON-serialised per-trial stimulus parameters |

### Event timestamp columns (from `tim` matrix, row-indexed)

| `tim` row | NWB column | Description |
|---|---|---|
| 0 | `previous_trial_end` | End of previous trial (s) |
| 1 | `start_time` | Trial-ready timestamp (used as interval start) |
| 2 | `cue_start` | Auditory cue onset (s) |
| 3 | `poke_in` | Centre-port poke-in (s) |
| 4 | `poke_out` | Centre-port poke-out (s) |
| 5 | `choice_time` | Choice timestamp (s) |
| 6 | `stop_time` | Trial end (used as interval stop) |

### PSTH (written inline by `ProcessedTrialsInterface.add_to_nwbfile`)

`rrr4` is written as a standard `TimeSeries` during the same `run_conversion()` call
as the rest of the dati data — no post-write step required.

| MATLAB variable | NWB destination | Notes |
|---|---|---|
| `d["rrr4"]` transposed | `processing["ecephys"]["rrr4_psth"]` (TimeSeries) | Shape `(n_trials, n_units, n_bins)` float32, unit `spikes/s`; axis 0 = trial, axis 1 = unit, axis 2 = time bin |
| `d["centers4"]` | Stored in description string | Bin width and time range (−2 to +3 s re cue onset) documented in `TimeSeries.description` |
| `tim[1, :]` (trial_ready) | `TimeSeries.timestamps` | Monotonically increasing per-trial timestamps; PSTH bins are relative to cue_start within each trial |

The `processed_trials` TimeIntervals table gains a built-in **`timeseries`** column
(type `TimeSeriesReferenceVectorData`): each row i holds a `TimeSeriesReference`
with `idx_start=i, count=1` pointing to row i of `rrr4_psth`, so both tables
remain aligned by trial index.

Access in Python:
```python
# Read PSTH for all trials
psth_ts = nwbf.processing["ecephys"]["rrr4_psth"]
data = psth_ts.data[:]           # (n_trials, n_units, n_bins), spikes/s
timestamps = psth_ts.timestamps[:]  # trial_ready times (s)

# Reconstruct bin centres (10 ms bins, −2 to +3 s re cue onset)
bin_centres = -2.0 + np.arange(data.shape[2]) * 0.010  # seconds

# Access via processed_trials timeseries column (returns same TimeSeries)
ts_refs = nwbf.processing["behavior"]["processed_trials"]["timeseries"][i]
# ts_refs[0].timeseries is rrr4_psth; ts_refs[0].idx_start == i, count == 1
```

---

## Behavioral video (`video_@*.mp4`) → `SpyglassVideoInterface`

| Source | NWB destination | Notes |
|---|---|---|
| MP4 file path | `processing["behavior"]["video"]` → `BehavioralEvents` → `ImageSeries.external_file` | File is not embedded; only the path is stored |
| Nominal frame rate (~19.98 fps) | `ImageSeries.timestamps` | Uniform timestamps; no hardware sync signal available |
| `metadata/_video_metadata.yaml` | `nwbfile.devices["camera_device 1"]` (CameraDevice) | `camera_name`, `model`, `lens`, `meters_per_pixel` |

---

## Spyglass custom table (`ProcessedTrials`, schema `behavior_pagan`)

Defined in `spyglass_extensions/spyglass_processed_trials_table.py`.
Stores a pointer (NWB `object_id`) to the `processed_trials` TimeIntervals table
rather than duplicating all columns into DataJoint. Data is read back via `fetch_nwb()`.

| DataJoint column | Value | Notes |
|---|---|---|
| `nwb_file_name` | Spyglass copy filename (`…_.nwb`) | FK → `Nwbfile` |
| `processed_trials_object_id` | UUID from `processed_trials.object_id` | Resolved by `fetch_nwb()` |

**Usage:**
```python
df = (ProcessedTrials() & {"nwb_file_name": copy_name}).fetch1_dataframe()
# Returns the full processed_trials TimeIntervals as a pandas DataFrame
```
