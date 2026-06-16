# arc_ecephys — Conversion Documentation

This folder documents the `arc_ecephys` pipeline, which converts Pagan Lab
electrophysiology sessions to NWB and inserts them into a Spyglass/DataJoint
database.

> **Prototype status.** This pipeline was built on two example sessions provided
> by the lab (`P100` — ephys+behavior, `P267` — video+behavior). Several metadata
> fields are currently placeholders that must be filled in by the lab before a
> production DANDI upload. See [Placeholder Fields](#placeholder-fields-requiring-lab-confirmation)
> for a prioritised list.

---

## Files

| File | Role |
|---|---|
| `convert_session.py` | Converts one session (behavior + optional spikes/dati/video) to NWB. Edit the `__main__` block or call `session_to_nwb()` directly. |
| `nwbconverter.py` | `ArcEcephysNWBConverter` — wires the four interfaces together. |
| `insert_session.py` | Inserts a converted NWB file into the Spyglass database. Also seeds lookup tables, populates BControl / spike-sorting / processed-trials tables, and runs verification tests. |
| `metadata.yaml` | Session-level metadata only: `NWBFile` (experiment description, institution, lab) and `Subject` (species, strain). **Edit here** when experiment-level details change. |
| `dj_local_conf.json` | DataJoint connection config (host, user, password, Spyglass dirs). **Gitignored** — copy `dj_local_conf.example.json` and fill in your paths (see [Spyglass setup guide](../../spyglass_mock/README.md)). |
| `dj_local_conf.example.json` | Template DataJoint config with placeholder paths. |
| `documentation/README.md` | This file. |
| `documentation/data_manifest.md` | Field-by-field map from source files to NWB destinations. |
| `documentation/open_questions.md` | Full list of every open question sent to the lab, with resolution status and action taken. |
| `documentation/spyglass_notes.md` | Notes from the Spyglass insertion phase: errors encountered, fixes applied, expected warnings, and phase progress log. |

---

## Data Streams

Each stream is an independent, optional interface. Omit a file path in `session_to_nwb()`
to skip that stream.

| Stream | Source format | File pattern | NWB destination | Converter key |
|---|---|---|---|---|
| BControl behavior | MATLAB v5 `.mat` | `data_@{protocol}_{exp}_{sub}_{date}.mat` | `nwbfile.trials`, `lab_meta_data["task"]`, optogenetics | `Behavior` |
| Raw ephys recording | SpikeGadgets `.rec` | `*.rec` | `nwbfile.acquisition["ElectricalSeriesRaw"]`, `nwbfile.electrodes`, `nwbfile.electrode_groups`, devices | `SpikeGadgets` |
| Spike sorting | MATLAB v7.3 (HDF5) `.mat` | `spikes_@{protocol}_{exp}_{sub}_{date}.mat` | `nwbfile.units`, `nwbfile.electrodes`, `nwbfile.electrode_groups` | `SpikeSorting` |
| Processed trials | MATLAB v5 `.mat` | `dati_{protocol}_{exp}_{sub}_{date}.mat` | `processing["behavior"]["processed_trials"]` (TimeIntervals) | `ProcessedTrials` |
| Behavioral video | MP4 | `video_@{protocol}_{exp}_{sub}_{date}.mp4` | `processing["behavior"]["video"]` (ImageSeries) | `Video` |

> **Status:** the `SpikeGadgets` stream (`SpyglassSpikeGadgetsRecordingInterface`,
> `spikegadgets_file_path`) is implemented but **untested end-to-end** — no `.rec`
> files are available yet (see `open_questions.md` Q5).

**Interface order matters** (enforced by `ArcEcephysNWBConverter`):
1. `Behavior` — creates `nwbfile.trials` (required by `ProcessedTrials`)
2. `SpikeGadgets` *(optional)* — writes raw `ElectricalSeriesRaw` to acquisition and creates
   the `DataAcqDevice`/`Probe`/`NwbElectrodeGroup` hierarchy plus the electrode table.
   Must run before `SpikeSorting` so the electrode table can be shared.
3. `SpikeSorting` — creates `nwbfile.units` and the `behavior` processing module; reuses the
   electrode table from `SpikeGadgets` if present, otherwise builds its own electrodes,
   `Probe`/`DataAcqDevice` devices, and electrode groups
4. `ProcessedTrials` — appends columns to the behavior processing module
5. `Video` — adds `CameraDevice` and `ImageSeries` to the behavior processing module

---

## NWB File Structure

```
nwbfile
├── session_description        "This session contains ..."
├── lab_meta_data["task"]      ndx-structured-behavior Task
│                              (StateTypes, EventTypes, ActionTypes, TaskArguments)
├── acquisition
│   └── ElectricalSeriesRaw    Raw broadband signal, all tetrode channels  [SpikeGadgets, optional]
├── devices
│   ├── HH128                  DataAcqDevice (SpikeGadgets HH128)         [SpikeGadgets/SpikeSorting]
│   ├── tetrode_array          Probe → Shank → ShanksElectrode × 4       [SpikeGadgets/SpikeSorting]
│   └── camera_device 1        CameraDevice (top_camera)                  [Video]
├── electrode_groups           NwbElectrodeGroup per tetrode (tetrode{N})  [SpikeGadgets/SpikeSorting]
├── electrodes                 1 row per recorded channel; probe_shank, probe_electrode,
│                              bad_channel, ref_elect_id                   [SpikeGadgets/SpikeSorting]
├── units                      spike_times, waveform_mean, waveform_sd,
│                              trode_id per sorted unit                    [SpikeSorting]
├── trials                     BControl trial structure (from Behavior),
│                              plus BControl stimulus columns
├── epochs                     One interval [0, max_spike_time]           [SpikeGadgets/SpikeSorting]
├── time_intervals["goodp"]    Usable recording window from spike sorter  [SpikeSorting]
└── processing
    ├── behavior
    │   ├── task_recording     ndx-structured-behavior TaskRecording
    │   │                      (states, events, actions)
    │   ├── processed_trials   TimeIntervals — dati behavioral variables  [ProcessedTrials]
    │   │                      (choice, hits, nta, task_context, gdir,
    │   │                       gfreq, tim event timestamps, stim_params)
    │   └── video              BehavioralEvents → ImageSeries (MP4)       [Video]
    ├── ecephys
    │   └── rrr4_psth          TimeSeries — PSTH aligned to cue onset
    │                          shape (n_trials, n_units, n_bins), spikes/s [ProcessedTrials]
    └── tasks                  DynamicTable with task_name, task_type,
                               camera_id, task_epochs                     [SpikeSorting]
```

---

## Running a Conversion

```python
from pagan_lab_to_nwb.arc_ecephys.convert_session import session_to_nwb

nwb_path = session_to_nwb(
    behavior_file_path="/path/to/data_@TaskSwitch4_Marino_P100_181010a.mat",
    nwb_folder_path="/path/to/output/",
    spikes_file_path="/path/to/spikes_@TaskSwitch4_Marino_P100_181010a.mat",  # optional
    dati_file_path="/path/to/dati_TaskSwitch4_Marino_P100_181010a.mat",       # optional
    video_file_path="/path/to/video_@TaskSwitch6_Marino_P267_221211a.mp4",    # optional
    stub_test=False,
    overwrite=True,
)
```

The NWB file is written to `nwb_folder_path/sub-{subject_id}/sub-{subject_id}_ses-{session_id}.nwb`.

---

## Spyglass Insertion

Before inserting, copy the NWB file to the Spyglass raw directory (`/Volumes/T9/data/Pagan/raw/`).
Then run from the `arc_ecephys/` directory (so `dj_local_conf.json` is found automatically):

```bash
cd src/pagan_lab_to_nwb/arc_ecephys
python insert_session.py
```

Or call the function directly:

```python
from pagan_lab_to_nwb.arc_ecephys.insert_session import insert_session
insert_session(nwbfile_path="/Volumes/T9/data/Pagan/raw/sub-P100_ses-TaskSwitch4-181010a.nwb",
               clean_existing=True)
```

`clean_existing=True` deletes any prior DB entries for the session before re-inserting — use this
whenever you need to re-run after a partial or failed insertion.

### Spyglass tables populated

| Table | Source | Session |
|---|---|---|
| `Nwbfile`, `Session`, `Subject` | NWB metadata | Both |
| `DataAcquisitionDevice` | `devices["HH128"]` (DataAcqDevice) | P100 |
| `ProbeType`, `Probe` | `devices["tetrode_array"]` (Probe hierarchy) | P100 |
| `ElectrodeGroup`, `Electrode` | `electrode_groups`, `electrodes` | P100 |
| `IntervalList` | `epochs` | Both |
| `TaskEpoch` | `processing["tasks"]` DynamicTable | Both |
| `CameraDevice`, `VideoFile` | `devices["camera_device 1"]`, ImageSeries | P267 |
| `TaskRecordingTypes`, `TaskRecording` | `lab_meta_data["task"]` (ndx-structured-behavior) | Both |
| `ImportedSpikeSorting`, `SpikeSortingOutput` | `nwbfile.units` | P100 |
| `SortedSpikesGroup` | created from ImportedSpikeSorting | P100 |
| `UnitAnnotation` | `trode_id` per unit | P100 |
| `ProcessedTrials` (custom) | `processing["behavior"]["processed_trials"]` | P100 |
| `Raw` | `acquisition["ElectricalSeriesRaw"]` (`SpikeGadgets` interface, when `spikegadgets_file_path` is given) | Not yet exercised — no `.rec` files available (see open_questions.md Q5) |

**Not yet populated:** `LFP` (no LFP data in source files). `Raw` has a code path via the
`SpikeGadgets` interface but is untested end-to-end — no `.rec` files available yet (see
open_questions.md Q5).

---

## Metadata

Metadata is split into two layers (see `interfaces/README.md` for full details):

**Interface-level** (single source of truth — edit these for hardware/table descriptions):
- `metadata/_spike_sorting_mat_metadata.yaml` — `DataAcqDevice` (HH128), `Probe`, `Units` description
- `metadata/_processed_trials_metadata.yaml` — `processed_trials` column descriptions
- `metadata/_bcontrol_metadata.yaml` — Behavior table descriptions, Optogenetics hardware
- `metadata/_video_metadata.yaml` — `CameraDevice` specs

**Session-level** (experiment/dataset overrides — edit here for NWBFile/Subject fields):
- `arc_ecephys/metadata.yaml` — `NWBFile.experiment_description`, `Subject.species`/`strain`/`sex`

Per-subject fields (`date_of_birth`) are injected from `arc_behavior/rat_information.xlsx`.

---

## Placeholder Fields Requiring Lab Confirmation

The items below use prototype/placeholder values and **must be updated** before a
production DANDI upload or before the Spyglass database is used for analysis.
All changes are pure YAML edits — no Python code needs to be touched.

### Tetrode → brain region mapping

**File to edit:** `src/pagan_lab_to_nwb/metadata/_spike_sorting_mat_metadata.yaml`

Find the `tetrode_locations` key and replace `{}` with a dict mapping each tetrode
number (1–32) to its brain region string:

```yaml
tetrode_locations:
  1: "PFC"
  2: "PFC"
  9: "Striatum"
  10: "Striatum"
  # ... add all 32 tetrodes
```

Any tetrode number not listed falls back to `tetrode_location_default` (currently
`"unknown"`). If the mapping **differs between animals**, add an `Ecephys.tetrode_locations`
override to `src/pagan_lab_to_nwb/arc_ecephys/metadata.yaml` for the sessions that differ
— the per-session YAML is merged on top of the shared metadata at conversion time.

### Probe contact size (wire diameter)

**File to edit:** `src/pagan_lab_to_nwb/metadata/_spike_sorting_mat_metadata.yaml`

```yaml
Ecephys:
  Probe:
    contact_size: 0.0125   # ← replace with actual wire diameter in mm
```

> Must be a real number — `nan`/`None` breaks Spyglass `Probe.Electrode` re-insertion.
> See `open_questions.md` Q8 and `spyglass_notes.md` for details.

### Camera metadata

**File to edit:** `src/pagan_lab_to_nwb/metadata/_video_metadata.yaml`

```yaml
Video:
  CameraDevice:
    manufacturer: unknown      # ← replace with manufacturer name if available
    lens: unknown              # ← replace with lens model / focal length
    meters_per_pixel: 0.001   # ← replace with (cage_width_m / frame_width_px)
    camera_name: top_camera   # ← must match the name in the lab's Spyglass sgc.CameraDevice table
```

`meters_per_pixel` can be estimated from a known object in the frame: divide the
object's real width (metres) by its width in pixels.

`camera_name` must exactly match a row already registered in the Spyglass
`sgc.CameraDevice` table. If the table is empty, insert the desired name first, then
update this field to match.

### Hardware identifiers (lower priority)

**File to edit:** `src/pagan_lab_to_nwb/metadata/_spike_sorting_mat_metadata.yaml`

```yaml
Ecephys:
  DataAcqDevice:
    amplifier: "Horizontal Headstage 128-Channel Datalogger"   # ← confirm or correct
    adc_circuit: "Horizontal Headstage 128-Channel Datalogger" # ← confirm or correct
```

These are currently assumed from the HH128 product description (confirmed by lab that
amplifier and ADC are integrated in the same unit). Only update if the lab identifies
a more specific part name.

### Data availability

| Item | Status | Action |
|---|---|---|
| Raw SpikeGadgets `.rec` files | Not included (Princeton backup inaccessible) | `SpyglassSpikeGadgetsRecordingInterface` is implemented and wired into `ArcEcephysNWBConverter` — pass `spikegadgets_file_path` to `session_to_nwb()` once `.rec` access is restored. Untested end-to-end (no `.rec` files, no Spyglass `Raw`-table insertion run yet). |
| PSTH (`rrr4`) — keep or drop? | Currently stored in `processing["ecephys"]["rrr4_psth"]` | Confirm with lab whether it should be published. It significantly increases file size. |
| Video sync method | Uniform timestamps from nominal frame rate (~19.98 fps); optional constant offset via `video_time_offset` | Update `SpyglassVideoInterface` when a hardware sync signal is available. |

---

## Video Synchronization

### Current state

Video timestamps are derived from the nominal frame rate (~19.98 fps) reported by the
MP4 container, starting at t = 0. No hardware sync signal (TTL pulse, LED flash, etc.)
is available for the current dataset (confirmed by lab, 2026-04-21).

The pipeline exposes a `video_time_offset` parameter in `session_to_nwb()` that shifts
**all** video timestamps by a constant (seconds). This lets you align the video to the
behaviour session time-base without any code changes:

```python
nwb_path = session_to_nwb(
    behavior_file_path="data_@TaskSwitch6_Marino_P267_221211a.mat",
    nwb_folder_path="/path/to/output/",
    video_file_path="video_@TaskSwitch6_Marino_P267_221211a.mp4",
    video_time_offset=12.5,  # video started 12.5 s after session_start_time
)
```

After conversion the `ImageSeries` timestamps in the NWB file will be in seconds
relative to `nwbfile.session_start_time` (the BControl session clock).

### Estimating the offset from file metadata

When video and behaviour are recorded on the **same machine**, the video file's creation
timestamp can serve as a rough estimate of when recording started:

```python
import platform
from datetime import datetime, timezone
from pathlib import Path

video_path = Path("video_@TaskSwitch6_Marino_P267_221211a.mp4")
session_start = nwbfile.session_start_time  # timezone-aware datetime

# macOS exposes true file-creation time via st_birthtime
if platform.system() == "Darwin":
    video_start_ts = video_path.stat().st_birthtime
else:
    # Linux: no creation time; use mtime minus video duration as fallback
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    video_duration = n_frames / fps
    video_start_ts = video_path.stat().st_mtime - video_duration

video_start = datetime.fromtimestamp(video_start_ts, tz=timezone.utc)
time_offset = (video_start - session_start).total_seconds()
print(f"Estimated video_time_offset: {time_offset:.2f} s")
```

> **Caveats:**
> - File timestamps can be unreliable if data were copied between machines (mtime is
>   typically preserved on `cp -p` / rsync but creation time is reset on Linux).
> - The estimate assumes the machine clocks were synchronised at recording time.
> - A constant frame-rate assumption means drift accumulates; for long sessions (>30 min)
>   even small clock offsets become visible.

### Future: hardware sync

When a hardware sync method is available (TTL pulse on a dedicated channel, sync LED
visible in the video, etc.), replace the constant offset with per-frame timestamps
derived from the sync signal. The `SpyglassVideoInterface.add_to_nwbfile()` signature
already accepts `time_offset: float | None`, so the extension point is clear — replace
the scalar with an array of real timestamps computed from the sync signal.

The relevant file is:
`src/pagan_lab_to_nwb/interfaces/spyglass_video_interface.py` — lines 83–85.

---
