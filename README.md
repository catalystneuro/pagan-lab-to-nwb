# pagan-lab-to-nwb
NWB conversion scripts for Pagan lab data to the
[Neurodata Without Borders](https://nwb-overview.readthedocs.io/) data format.

This repository hosts two independent conversion pipelines:

- **[ARC Behavior](#arc-behavior)** — BControl task-switching behavioral data underlying
  [Mah et al. 2024, *Nature*](https://www.nature.com/articles/s41586-024-08433-6),
  published as [DANDI:001550](https://dandiarchive.org/dandiset/001550).
- **[ARC Ecephys](#arc-ecephys)** — ongoing data collection combining tetrode
  electrophysiology, video, and BControl behavior, converted for ingestion into a
  Spyglass (DataJoint) database.

## Installation

### From GitHub (recommended for development)

Installing from source lets you modify the conversion code directly.
We use [uv](https://docs.astral.sh/uv/) for environment and dependency management
([installation instructions](https://docs.astral.sh/uv/getting-started/installation/)).

```bash
git clone https://github.com/catalystneuro/pagan-lab-to-nwb
cd pagan-lab-to-nwb
uv venv --python 3.12
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv pip install --editable .
```

This installs the package in [editable mode](https://pip.pypa.io/en/stable/cli/pip_install/#editable-installs)
so any changes you make to the source are immediately reflected without reinstalling.

### Spyglass database setup (optional)

Both pipelines write NWB files that are compatible with [Spyglass](https://github.com/LorenFrankLab/spyglass)
(DataJoint). To insert converted sessions into a local Spyglass/MySQL database —
e.g. to run `insert_session.py` or the `spyglass_tutorial.ipynb` notebooks — follow
the consolidated setup guide at
[`src/pagan_lab_to_nwb/spyglass_mock/README.md`](src/pagan_lab_to_nwb/spyglass_mock/README.md)
(Docker MySQL container, DataJoint config, conversion + insertion steps for both
pipelines).

## ARC Behavior

BControl behavioral data from the task-switching auditory decision-making paradigm
described in [Mah et al. 2024, *Nature*](https://www.nature.com/articles/s41586-024-08433-6).
Each session is converted to NWB using
[`ndx-structured-behavior`](https://github.com/rly/ndx-structured-behavior) to represent
the finite-state-machine states, events, actions, and per-trial stimulus data for all
11 BControl protocols (`PBups`, `ProAnti3`, `ProAnti3Marino`, `TaskSwitch`,
`TaskSwitch2`–`TaskSwitch6` and variants).

The full dataset (16,113 NWB files) is published as
[DANDI:001550](https://dandiarchive.org/dandiset/001550), and the files are also
Spyglass-compatible (see [Spyglass insertion](#spyglass-insertion) below).

### Running a conversion

Convert a single session:
```bash
python src/pagan_lab_to_nwb/arc_behavior/convert_session.py
```

Convert, validate, and upload an entire protocol to DANDI in batches:
```bash
.venv/bin/python src/pagan_lab_to_nwb/arc_behavior/convert_and_upload_batched.py --protocol TaskSwitch6
```

Use `--dry-run` to preview the batch plan before committing to a full run. See the
script docstring for all options (`--start-batch`, `--batch-size`, `--data-dir`, `--output-dir`).

### Tutorials

| Notebook | Contents |
|---|---|
| `arc_behavior_dandi_demo_notebook.ipynb` | DANDI streaming demo: reads NWB files directly from DANDI:001550 without downloading |
| `arc_behavior_optogenetics_notebook.ipynb` | Optogenetics deep-dive: visualises per-trial laser power, stimulation windows, and FOF site metadata |
| `protocol_comparison_notebook.ipynb` | Cross-protocol sanity check: loads one NWB file per protocol and compares table structure, heatmaps, and transition graphs |
| `arc_behavior_example_notebook.ipynb` | Single-session explorer: loads a TaskSwitch6 NWB file and walks through states, events, actions, trials, and stimulus data |
| `spyglass_tutorial.ipynb` | Spyglass access tutorial: queries `TaskRecordingTypes` and `TaskRecording` for an inserted arc_behavior session |

### Conversion documentation

Detailed notes for the `arc_behavior` conversion live in
[`src/pagan_lab_to_nwb/arc_behavior/documentation/`](src/pagan_lab_to_nwb/arc_behavior/documentation/README.md):

| File | Contents |
|---|---|
| `data_manifest.md` | Field-by-field map from BControl `.mat` to NWB |
| `conversion_issues.md` | Bugs, data quirks, and fixes encountered during conversion |
| `conversion_progress.md` | Per-protocol file counts and DANDI upload status |
| `nwbinspector_report.md` | NWBInspector results and explanation of each violation |

### Spyglass insertion

```bash
cd src/pagan_lab_to_nwb/arc_behavior
python insert_session.py
```

See [`spyglass_mock/README.md`](src/pagan_lab_to_nwb/spyglass_mock/README.md) for
database setup, and `spyglass_tutorial.ipynb` for an example of querying an
inserted session.

## ARC Ecephys

Conversion code for the lab's ongoing data collection, combining tetrode
electrophysiology (SpikeGadgets), spike-sorted units, an overhead video recording, and
BControl behavior for the same task-switching paradigm as ARC Behavior. The converter
(`ArcEcephysNWBConverter`) is built to be Spyglass-compatible from the start:

- `DataAcqDevice` + `Probe` + `NwbElectrodeGroup` hierarchy (named `nTrode{N}`) with
  the Spyglass-required electrode columns (`probe_shank`, `probe_electrode`,
  `bad_channel`, `ref_elect_id`, `group_name`, `brain_area`)
- Video stored as `ImageSeries(external_file=[...])`
- Processed trial data (e.g. PSTHs) added via `ProcessedTrialsInterface`

This dataset has not yet been published to DANDI; it is converted and inserted into a
local Spyglass database (`arc_ecephys/insert_session.py`) as new sessions are recorded.

### Running a conversion

Convert a single session (behavior + spikes + processed trials + video, each optional):
```bash
python src/pagan_lab_to_nwb/arc_ecephys/convert_session.py
```

### Tutorials

| Notebook | Contents |
|---|---|
| `arc_ecephys_spyglass_tutorial.ipynb` | Spyglass access tutorial covering electrode/probe geometry, video, spike-sorted units, PSTHs, and processed trials for example sessions P100 and P267 |

### Conversion documentation

Detailed notes for the `arc_ecephys` conversion live in
[`src/pagan_lab_to_nwb/arc_ecephys/documentation/`](src/pagan_lab_to_nwb/arc_ecephys/documentation/README.md):

| File | Contents |
|---|---|
| `data_manifest.md` | Field-by-field map from source files to NWB destinations |
| `open_questions.md` | Open questions sent to the lab, with resolution status |
| `spyglass_notes.md` | Errors encountered during Spyglass insertion, fixes applied, and expected warnings |

### Spyglass insertion

```bash
cd src/pagan_lab_to_nwb/arc_ecephys
python insert_session.py
```

See [`spyglass_mock/README.md`](src/pagan_lab_to_nwb/spyglass_mock/README.md) for
database setup, and `arc_ecephys_spyglass_tutorial.ipynb` for example queries
(electrode/probe geometry, spike-sorted units, PSTHs, processed trials, video).

## Repository structure

    pagan-lab-to-nwb/
    ├── LICENSE
    ├── pyproject.toml
    ├── README.md
    └── src
        └── pagan_lab_to_nwb
            ├── arc_behavior/                      # BControl → NWB conversion (all protocols)
            │   ├── convert_session.py              # Convert a single session
            │   ├── convert_all_sessions.py         # Convert all sessions (no upload)
            │   ├── convert_and_upload_batched.py   # Convert + validate + upload any protocol
            │   ├── insert_session.py               # Insert into Spyglass database
            │   ├── nwbconverter.py
            │   ├── metadata.yaml                   # ← Edit here: NWBFile, Subject, Session
            │   └── documentation/                  # See below
            ├── arc_ecephys/                        # Ephys + behavior + video → NWB (Spyglass)
            │   ├── convert_session.py
            │   ├── insert_session.py               # Insert into Spyglass database
            │   ├── nwbconverter.py
            │   ├── metadata.yaml                   # ← Edit here: NWBFile, Subject
            │   ├── dj_local_conf.example.json      # Template DataJoint config (copy → dj_local_conf.json)
            │   └── documentation/                  # See below
            ├── spyglass_mock/                       # Spyglass/DataJoint database setup (Docker MySQL)
            │   ├── README.md                        # Setup guide for both pipelines
            │   └── docker-compose.yml
            ├── metadata/                           # Interface-level metadata (single source of truth)
            │   ├── _bcontrol_metadata.yaml          #   Behavior tables + Optogenetics hardware
            │   ├── _spike_sorting_mat_metadata.yaml #   Ecephys hardware (DataAcqDevice, Probe, Units)
            │   ├── _processed_trials_metadata.yaml  #   processed_trials column descriptions
            │   └── _video_metadata.yaml             #   CameraDevice specs
            └── interfaces/                         # Custom NeuroConv interfaces (load metadata/)

## Metadata

Metadata is split into two layers:

- **`metadata/`** — hardware specs and NWB table/column descriptions, one YAML per interface.
  These are the single source of truth and are loaded automatically by each interface's
  `get_metadata()`. Edit these when device specifications or field descriptions change.
- **`arc_behavior/metadata.yaml`** / **`arc_ecephys/metadata.yaml`** — session- and
  dataset-level fields only (`NWBFile`, `Subject`, `Session`). Edit these when the
  experiment description, publication DOI, institution, or subject strain changes.

See [`src/pagan_lab_to_nwb/interfaces/README.md`](src/pagan_lab_to_nwb/interfaces/README.md)
for the full mapping of which YAML file each interface reads.
