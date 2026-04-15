# pagan-lab-to-nwb
NWB conversion scripts for Pagan lab data to the
[Neurodata Without Borders](https://nwb-overview.readthedocs.io/) data format.


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
            │   ├── nwbconverter.py
            │   ├── metadata.yaml
            │   └── documentation/                  # See below
            └── interfaces/                         # Custom NeuroConv interfaces

## Tutorials

Interactive notebooks are in [`src/pagan_lab_to_nwb/tutorials/`](src/pagan_lab_to_nwb/tutorials/):

| Notebook | Contents |
|---|---|
| `arc_behavior_dandi_demo_notebook.ipynb` | DANDI streaming demo: reads NWB files directly from DANDI:001550 without downloading |
| `arc_behavior_optogenetics_notebook.ipynb` | Optogenetics deep-dive: visualises per-trial laser power, stimulation windows, and FOF site metadata |
| `protocol_comparison_notebook.ipynb` | Cross-protocol sanity check: loads one NWB file per protocol and compares table structure, heatmaps, and transition graphs |
| `arc_behavior_example_notebook.ipynb` | Single-session explorer: loads a TaskSwitch6 NWB file and walks through states, events, actions, trials, and stimulus data |

## Conversion documentation

Detailed notes for the `arc_behavior` conversion live in
[`src/pagan_lab_to_nwb/arc_behavior/documentation/`](src/pagan_lab_to_nwb/arc_behavior/documentation/README.md):

| File | Contents |
|---|---|
| `data_manifest.md` | Field-by-field map from BControl `.mat` to NWB |
| `conversion_issues.md` | Bugs, data quirks, and fixes encountered during conversion |
| `conversion_progress.md` | Per-protocol file counts and DANDI upload status |
| `nwbinspector_report.md` | NWBInspector results and explanation of each violation |
