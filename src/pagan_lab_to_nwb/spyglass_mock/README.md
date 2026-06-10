# Spyglass Database Setup — Pagan Lab

This guide walks you through setting up a local Spyglass (DataJoint/MySQL) database
and inserting Pagan Lab NWB files into it. It covers everything from installing
software prerequisites to running the example tutorial notebooks.

Both conversion pipelines in this repo (`arc_behavior` and `arc_ecephys`) write
NWB files that are Spyglass-compatible and share this same database setup.

Estimated time: ~30 minutes for a fresh setup.

---

## What You Need Before Starting

- **macOS or Linux** (Windows is not officially supported by Spyglass)
- **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** — the database runs inside a Docker container
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** — the package manager used by this project (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **Git** — for cloning the repository

---

## Step 1: Install the Python Environment

Clone the repository and install all dependencies with `uv`:

```bash
git clone https://github.com/catalystneuro/pagan-lab-to-nwb
cd pagan-lab-to-nwb
uv venv --python 3.12
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv pip install --editable .
```

This installs everything needed for both pipelines, including a CatalystNeuro fork
of Spyglass (`weiglszonja/spyglass`, branch `pagan-lab-to-nwb-fixes-v2`, pinned in
`pyproject.toml`). The fork adds support for `ndx-structured-behavior` (the BControl
task-recording extension) plus several Spyglass bugfixes — see
[`arc_ecephys/documentation/spyglass_notes.md`](../arc_ecephys/documentation/spyglass_notes.md)
for the full list.

> **Why not `pip install spyglass-neuro`?** The standard PyPI package does not
> support `ndx-structured-behavior`, and Spyglass requires DataJoint 0.14.x —
> DataJoint 2.x changes the schema API in a way that is incompatible. The fork
> pinned in `pyproject.toml` has the correct versions of both.

---

## Step 2: Set Up the Data Directories

Spyglass expects your data to live in a specific folder structure. Create these
folders somewhere with enough disk space (raw NWB files can be large):

```bash
mkdir -p /path/to/pagan_data/{raw,analysis,recording,spikesorting,waveforms,tmp,video,export}
```

> **Important:** NWB files you want to insert must be placed in the `raw/`
> subfolder before running an insertion script.

---

## Step 3: Start the Database with Docker

Spyglass stores all metadata in a MySQL database, run locally via Docker.

```bash
cd src/pagan_lab_to_nwb/spyglass_mock
docker compose up -d
docker ps
```

You should see a container named `spyglass_mock-db-1` with port `3306` listed.

> **Apple Silicon (M1/M2/M3) Macs:** `docker-compose.yml` already includes
> `platform: linux/amd64`, which runs the x86 MySQL image under Rosetta emulation
> (a native ARM build of the DataJoint MySQL image is not available). This is
> slower than native but works correctly.

> **Stopping the database:** `docker compose down` in this directory. Data is
> preserved in `spyglass_mock/data/` (gitignored) and available again on the next
> `docker compose up -d`.

---

## Step 4: Configure DataJoint

DataJoint reads its settings from a JSON file. Copy the template and edit the
paths to match the directories you created in Step 2:

```bash
cd ../arc_ecephys
cp dj_local_conf.example.json dj_local_conf.json
```

Open `dj_local_conf.json` and replace every `/path/to/pagan_data` with the actual
path you chose in Step 2.

> **Do not use `~` (home directory shorthand).** Spyglass does not expand `~` —
> use the full absolute path (e.g. `/Users/yourname/pagan_data`).

`dj_local_conf.json` is gitignored (it contains machine-specific paths) and is
shared by both pipelines' `insert_session.py` scripts.

---

## Step 5: Verify the Connection

```bash
python -c "
import datajoint as dj
dj.config.load('src/pagan_lab_to_nwb/arc_ecephys/dj_local_conf.json')
dj.conn(use_tls=False)
print('Connected successfully')
"
```

You should see `Connected successfully` with no errors.

> **"Connection refused" error?** The Docker container is not running — go back
> to Step 3 and start it with `docker compose up -d`.

---

## Step 6: Convert a Session to NWB

Each pipeline has its own `convert_session.py`. Edit the paths in the script's
`__main__` block to point to your data files, then run:

```bash
# arc_behavior (BControl-only sessions)
python src/pagan_lab_to_nwb/arc_behavior/convert_session.py

# arc_ecephys (ephys + video + behavior sessions)
python src/pagan_lab_to_nwb/arc_ecephys/convert_session.py
```

Copy the resulting `.nwb` file(s) into your `raw/` directory (Step 2) before
proceeding to insertion.

---

## Step 7: Insert the Session into Spyglass

Each pipeline has its own `insert_session.py`. Both load
`arc_ecephys/dj_local_conf.json`, so run them from their own directory:

```bash
# arc_behavior
cd src/pagan_lab_to_nwb/arc_behavior
python insert_session.py

# arc_ecephys
cd src/pagan_lab_to_nwb/arc_ecephys
python insert_session.py
```

Edit the `__main__` block at the bottom of each script to list the NWB file(s)
you want to insert. A successful run prints lines like:

```
Inserting: sub-P100_ses-TaskSwitch4-181010a.nwb
Core insertion complete.
BControl tables (TaskRecordingTypes, TaskRecording) populated.
  SortedSpikesGroup 'all_units' created with 641 units annotated.
test_task_recording_types passed for sub-P100_ses-TaskSwitch4-181010a.nwb
test_sorted_spikes passed for sub-P100_ses-TaskSwitch4-181010a_.nwb (641 units)
```

A summary of the populated tables is written to `tables.txt` in the same directory.

> **Re-inserting after a failed run:** `clean_existing=True` (the default in
> `__main__`) deletes any partial entries before re-inserting, preventing
> "duplicate key" errors.

---

## Step 8: Explore the Data with the Tutorial Notebooks

| Notebook | Covers |
|---|---|
| [`tutorials/spyglass_tutorial.ipynb`](../tutorials/spyglass_tutorial.ipynb) | `arc_behavior` session — `TaskRecordingTypes`, `TaskRecording`, optogenetics |
| [`tutorials/arc_ecephys_spyglass_tutorial.ipynb`](../tutorials/arc_ecephys_spyglass_tutorial.ipynb) | `arc_ecephys` sessions P100 (ephys + spikes + PSTH + processed trials) and P267 (video) |

```bash
jupyter notebook src/pagan_lab_to_nwb/tutorials/spyglass_tutorial.ipynb
```

---

## Expected Warnings (These Are Normal)

The following warnings appear during insertion and can be safely ignored:

| Warning | Reason |
|---------|--------|
| `No conforming camera device metadata found` | Session has no video (e.g. an ephys-only session) |
| `No conforming probe metadata found` | Session has no ephys (e.g. a video-only session) |
| `Unable to import SampleCount` | SpikeGadgets sample count not present in these files |
| `No conforming behavioral events data interface found` | BControl does not write DIOEvents |
| `Found overlap(s)` | Epoch start/stop timestamps are approximate — non-blocking |

See [`arc_ecephys/documentation/spyglass_notes.md`](../arc_ecephys/documentation/spyglass_notes.md)
for the full list of issues encountered (and fixed) during development.

---

## Troubleshooting

### "Can't connect to MySQL server on 'localhost'"
The Docker container is not running:
```bash
cd src/pagan_lab_to_nwb/spyglass_mock
docker compose up -d
```

### "DuplicateError: Duplicate entry ... for key PRIMARY"
A previous insertion left partial entries in the database. Re-run the insertion
script — it uses `clean_existing=True` by default, which removes stale entries
first. If the error persists, delete the entry manually:
```python
from spyglass.common import Nwbfile
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
copy_name = get_nwb_copy_filename("your_file.nwb")
(Nwbfile & {"nwb_file_name": copy_name}).delete(safemode=False)
```

### Import errors when running Python scripts
Always load the DataJoint config **before** importing any Spyglass module:
```python
import datajoint as dj
dj.config.load("/path/to/dj_local_conf.json")
dj.conn(use_tls=False)

# Only import spyglass AFTER the config is loaded
import spyglass.common as sgc
```
This is already done correctly in both `convert_session.py` and `insert_session.py`
for both pipelines.
