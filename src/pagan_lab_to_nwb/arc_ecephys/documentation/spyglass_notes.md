# Spyglass Conversion Notes — arc_ecephys

**Repo:** pagan-lab-to-nwb
**Data:** `/Volumes/T9/data/Ephys_data_examples`, `/Volumes/T9/data/Video_data_examples`
**Updated:** 2026-06-10 (handover-readiness re-verification; original Phase 8: 2026-04-09)

---

## Experiment Overview

Auditory decision-making task-switching paradigm (BControl) in Long-Evans rats.
Recordings from 32-tetrode SpikeGadgets arrays. Sessions include spike-sorted
tetrode data (`.mat`), processed neural responses (`dati_*.mat`), BControl
behavior (`data_@*.mat`), and behavioral video (`.mp4`).

---

## Data Streams

| Stream | Format | File Pattern | Spyglass Pipeline |
|--------|--------|--------------|-------------------|
| BControl behavior | MATLAB v5 `.mat` | `data_@{...}.mat` | Session / trials |
| Spike-sorted units | MATLAB v7.3 (HDF5) | `spikes_@{...}.mat` | Electrode / Units |
| Processed trial+neural data | MATLAB v5 `.mat` | `dati_{...}.mat` | Trials (extra cols) |
| Behavioral video | MP4 | `video_@{...}.mp4` | VideoFile |
| Raw ephys recording | SpikeGadgets `.rec` | `*.rec` | Raw (`ElectricalSeriesRaw`) — `SpyglassSpikeGadgetsRecordingInterface` added 2026-05-19, untested (no `.rec` files yet) |

---

## Source Data Files

### Ephys_data_examples (P100, TaskSwitch4)

| File | Notes |
|------|-------|
| `data_@TaskSwitch4_Marino_P100_181010a.mat` | Behavior — confirmed correct by lab (2026-04-22) |
| `spikes_@TaskSwitch4_Marino_P100_181010a.mat` | Spike-sorted |
| `dati_TaskSwitch4_Marino_P100_181010a.mat` | Processed data |

> An earlier file `180110a` was included by mistake and should be ignored.

### Video_data_examples (P267, TaskSwitch6)

| File | Notes |
|------|-------|
| `data_@TaskSwitch6_Marino_P267_211103a.mat` | Behavior (no matching video) |
| `data_@TaskSwitch6_Marino_P267_221211a.mat` | Behavior (paired with video below) |
| `video_@TaskSwitch6_Marino_P267_221103a.mp4` | Video (no matching behavior — possible typo) |
| `video_@TaskSwitch6_Marino_P267_221211a.mp4` | Video (paired with 221211 behavior) |

---

## Spyglass Ingestion Targets

- [x] `DataAcquisitionDevice` ← `DataAcqDevice` (ndx-franklab-novela) in `nwbfile.devices`
- [x] `Probe` ← `Probe` + `Shank` + `ShanksElectrode` hierarchy in `nwbfile.devices`
- [x] `ElectrodeGroup` ← `NwbElectrodeGroup` with targeted_location/xyz/units
- [x] `Electrode` ← electrodes table with probe_shank, probe_electrode, bad_channel, ref_elect_id
- [ ] `Raw` — `SpyglassSpikeGadgetsRecordingInterface` (added 2026-05-19, `add_spikegadgets_code`
      branch) writes `ElectricalSeriesRaw` to acquisition when `spikegadgets_file_path` is
      given; not yet exercised end-to-end (no `.rec` files available — see open_questions.md Q5)
- [ ] `LFP` — not populated (LFP data not in source files)
- [x] `CameraDevice` ← `CameraDevice` (ndx-franklab-novela) in `nwbfile.devices`
- [x] `TaskEpoch` ← `nwbfile.epochs` + `processing["tasks"]` DynamicTable
- [x] `VideoFile` ← `ImageSeries` in `processing["behavior"]["video"]` with CameraDevice

---

## NWB File Structure

```
nwbfile
├── acquisition
│   └── ElectricalSeriesRaw  Raw broadband signal, all tetrode channels   [ephys, raw .rec only]
├── devices
│   ├── SpikeGadgets    DataAcqDevice (system, amplifier, adc_circuit)   [ephys]
│   ├── tetrode_array   Probe → Shank → ShanksElectrode                  [ephys]
│   └── camera_device 1 CameraDevice (camera_name, meters_per_pixel)     [video]
├── electrode_groups    NwbElectrodeGroup per tetrode (tetrode{N})        [ephys]
├── electrodes          4 ch × n_tetrodes; probe_shank/electrode/bad/ref  [ephys]
├── units               spike_times, waveform_mean/sd, trode_id           [ephys]
├── trials              BControl + dati columns (choice, hits, tim_0..6)
├── epochs              TimeIntervals (tags=["01"])                        [all]
├── processing
│   ├── behavior
│   │   └── video → BehavioralEvents → ImageSeries (external MP4)        [video]
│   ├── tasks           DynamicTable with task_name, camera_id, epochs    [all]
│   └── neural_responses → rrr4_psth TimeSeries                          [dati]
└── lab_meta_data["task"]  ndx-structured-behavior Task                   [all]
```

---

## Placeholder Values Requiring Lab Confirmation

| Field | Value | Status |
|-------|-------|--------|
| `NwbElectrodeGroup.location` | `"unknown"` | **Open** — tetrode→brain region map (Q2) |
| `DataAcqDevice` name/system | `"HH128"` / `"SpikeGadgets"` | ✓ Confirmed (HH128, 2026-04-21) |
| `DataAcqDevice` amplifier/adc_circuit | `"Intan"` / `"Intan"` | **Open** — assumed Intan, not confirmed by lab (Q7) |
| `Probe.contact_size` | `0.0125` mm (12.5 µm) | **Open** — placeholder value, wire diameter unconfirmed by lab (Q8). Must remain a real float, not `nan`/`None` — see error #11 |
| `CameraDevice.model` | `"HDE-Security-Camera-Vision-Pinhole"` | ✓ Confirmed (2026-04-21); now wired up via `DeviceModel` (see error #10) |
| `CameraDevice.lens` | `"unknown"` | **Open** — lab could not find spec (Q9) |
| `CameraDevice.meters_per_pixel` | `0.001` | **Open** — depends on cage geometry (Q9) |
| `CameraDevice.camera_name` | `"top_camera"` | **Open** — match sgc.CameraDevice table (Q11) |
| `ImageSeries.timestamps` | Uniform 19.98 fps | ✓ Final — no sync signal (Q3, 2026-04-21) |
| `nwbfile.epochs.stop_time` | `0.0` | **Open** — real session end time |
| `tim` column names | `previous_trial_end` … `trial_end` | ✓ Confirmed (2026-04-21) |

---

## Phase Progress

- [x] Phase 1: Spyglass environment setup
- [x] Phase 2: Experiment discovery / data inspection
- [x] Phase 3: Data inspection (mat file structure)
- [x] Phase 4: Metadata collection (metadata.yaml)
- [ ] Phase 5: Synchronization analysis (video sync — pending Q3)
- [x] Phase 6: Code generation (interfaces + converter done)
- [x] Phase 7: Testing & validation (stub conversion + nwbinspector)
- [x] Phase 8: Session insertion (both sessions in DB — see Insertion Log below)
- [x] Phase 9: Table verification (`tables.txt`, `test_task_recording_types` pass)
- [x] Phase 10: Tutorial notebook (`tutorials/spyglass_tutorial.ipynb` — runs end-to-end)

---

## Insertion Log (Phase 8)

### Sessions inserted

| Session | NWB file (raw dir) | Copy (Spyglass) | Tables |
|---------|-------------------|-----------------|--------|
| P100 (TaskSwitch4, ephys+behavior) | `sub-P100_ses-TaskSwitch4-181010a.nwb` | `sub-P100_ses-TaskSwitch4-181010a_.nwb` | Session, IntervalList, ElectrodeGroup(1), Electrode(4), TaskEpoch, TaskRecordingTypes, TaskRecording |
| P267 (TaskSwitch6, video+behavior) | `sub-P267_ses-TaskSwitch6-221211a.nwb` | `sub-P267_ses-TaskSwitch6-221211a_.nwb` | Session, IntervalList, TaskEpoch, VideoFile, TaskRecordingTypes, TaskRecording |

> Note: the session date for P100 is `181010a` (10-Oct-2018) — an earlier version
> of this table had a transposed-digit typo (`180110a`).

**Raw dir:** `/Volumes/T9/data/Pagan/raw/`
**Output verified:** `test_task_recording_types` passes for both sessions.

### Errors encountered and fixes

#### 1. `expression VARCHAR(127)` too short for BControl argument history arrays
- **Symptom:** `pymysql.err.DataError: Data too long for column 'expression' at row 159`
- **Root cause:** BControl `HistorySection_*` arguments store trial-history arrays as
  string repr (up to ~503 chars). Spyglass fork `common_task_rec.py` had `varchar(127)`.
- **Fix (Spyglass fork):** Changed `expression=NULL: varchar(127)` → `expression=NULL: text`
  in `TaskRecordingTypes.Arguments.definition`. Also applied via
  `ALTER TABLE cbroz_common_task_rec.task_recording_types__arguments MODIFY expression TEXT DEFAULT NULL;`

#### 2. `contact_size=nan` breaks DataJoint WHERE clauses — SUPERSEDED, see error #11
- **Symptom:** `UnknownAttributeError: Unknown column 'nan' in 'where clause'`
- **Root cause:** `float("nan")` stored as `contact_size` in ndx-franklab-novela Probe
  becomes literal SQL identifier `nan` in DataJoint queries.
- **Fix (interface), now superseded:** Changed `_spikes_mat.py` to use
  `contact_size=None` instead of `float("nan")` in both `get_metadata()` and
  `add_to_nwbfile()`.
- **Fix (Spyglass fork `common_device.py`):** Added `nan` → `None` coercion when building
  the elect_dict from `nwb_probe_obj.contact_size`.
- **⚠️ This `None`/`nan` fix turned out to be only a partial fix** — it avoided the
  immediate `UnknownAttributeError` on a *first-ever* insert, but introduced a
  latent bug that broke every *re-insertion*. See error #11 for the full story and
  the final fix (a real numeric `contact_size`).

#### 3. `Task._table_to_dict()` fails on missing `task_type`/`task_subtype`
- **Symptom:** `AttributeError: 'Series' object has no attribute 'task_type'`
- **Root cause:** Our tasks DynamicTable lacked `task_type`/`task_subtype` columns.
  Spyglass `Task._table_to_dict()` accessed `row.task_type` unconditionally.
- **Fix (NWB conversion):** Added `task_type` and `task_subtype` columns to the tasks
  DynamicTable in `_add_spyglass_epoch_and_tasks()` (`convert_session.py`).
- **Fix (Spyglass fork `common_task.py`):** Changed to `row.get("task_type")` /
  `row.get("task_subtype")` to tolerate missing columns.

#### 4. `camera_names` required field missing for sessions without video
- **Symptom:** `MissingAttributeError: Field 'camera_names' doesn't have a default value`
- **Root cause:** `TaskEpoch.make()` didn't set `camera_names` when no matching camera
  device was found. MySQL requires the field.
- **Fix (Spyglass fork `common_task.py`):** Added `key["camera_names"] = []` as default
  when no valid camera IDs found.

#### 5. Empty numpy array truthiness error
- **Symptom:** `ValueError: The truth value of an empty array is ambiguous`
- **Root cause:** `if camera_ids:` on a numpy empty array raises ValueError.
  Our `camera_id=np.array([], dtype=np.int32)` for no-video sessions triggered this.
- **Fix (Spyglass fork `common_task.py`):** Changed `if camera_ids:` → `if len(camera_ids) > 0:`.
- **Fix (NWB conversion):** Changed `camera_id=[1]` → `camera_id=[1] if has_video else np.array([], dtype=np.int32)`.

#### 6. FK constraint failure: `common_ephys._electrode` → `probe__electrode`
- **Symptom:** FK constraint violation during Electrode population.
- **Root cause:** `_spikes_mat.py` built Probe hierarchy with one shank per tetrode
  (probe_shank=trode_id), but the Electrode table used probe_shank=0, probe_electrode=0..3.
  Spyglass FK constraint requires exact match.
- **Fix (`_spikes_mat.py`):** Restructured Probe to one shank (shank="0") with 4
  ShanksElectrodes (name="0".."3"), matching the Electrode table values.

#### 7. `DuplicateError` on TaskEpoch during re-insertion
- **Symptom:** `DuplicateError: Duplicate entry 'sub-P100..._-1' for key 'PRIMARY'`
- **Root cause:** Orphaned TaskEpoch row from a previous partial run (its parent Session
  had been deleted via raw SQL, bypassing DataJoint cascade). DataJoint cascade delete
  of Nwbfile only cascades through rows that still have parent entries — orphaned
  TaskEpoch rows (no parent Session) are not found and not deleted.
- **Fix:** Manually deleted orphaned TaskEpoch rows via DataJoint before re-insertion.
- **Prevention:** Always use `clean_existing=True` when re-inserting; use DataJoint
  `.delete()` (not raw SQL) to ensure cascade works correctly.

#### 8. `FileNotFoundError: NWB file not found in kachery or Dandi`
- **Symptom:** After `sgi.insert_sessions()` returned, `get_nwb_file()` failed with
  kachery/Dandi error for the `_.nwb` copy file.
- **Root cause:** The `_.nwb` copy was deleted during a previous raw-SQL cleanup while
  the Nwbfile DB entry remained. On re-run, `insert_sessions()` saw the existing entry
  and silently skipped (no exception, just a `warnings.warn()`). The copy was never
  re-created.
- **Fix:** Deleted the stale Nwbfile entry via DataJoint, then re-ran insertion.
- **Prevention:** Use `clean_existing=True` before each insert attempt.

#### 9. Spyglass soft rollback leaves TaskRecordingTypes populated without Session
- **Symptom:** After `sgi.insert_sessions()` returned normally, Session=0 but
  TaskRecordingTypes=1 and TaskRecording=1.
- **Root cause:** `populate_all_common(rollback_on_fail=True)` does a "soft rollback"
  by deleting Nwbfile (and cascading to Session) if InsertError entries exist.
  Our code calls `TaskRecordingTypes.insert_from_nwbfile()` AFTER `insert_sessions()`
  returns — but if a soft rollback happened, Session no longer exists.
- **Mitigation:** Add a post-insertion check: verify Session exists before calling
  BControl insertion. `insert_session.py` now uses `clean_existing=True` in `__main__`
  to ensure idempotent re-runs.

---

### Errors found during the 2026-06-10 handover-readiness re-verification

The Phase 8 insertion above (2026-04-09) was the only insertion ever performed
against this database. The handover-readiness check re-converted both sessions
from scratch with a fresh `uv` venv and re-inserted them with
`clean_existing=True` — the first time these sessions were ever *re*-inserted.
This surfaced several latent bugs that a first-time insertion can't trigger.

#### 10. `CameraDevice` `model` field requires a `DeviceModel` reference (P267)
- **Symptom:**
  ```
  ValueError: ... CameraDevice.model: incorrect type ... expected DeviceModel
  ```
  during `sgi.insert_sessions()` for the P267 (video) session.
- **Root cause:** `ndx-franklab-novela >= 0.2.4` changed `CameraDevice.model` from a
  plain string to a `DeviceModel` reference (with `.name` / `.manufacturer`).
  `interfaces/spyglass_video_interface.py` was still passing a bare string.
- **Fix:** `spyglass_video_interface.py` now calls
  `nwbfile.create_device_model(name=model_name, manufacturer=manufacturer,
  description=...)` and passes the resulting `DeviceModel` object as
  `CameraDevice(model=device_model, ...)`. Added `manufacturer` to
  `metadata/_video_metadata.yaml` (`Video.CameraDevice.manufacturer`, currently
  `"unknown"` — see open_questions.md Q9).
- **Status:** Fixed and verified end-to-end — `sgi.insert_sessions()` now logs
  `inserts 1 into CameraDevice` for P267 with no error.

#### 11. `Probe.Electrode.contact_size` `'None'`/`'nan'` divergence on re-insertion (P100)
- **Symptom:** On the *second* insertion against `probe_id='tetrode'`
  (`clean_existing=True` re-run), `sgi.insert_sessions()` logs
  `inserts 0 into Probe` / `inserts 0 into ProbeShank`, then fails with:
  ```
  Existing entry differs in 'contact_size' column of 'ProbeElectrode'.
  Accept the existing value of:
  'None'
  in place of the new value:
  'nan' ?
   [yes, no]: EOFError: EOF when reading a line
  ```
  (the `accept_divergence()` interactive prompt has no TTY in a script, so it
  raises `EOFError` and the whole `insert_sessions()` call rolls back.)
- **Root cause — why this is a *second-insert-only* bug:**
  1. The error #2 fix changed `contact_size` from `float("nan")` to `None`.
     `ndx_franklab_novela.Probe(contact_size=None, ...)` is accepted by docval
     (no type check on `None`), but after an HDF5 write/read round-trip,
     `nwbfile.devices["tetrode_array"].contact_size` comes back as
     `np.float64(nan)`, **not** `None`.
  2. On the **first-ever** insert (Phase 8, 2026-04-09), `Probe.Electrode` was
     empty, so `validate1_duplicate()` found no existing row and inserted
     directly — `contact_size=nan` silently became SQL `NULL` with no error,
     no comparison, and no warning.
  3. On a **second** insert against the same `probe_id='tetrode'` (this
     re-verification's `clean_existing=True` run), `validate1_duplicate()`
     finds the existing `Probe.Electrode` row (`contact_size=NULL`) and calls
     `_unequal_vals('contact_size', {'contact_size': nan}, {'contact_size':
     None})`. That helper does `a, b = a.get(key) or "", b.get(key, "") or ""`
     — `None or ""` → `""`, but `nan or ""` → `nan` (`nan` is truthy as a
     float). So it compares `nan != ""`, which is **always `True`** — the
     fields can never be judged equal, and `accept_divergence()` is invoked
     unconditionally.
  4. `clean_db_entry()`'s `(sgc.Probe & probe_ids).delete(safemode=False)` was
     *intended* to remove the stale `Probe`/`Probe.Electrode` rows before
     re-insertion, which would have avoided this — but it silently failed to
     persist (see error #12), so the `contact_size=NULL` row survived from
     Phase 8 and caused the divergence on this re-run.
  5. **This was a latent, ticking-time-bomb bug present since Phase 8** — it
     could not manifest until *something* triggered a second insert against
     `probe_id='tetrode'` (a `clean_existing=True` re-run, or a new session
     sharing the same probe), which had never happened before this
     handover-readiness check.
- **Fix:**
  - Changed `Ecephys.Probe.contact_size` in
    `metadata/_spike_sorting_mat_metadata.yaml` from `.nan` to `0.0125` (12.5 µm,
    a standard nichrome tetrode wire diameter — see open_questions.md Q8). A real
    float round-trips through HDF5 and MySQL `FLOAT` unchanged (`0.0125 ==
    0.0125` verified empirically), so `_unequal_vals()` now correctly judges the
    field equal on every re-insert.
  - One-time DB cleanup: removed the orphaned `contact_size=NULL`
    `Probe`/`Probe.Shank`/`Probe.Electrode` rows for `probe_id='tetrode'` (4 rows,
    confirmed to have zero `ElectrodeGroup`/`Electrode` references across all
    subjects) via raw SQL (see error #12 for why DataJoint's `.delete()` couldn't
    be used). `ProbeType('tetrode')` was left in place (separate, still-seeded
    table).
- **Status:** Fixed and verified end-to-end — re-run of `insert_session.py` with
  `clean_existing=True` now logs `inserts 1 into Probe`, `inserts 1 into
  ProbeShank`, `inserts 4 into ProbeElectrode` with no divergence prompt.
  `test_sorted_spikes` passed (641 units).

#### 12. DataJoint 0.14.9 `Table.delete(safemode=False)` does not persist for `common_device.probe*`
- **Symptom:** `(sgc.Probe & "probe_id='tetrode'").delete(safemode=False)` (and
  the equivalent calls for `Probe.Shank`, `Probe.Electrode`, `CameraDevice` made
  by `clean_db_entry()`) logs `[INFO] Deleting N rows from
  common_device.probe...` with `N` matching the actual row count — but the rows
  are still present immediately afterwards, in the *same* connection, after an
  explicit `conn._conn.commit()`, and even from a brand-new Python process.
- **Investigated:** `dj.config['database.prefix']` is unset; only one
  `common_device` schema exists; `conn.in_transaction` is `False` and
  `autocommit` is `True` both before and after `.delete()`. Read DataJoint
  0.14.9's `table.py` `delete()` (lines ~485-690): with `safemode=False` and
  `delete_count > 0`, the code path that should run
  `self.connection.commit_transaction()` appears to be reached — but the delete
  still doesn't persist. Exact mechanism not determined.
- **Workaround:** Raw SQL via
  `conn.query("DELETE FROM \`common_device\`.\`probe__electrode\` WHERE
  probe_id='tetrode'")` (and similarly for `probe__shank`, `probe`), followed by
  `conn._conn.commit()`. Confirmed to persist (verified empty from a fresh
  process).
- **Implication:** `clean_db_entry()`'s `(sgc.Probe & probe_ids).delete(...)` and
  `(sgc.CameraDevice & camera_names).delete(...)` calls (in `insert_session.py`)
  are likely **silently no-ops** on this DataJoint version. With the error #11
  fix (`contact_size=0.0125`, consistent across all future inserts of
  `probe_id='tetrode'`), this no longer causes a *visible* problem for `Probe` —
  but it means stale `CameraDevice`/`Probe` rows can accumulate invisibly across
  `clean_existing=True` re-runs. **Not yet root-caused; flagged for follow-up.**
  If a future session needs a genuinely different `Probe`/`CameraDevice` row
  re-created from scratch, use the raw-SQL workaround above, not
  `.delete(safemode=False)`.

#### 13. Duplicate `tasks` processing-module table from `SpikeSortingMatInterface` — ⚠️ INCORRECT DIAGNOSIS, see error #15
- **Original (incorrect) symptom/diagnosis:** This entry originally claimed two
  independent code paths each created a `tasks` processing-module `DynamicTable`
  — one in a "Behavior interface (`_add_spyglass_epoch_and_tasks`)" and a second,
  near-duplicate one inside `SpikeSortingMatInterface.add_to_nwbfile()`.
- **Correction (2026-06-10):** No such `_add_spyglass_epoch_and_tasks` function
  ever existed in the Behavior interface. There was no duplicate. In fact,
  `SpikeSortingMatInterface.add_to_nwbfile()` was the **only** code path that
  created `processing["tasks"]` for non-video (ephys-only) sessions like P100;
  `SpyglassVideoInterface` separately created its own `tasks` table for
  video sessions like P267. The "fix" below removed the *only* source of
  `processing["tasks"]` for P100, silently breaking `TaskEpoch.make()` for all
  no-video sessions (no exception raised — see error #15).
- **Original (incorrect) fix, now reverted:** Removed the table-construction
  block (and `protocol` parameter, `DynamicTable`/`get_module`/`numpy` imports)
  from `interfaces/spike_sorting_mat_interface.py::add_to_nwbfile()`, and removed
  `protocol=protocol_name` from the `SpikeSorting` conversion options in
  `arc_ecephys/convert_session.py`.
- **Status:** Diagnosis and fix were both wrong. Superseded by error #15.

#### 14. `Subject.description` required by Spyglass `Subject.make()`
- **Symptom:** Spyglass's `Subject` ingestion expects a non-empty
  `nwbfile.subject.description`; arc_ecephys's `metadata.yaml` did not set one
  (only `species`/`sex`/`strain`).
- **Fix:** Added `Subject.description: "Long-Evans rat, wild-type."` to
  `arc_ecephys/metadata.yaml`.
- **Status:** Fixed and verified — `Subject` inserts cleanly for P100.

#### 15. `TaskEpoch` silently empty for P100 — error #13's fix removed the only `processing["tasks"]` source for non-video sessions
- **Symptom:** `insert_session.py` for P100 (run4, 2026-06-10) reported
  `EXIT_CODE=0` and `test_task_recording_types` passed, but a direct query
  `(sgc.TaskEpoch() & {"nwb_file_name": "sub-P100_ses-TaskSwitch4-181010a_.nwb"}).fetch(as_dict=True)`
  returned `[]`. P267 (video) returned a correct row for the same query.
- **Root cause:** `TaskEpoch.make()` reads `nwbf.processing.get("tasks")`; if
  `None` and there's no Spyglass YAML config, it logs
  `"No tasks processing module found..."` and returns early — **no exception is
  raised**, so `EXIT_CODE=0` does not catch this. After error #13's (incorrect)
  fix removed the table-construction block from
  `SpikeSortingMatInterface.add_to_nwbfile()`, P100's NWB file had no
  `processing["tasks"]` at all (only `SpyglassVideoInterface` created one, and
  P100 has no video).
- **Fix:** Created a shared, idempotent helper
  `interfaces/_spyglass_tasks.py::add_spyglass_task_table(nwbfile, protocol,
  camera_id=None, task_epochs=None)` that builds the
  `processing["tasks"][protocol]` `DynamicTable` with the columns
  `Task`/`TaskEpoch` require (`task_name`, `task_description`, `task_type`,
  `task_subtype`, `task_environment`, `camera_id`, `task_epochs`), and is a
  no-op if `protocol` already has a table (so an ephys+video session run
  through both interfaces doesn't get a duplicate).
  - `spike_sorting_mat_interface.py::add_to_nwbfile()`: restored the `protocol:
    str = "Ecephys"` parameter; calls
    `add_spyglass_task_table(nwbfile, protocol=protocol,
    camera_id=np.array([], dtype=np.int32))` after `nwbfile.add_epoch(...)`
    (no camera info available at this point).
  - `arc_ecephys/convert_session.py`: restored
    `conversion_options["SpikeSorting"] = dict(stub_test=stub_test,
    protocol=protocol_name)`.
  - `spyglass_video_interface.py`: replaced its inline `tasks_module`/
    `task_table` construction with
    `add_spyglass_task_table(nwbfile, protocol=protocol, camera_id=[1])`
    (same shared helper, so an ephys+video session gets one consistent table).
- **Status:** Fixed and verified — P100's regenerated NWB has
  `processing["tasks"]["TaskSwitch4"]` with `camera_id=[]`,
  `task_environment="behavioral_box"`. `sgc.TaskEpoch.populate()` now inserts a
  row (`epoch=1`, `task_name='TaskSwitch4'`, `interval_list_name='01'`,
  `camera_names=[]`), and `Task` table now has both `TaskSwitch4` and
  `TaskSwitch6`. Full `insert_session.py --clean_existing=True` re-run confirmed
  end-to-end (see error #16 for the companion Spyglass-fork fix needed for this
  to work).

#### 16. Spyglass-fork-drift: errors #4 and #5 had regressed in the commit pinned by `pyproject.toml`
- **Symptom:** While testing the error #15 fix, `sgc.TaskEpoch.populate()` for
  P100 raised `ValueError: The truth value of an empty array is ambiguous` from
  `_get_valid_camera_names`'s `if camera_ids:` on `np.array([],
  dtype=int32)` — i.e. error #5 (Phase 8, 2026-04-09), already documented above
  as fixed, was happening again.
- **Root cause:** `pyproject.toml`/`uv.lock` pinned
  `spyglass-neuro @ git+https://github.com/weiglszonja/spyglass.git@ndx-structured-behavior`
  resolving to commit `2d94afed72a6a988b6d0118c3b8c733d230878b7`. This commit is
  an **orphaned/superseded** commit — `git merge-base --is-ancestor 2d94afed
  ... myfork/ndx-structured-behavior` is `false`. The branch's actual current
  tip (`95cb3acc`) contains the errors #4/#5 fixes (and a `common_task_rec.py`
  schema/type fix, see below) but `2d94afed` predates/lacks them. `uv pip
  install --editable .` (used for the fresh-venv handover test) re-resolves the
  git branch ref, which can pick up commits not equal to `uv.lock`'s pinned rev
  — this fresh venv ended up with `95cb3acc`-based code initially, then with
  `uv.lock` pinning `2d94afed` after a `uv lock --upgrade-package
  spyglass-neuro`, regressing back to the broken `if camera_ids:` /
  conditional-`camera_names` code.
- **Fix:** Created branch `pagan-lab-to-nwb-fixes-v2` on `myfork`
  (`weiglszonja/spyglass`), based on `95cb3acc` (the branch's actual tip, which
  already has the correct `common_task_rec.py` — `schema =
  dj.schema("common_task_rec")` and `expression=NULL: varchar(2000)`, not the
  broken `cbroz_common_task_rec` / `text` type that `2d94afed` has). On top of
  that, re-applied the errors #4/#5 fixes to `common_task.py`:
  - `_get_valid_camera_names`: `if camera_ids:` → `if len(camera_ids) > 0:`
    (error #5).
  - `make()`: both `if camera_names_list: key["camera_names"] = ...` /
    `task_key["camera_names"] = ...` → unconditional
    `key["camera_names"] = camera_names_list or []` /
    `task_key["camera_names"] = camera_names_list or []`, with a comment
    `# camera_names has no SQL default; always set it, even when empty.`
    (error #4).
  - Committed (`6d8e4839`) and pushed to `myfork/pagan-lab-to-nwb-fixes-v2`.
    `pyproject.toml` now pins
    `spyglass-neuro @ git+https://github.com/weiglszonja/spyglass.git@pagan-lab-to-nwb-fixes-v2`;
    `uv.lock` regenerated via `uv lock --upgrade-package spyglass-neuro`
    (resolves to `6d8e4839c53846470951603522e2677db26bc9a3`,
    `spyglass-neuro==0.1.dev2245+g6d8e4839c`).
- **Note:** A stale branch `pagan-lab-to-nwb-fixes` (singular, based on the
  broken `2d94afed`) also exists on `myfork` from an earlier attempt at this
  fix and is unused — safe to delete.
- **Status:** Fixed and verified — `sgc.TaskEpoch.populate()` for P100 now
  succeeds with no error and inserts the expected row (see error #15).

### Expected warnings (non-blocking)

| Warning | Session | Reason |
|---------|---------|--------|
| `No conforming camera device metadata found` | P100 | No video → no CameraDevice in NWB |
| `No conforming probe metadata found` | P267 | Behavior-only → no Probe in NWB |
| `No conforming data acquisition device metadata found` | P267 | Behavior-only → no DataAcqDevice |
| `No camera device found with ID [1] in NWB file` | P267 | CameraDevice pre-seeded in DB only, not in NWB file devices |
| `Skipping entry in common_ephys._raw` | Both | Raw `.rec` files not included |
| `Found overlap(s). Use no_overlap flag` | P267 | Epoch start/stop both 0.0 (placeholder) |
| `Unable to import SampleCount` | Both | SpikeGadgets sample_count not in source |
| `No conforming behavioral events data interface found` | Both | No DIOEvents in BControl NWB |

---

## Open Questions

See `open_questions.md` for full details.

**Resolved (2026-04-21):**
- Q1 ✓ `tim` row labels confirmed
- Q3/Q10 ✓ No sync signal — nominal timestamps are final
- Q5 ✓ Raw `.rec` files not currently accessible; mat-only pipeline is final for now.
  Update (2026-05-19): `SpyglassSpikeGadgetsRecordingInterface` + `spikegadgets_file_path`
  implemented on `add_spikegadgets_code` so `.rec` files can be added with no further code
  changes once access is restored — but this path is untested (no `.rec` files, no Spyglass
  `Raw`-table insertion run yet)
- Q6 ✓ Correct behavior file received and used (`181010a`)
- Q7 (partial) ✓ Device is HH128 (SpikeGadgets); amplifier/ADC chip unconfirmed (assumed Intan)

**Still open:**
1. **Q2** — Tetrode → brain region mapping (critical for electrode location)
2. **Q7** — Amplifier chip manufacturer/model (assumed "Intan" — needs lab confirmation)
3. **Q8** — Tetrode wire / contact size (mm)
4. **Q9** — `meters_per_pixel` and `lens` for CameraDevice
5. **Q11** — `camera_name` registration in Spyglass DB
