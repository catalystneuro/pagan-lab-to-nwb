# Conversion Issues — Pagan Lab to NWB

Documented bugs, root causes, fixes, and known data quirks encountered during
conversion of all protocols to NWB for DANDI:001550.

---

## 1. Null bytes in `CommentsSection_comments` (TaskSwitch2, subject P007)

**Affected files:** 86/107 `data_@TaskSwitch2_Marino_P007_*.mat` files
**Symptom:**
```
Exception: Could not create scalar dataset notes in /general
  ValueError: VLEN strings do not support embedded NULLs
```
**Root cause:** BControl wrote placeholder comment strings with embedded null
bytes (`\x00`) for numeric fields that had not yet been formatted at the time
of saving. Example from `CommentsSection_comments`:
```python
['*** 25-Jun-2015 *** ', '                    ', 'Final stage: Stage 2',
 '%hit dir: \x00         ', '%hit freq: \x00        ']
```
The null characters are legal in MATLAB strings but invalid in HDF5 VLEN
string datasets, causing the NWB write to fail.

**Fix:** In `interfaces/bcontroldatainterface.py` `get_metadata()`, strip null
bytes before joining comment strings into `NWBFile.notes`:
```python
"\n".join(s.replace("\x00", "").strip() for s in comments_arr.tolist())
```
**Status:** Fixed in `bcontroldatainterface.py`. All 107 P007 TaskSwitch2
files now convert successfully.

---

## 2. "Ended at" abort sessions excluded from conversion

**Affected files:** Sporadic across protocols (e.g., 1 TaskSwitch4new, 5
TaskSwitch4double, 6 TaskSwitch4double).
**Symptom:**
```
ValueError: Could not extract session start time from protocol title:
  'TaskSwitch4new: experimenter, ratname, Ended at 11:59'.
  Expected format is 'Started at HH:MM'.
```
**Root cause:** BControl saves "Ended at HH:MM" in the protocol title when the
session was aborted before starting. These files contain no trial data and
cannot have a meaningful `session_start_time`.
**Fix:** By design — these files are intentionally excluded. The check is in
`bcontroldatainterface.py::get_metadata()`.
**Status:** Known exclusion; documented in nwbinspector_report.md.

---

## 3. Empty `ActionsTable` / `EventsTable` (BEST_PRACTICE_VIOLATION)

**Affected files:** 4 files with empty `ActionsTable`; 2 also have empty
`EventsTable`.

| File | Notes |
|---|---|
| `sub-P116_ses-TaskSwitch6-190719b.nwb` | Short/abort session |
| `sub-P187_ses-TaskSwitch6-190808a.nwb` | Also has empty EventsTable |
| `sub-P189_ses-TaskSwitch6-190806a.nwb` | Also has empty EventsTable |
| `sub-P189_ses-TaskSwitch6-200803a.nwb` | Short/abort session |

**Root cause:** ndx-structured-behavior creates the table schema even when
no rows are added. Would require skipping table creation when empty, or
upstream changes to the extension.
**Status:** Known limitation; not a read error. Files are fully valid.

---

## 4. `TaskArgumentsTable.expression` JSON dict string (BEST_PRACTICE_VIOLATION)

**Affected file:** `sub-P100_ses-TaskSwitch6-190610a.nwb`
**Symptom:** nwbinspector `check_table_values_for_dict` — the `expression`
column contains a JSON-loadable dict string.
**Root cause:** Some BControl task argument expressions are nested structures;
we serialize them as JSON strings for flexibility.
**Status:** Known limitation; intentional. No action taken.

---

## 5. `'no description'` placeholders (BEST_PRACTICE_SUGGESTION)

14,481 occurrences across ~1,494 TaskSwitch6 files (and 271 in 96
TaskSwitch4double files) where `TaskArgumentsTable` columns use
`description: 'no description'` because the BControl YAML did not provide a
human-readable tooltip.

**Resolution:** Descriptions for the most common columns (all `HistorySection`
history variables, 3 TaskSwitch4double-specific columns) have been added to
`arc_behavior/task_switch6_params.yaml` under version control. The 67
remaining column names with missing descriptions are low-priority parameters
(display toggles, internal counters).
**Status:** Partially resolved. Remaining columns documented in
`nwbinspector_report.md §3`.

---

## 6. Mixed int/dict values in `StimulusSection_ThisStimulus` (TaskSwitch2, P007)

**Affected files:** 19/107 `data_@TaskSwitch2_Marino_P007_15071*.mat` and
nearby files (later sessions of P007's TaskSwitch2 training).
**Symptom:**
```
TypeError: int() argument must be a string, a bytes-like object or a real number, not 'dict'
```
at `h5py` VLEN write of dataset `crosstalk_dir`.
**Root cause:** In these sessions, the `StimulusSection_ThisStimulus` struct
array has `crosstalk_dir` values that are scalar ints for the first ~54 trials
but switch to dicts (e.g., `{'lpos': 350}`) from trial 54 onward. The
`_stimulus.py` check only inspected `first_val` (trial 0, an int), so the
mixed-type column passed validation and failed at write time.
**Fix:** Changed the check in `interfaces/_stimulus.py` from
`isinstance(first_val, dict)` to `any(isinstance(v, dict) for v in
stimulus_values)`. Also updated the numpy array check to handle per-element
conversion (no longer relies on `first_val`).
**Status:** Fixed. All 107 P007 TaskSwitch2 files now convert successfully.

---

## 7. Sporadic stimulus column edge cases (isolated sessions)

**Affected files (known):**
- `sub-P124_ses-TaskSwitch6-190726a` — `AssertionError: cannot pass non-empty index with empty data to index`
- `sub-P131_ses-TaskSwitch6-190617a` — `TypeError: 'int' object is not iterable`

**Root cause:** Isolated malformed stimulus fields in specific sessions:
- The first error occurs when a ragged (indexed) column has a non-empty index
  but empty data array — a corrupt or partial trial record.
- The second error occurs when a stimulus field expected to be a list is a
  bare scalar integer (likely a single-trial session or BControl quirk).

**Fix:** These are data-specific one-off anomalies. No general fix applied;
the sessions are excluded from the dataset.
**Status:** Known exclusions. Error location: `interfaces/_stimulus.py`.

**Note on remaining TaskSwitch6 failures:** All other 15 TaskSwitch6 excluded sessions
(P131_190807a, P131_191023a, P127_190610a, P116_190719a, P116_190607a, P187_190831b,
P187_190801b, P187_190627b, and all 7 P189 sessions in 2021) are confirmed "Ended at"
aborts — intentional exclusions. No additional data-specific fixes are needed.

---

## 8. `wavelength_range_in_nm` collapses to length 1 for single-wavelength lasers

**Affected files:** All TaskSwitch6 sessions with active optogenetics (e.g.
`sub-P131_ses-TaskSwitch6-*.nwb`).
**Symptom:**
```
ValueError: CustomClassGenerator.set_init.<locals>.__init__: incorrect shape for
wavelength_range_in_nm: got (1,), and expected [2]
```
**Root cause:** `_bcontrol_metadata.yaml` correctly specifies
`wavelength_range_in_nm: [450.0, 450.0]`, but
`dict_deep_update(metadata, editable_metadata)` (used in
`bcontroldatainterface.get_metadata()`) deduplicates identical list elements
when merging into an empty dict, collapsing `[450.0, 450.0]` to `[450.0]`.
`ndx-optogenetics==0.3.0`'s `ExcitationSourceModel` requires shape `(2,)`.
**Fix:** In `interfaces/_optogenetics.py`, before constructing
`ExcitationSourceModel`, the `wavelength_range_in_nm` list is duplicated back
to length 2 if `dict_deep_update` collapsed it to length 1.
**Status:** Fixed.

**Wavelength value correction (2026-06-11):** The placeholder value of 473.0 nm
(a generic "blue DPSS laser" assumption) was replaced with 450.0 nm across
`_bcontrol_metadata.yaml` (`excitation_source_model.wavelength_range_in_nm`,
`laser_device.description`, `stimulation.wavelength_in_nm`,
`stimulus_sites.*.excitation_lambda`) and `data_manifest.md`. Source: Mah et al.
2024 (Nature, doi:10.1038/s41586-024-08433-6), Extended Data Fig. 4f,g caption,
which states the FOF inactivation experiments used "blue light (450 nm, 25 mW)"
via AAV2/5-mDlx-ChR2-mCherry and the Cerebro wireless system — matching the
25 mW power already in the metadata. No manufacturer datasheet for the Cerebro
laser diode giving an actual spectral *range*/tolerance was found, so the value
remains a single wavelength duplicated to satisfy the shape-(2,) requirement.

---

## 9. Spyglass `OptogeneticProtocol` insertion fails for per-trial `optogenetic_epochs`

**Affected files:** All sessions with active optogenetics (e.g.
`sub-P131_ses-TaskSwitch6-190815a.nwb`), when inserted into Spyglass.
**Symptom:**
```
[ERROR] Spyglass: Errors occurred during population for sub-P131_ses-TaskSwitch6-190815a_.nwb:
	Failed tables ['OptogeneticProtocol']
AttributeError: 'Pandas' object has no attribute 'epoch_number'
```
**Root cause:** Our optogenetics epochs table (`ndx-optogenetics`
`OptogeneticEpochsTable`) has **one row per stimulation interval** (the
documented structure used for DANDI:001550 and
`tutorials/arc_behavior_optogenetics_notebook.ipynb`). Spyglass's
`OptogeneticProtocol.make()` does an exact-name lookup
`nwb.intervals.get("optogenetic_epochs", None)` and, if found, expects **one
row per `TaskEpoch`** with extra `epoch_number`/`convenience_code` columns and
a primary key of `(nwb_file_name, epoch)` — incompatible with our per-trial
structure both in column names and in row cardinality. At the time this error
was hit, `_bcontrol_metadata.yaml`'s `epochs_table.name` had reverted to
`optogenetic_epochs` during the April 2026 metadata centralization
(`0af6155`), silently undoing an earlier rename to `opto_epochs` (May 2026,
`6d8063e`) that had been applied to a since-removed metadata file —
`tutorials/arc_behavior_optogenetics_notebook.ipynb` already referenced
`opto_epochs`, so it was out of sync with the NWB files it was written
against.
**Fix:** Renamed `_bcontrol_metadata.yaml`'s `Optogenetics.epochs_table.name`
back to `opto_epochs`. With this name, `OptogeneticProtocol.make()`'s
`nwb.intervals.get("optogenetic_epochs", None)` returns `None`, so it logs a
warning and returns — no `InsertError`, and `Session`/`TaskEpoch`/
`TaskRecordingTypes`/`TaskRecording` insert normally with the standard
`rollback_on_fail=True, raise_err=True`. This also makes
`arc_behavior_optogenetics_notebook.ipynb`'s existing `opto_epochs` references
correct again.
**Status:** Fixed. `OptogeneticProtocol` is intentionally not populated for
opto sessions — the rich per-trial stimulation data remains fully present and
queryable directly from the NWB file's `opto_epochs` table.
`VirusInjection`/`OpticalFiberImplant` also insert 0 rows for opto sessions
(`missing required attribute pitch`/`hemisphere` — `FiberInsertion`/
`ViralVectorInjection` in `_optogenetics.py` don't set these fields); this is
a separate, non-fatal gap, not yet investigated.

**Towards full Spyglass `OptogeneticProtocol` compatibility (not implemented):**
Populating `OptogeneticProtocol` would require an *additional*,
session/epoch-level table literally named `optogenetic_epochs` with exactly
one row per `TaskEpoch` (these sessions have one epoch), containing
`epoch_number`, `convenience_code`, `pulse_length_in_ms`,
`number_pulses_per_pulse_train`, `period_in_ms`, `intertrain_interval_in_ms`,
`power_in_mW`, and a `stimulus_signal` reference (an NWB `TimeSeries`/`DIO`
object whose `object_id` becomes `OptogeneticProtocol.stimulus_object_id`) —
alongside (not instead of) the existing per-trial `opto_epochs` table. This
needs lab input on what session-level "protocol" values to report when
per-trial parameters vary (window type, L/R power), plus a new `TimeSeries`
for `stimulus_signal`. Worth scoping as a follow-up if Spyglass-side
optogenetics queries become a priority.

---
