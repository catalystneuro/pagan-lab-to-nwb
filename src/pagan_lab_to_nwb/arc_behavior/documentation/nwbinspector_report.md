# NWBInspector Report — Pagan Lab Conversion

**Tool:** NWBInspector 0.7.1, `--config dandi`
**Environment:** `.venv` (pynwb 3.1.3, ndx-optogenetics 0.3.0, neuroconv 0.9.3)
**Dandiset:** [DANDI:001550](https://dandiarchive.org/dandiset/001550)

---

# All Protocols Combined

**Generated:** 2026-04-07
**Files scanned:** 2,336 (10 protocols; TaskSwitch4 deferred)

---

## Summary

| Severity | Count | Status |
|---|---|---|
| ERROR | 36 | nwbinspector internal crash (inspector bug, not file error) |
| PYNWB_VALIDATION | 0 | — |
| BEST_PRACTICE_VIOLATION | 620 | See breakdown below |
| BEST_PRACTICE_SUGGESTION | 5,392 | See breakdown below |

`dandi validate` passed for all 2,336 files (exit code 0, zero errors).

---

## 0. ERROR — nwbinspector internal crash (36 occurrences, 8 files)

**Checks:** `check_table_values_for_dict` (18) / `check_col_not_nan` (18)
**Cause:** nwbinspector 0.7.1 still crashes with `IndexError` when trying to
index into empty HDF5 datasets. This is a known inspector bug — the files
themselves are valid.
**Files affected:** The same 8 files with empty `ActionsTable` or `EventsTable`
(see §2.3 below).

---

## 1. PYNWB_VALIDATION — None

All files pass PyNWB schema validation.

---

## 2. BEST_PRACTICE_VIOLATION (620 occurrences)

### 2.1 `check_time_intervals_stop_after_start` — zero-duration states (304 files)

**Check:** `StatesTable` has rows where `stop_time == start_time` (diff = 0.0).
**Root cause:** Some BControl state machine states are instantaneous —
specifically `check_next_trial_ready`, which is a synchronization state that
the FSM passes through in a single clock tick. These states legitimately have
zero duration; the start and stop time are identical by design.
**Subjects affected:** P189 (172 files), P187 (87), P131 (15), P013 (14),
P116 (3), P007 (3), P124 (2), P100 (2), P078 (1), P049 (1).
**Protocols affected:** TaskSwitch6 (277), ProAnti3Marino (9), TaskSwitch4new
(8), TaskSwitch3 (6), TaskSwitch2 (2), TaskSwitch4double (1), PBups (1).
**Status:** Known data characteristic. Falsifying data by adding an epsilon
to stop_time would be incorrect. Not actionable.

### 2.2 `check_table_values_for_dict` — JSON dict strings (87 files)

**Check:** `TaskArgumentsTable.expression` contains a JSON-loadable dict string.
**Root cause:** BControl task argument expressions can be nested structures
(e.g., stimulus parameter dicts, cross-talk matrices). These are serialised as
JSON strings in the `expression` column for flexibility.
**Protocols affected:** PBups (292 occurrences, most files), TaskSwitch4repeat
(13), TaskSwitch6 (1).
**Status:** Known limitation — intentional design. Not actionable.

### 2.3 `check_empty_table` — empty `ActionsTable` / `EventsTable` (8 files)

These sessions have tables with the schema but no rows — BControl did not log
any actions/events during these sessions (short or abort sessions).

| File | Tables empty |
|---|---|
| `sub-P116_ses-TaskSwitch6-190719b.nwb` | ActionsTable |
| `sub-P187_ses-TaskSwitch6-190808a.nwb` | ActionsTable + EventsTable |
| `sub-P189_ses-TaskSwitch6-190806a.nwb` | ActionsTable + EventsTable |
| `sub-P189_ses-TaskSwitch6-200803a.nwb` | ActionsTable |
| *(4 more files)* | ActionsTable |

**Status:** Structural — ndx-structured-behavior creates the schema even when
no rows are added. Would require upstream changes to the extension to skip
empty-table creation. Not a read error; files are fully valid.

---

## 3. BEST_PRACTICE_SUGGESTION (5,392 occurrences)

### 3.1 `check_description` — `'no description'` placeholders (4,539, 925 files)

Task argument columns in `TaskArgumentsTable` use `description: 'no description'`
when the BControl YAML does not supply a human-readable description.
Descriptions for the most common `HistorySection` history variables have been
added to `arc_behavior/task_switch6_params.yaml`. 67 remaining column names
(display toggles, internal counters) are low-priority.
**Status:** Partially resolved. Remaining columns documented in YAML.

### 3.2 `check_column_binary_capability` — integer binary columns (590, all in `trials`)

`OptoSection_opto_connected` uses integer dtype but holds only 0/1 values.
Could be stored as boolean. Low priority — values are unambiguous.
**Status:** Known; not actionable without changing the conversion interface.

### 3.3 `check_col_not_nan` — all-NaN columns (237)

- `optogenetic_epochs.intertrain_interval_in_ms` (236 files): single-pulse
  stimulation always has NaN intertrain interval by design.
- `trials.cpoke_start_time` (1 file): session where rat never poked.

**Status:** Both are data-correct NaN values. Not actionable.

### 3.4 `check_single_row` — tables with a single row (26)

Low-priority suggestion; not actionable.

---

# Per-Protocol Breakdown

| Protocol | Files | PYNWB_VALIDATION | BEST_PRACTICE_VIOLATION | BEST_PRACTICE_SUGGESTION |
|---|---|---|---|---|
| TaskSwitch6 | 1,548 | 0 | 278 (stop_after_start) + 1 (dict) | ~3,800 |
| TaskSwitch2 | 201 | 0 | 2 (stop_after_start) | ~200 |
| ProAnti3Marino | 173 | 0 | 9 (stop_after_start) | ~170 |
| TaskSwitch3 | 111 | 0 | 6 (stop_after_start) | ~110 |
| TaskSwitch4double | 96 | 0 | 1 (stop_after_start) | ~100 |
| TaskSwitch4new | 86 | 0 | 8 (stop_after_start) | ~90 |
| PBups | 73 | 0 | 292 (dict) + 1 (stop_after_start) | ~75 |
| TaskSwitch | 19 | 0 | 0 | ~20 |
| ProAnti3 | 16 | 0 | 0 | ~15 |
| TaskSwitch4repeat | 13 | 0 | 13 (dict) | ~15 |
