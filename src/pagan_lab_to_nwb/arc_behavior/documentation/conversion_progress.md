# Conversion & Upload Progress — Pagan Lab → DANDI:001550

**Last updated:** 2026-10-22

---

## Overall Status

| | Count |
|---|---|
| Local NWB files converted | **16,115** |
| Files on DANDI (001550) | **16,115** |
| Total source .mat files | ~16,206 |
| Intentionally excluded | ~91 |

**Upload is complete.** All 16,115 converted NWB files are on DANDI.

**Re-conversion pending (issue #27):** the published files predate the 450 nm wavelength fix and the
issue #27 fixes. The per-protocol table below gives the counts after re-conversion: P189
TaskSwitch6-210516a and P013 TaskSwitch3-151113a are added, for 16,117 files and 89 exclusions.

**Note:** The nwbinspector crash (IndexError on empty HDF5 datasets — issue #5 in `conversion_issues.md`) affects 8 specific files that are on DANDI, uploaded with `--validation ignore`. These are valid files; the crash is a known nwbinspector 0.7.1 bug.

---

## Per-Protocol Breakdown

| Protocol | Source .mat | Converted NWB | Excluded | Exclusion reason |
|---|---|---|---|---|
| TaskSwitch6 | 1,565 | 1,551 | 14 | See §Exclusions below |
| TaskSwitch4 | 13,841 | 13,777 | 64 | 61 "Ended at" aborts, 3 zero-trial |
| TaskSwitch2 | 201 | 201 | 0 | All converted after null-byte + mixed-type fixes |
| TaskSwitch3 | 112 | 112 | 0 | P013/151113a previously failed with an older version of the code |
| TaskSwitch4double | 102 | 96 | 6 | 5 "Ended at" aborts, 1 zero-trial |
| TaskSwitch4new | 87 | 86 | 1 | "Ended at" abort |
| TaskSwitch4repeat | 14 | 13 | 1 | "Ended at" abort |
| ProAnti3Marino | 174 | 173 | 1 | "Ended at" abort |
| PBups | 75 | 73 | 2 | "Ended at" aborts |
| ProAnti3 | 16 | 16 | 0 | Complete |
| TaskSwitch | 19 | 19 | 0 | Complete |
| **Total** | **16,206** | **16,117** | **89** | |

---

## TaskSwitch4 — Final State (as of 2026-04-13)

| Error type | Unique files | Resolution |
|---|---|---|
| "Ended at" abort | 61 | Intentional exclusion (no trial data) |
| Zero completed trials | 3 | Intentional exclusion |
| Empty ragged column (AssertionError) | 2 | **Fixed** — files are now converted |
| Scalar int in pulse column (TypeError) | 2 | **Fixed** — files are now converted |

The 4 "fixed" files (H113_190110a, H113_190122b, P007_160220a, P007_161119a) were
re-converted after the fix was applied and their NWB files exist on disk.

---

## TaskSwitch6 — Exclusion Breakdown (14 files)

The 14 excluded TaskSwitch6 sessions are "Ended at" aborts with 0 trials:

| Subject | File | Reason |
|---|---|---|
| P189 | 210529b, 210529a, 210523b, 210518b, 210516b, 210328a | "Ended at" abort |
| P187 | 190831b, 190801b, 190627b | "Ended at" abort |
| P131 | 190807a, 191023a | "Ended at" abort |
| P127 | 190610a | "Ended at" abort |
| P116 | 190719a, 190607a | "Ended at" abort |

P189/210516a was previously listed here as an "Ended at" abort, which is wrong: its title has
"Started at 00:15" and it has 33 trials. Trials 26 and 27 (0-based, both timeout violations)
have corrupted state machine timestamps (up to 6.8e12 s), which raised a `ValueError` in
`create_states`. `BControlBehaviorInterface` now drops such trials and records them in the
file's `notes`, so the session is converted with 31 trials (27 valid, as in the paper's
session table). See issue #27. The two data-anomaly sessions previously
listed as excluded (P124/190726a — AssertionError; P131/190617a — TypeError) were
subsequently fixed by the same code fixes applied to TaskSwitch4 anomalies and are now
on DANDI. See `conversion_issues.md §10` for the original anomaly details.

---

## NWBInspector Summary (2,336 non-TaskSwitch4 files, run 2026-04-07)

| Severity | Count | Actionable? |
|---|---|---|
| ERROR (inspector crash) | 36 | No — nwbinspector 0.7.1 bug on empty tables |
| PYNWB_VALIDATION | 0 | — |
| BEST_PRACTICE_VIOLATION | 620 | No — all known, documented |
| BEST_PRACTICE_SUGGESTION | 5,392 | Partially — see below |

Key violations (all non-actionable):
- **Zero-duration states** (304 files): instantaneous FSM states by design
- **JSON dict strings in expression column** (87 files): intentional serialisation
- **Empty ActionsTable/EventsTable** (8 files): ndx-structured-behavior creates schema even with no rows; inspector bug causes crash on these

Key suggestions (partially addressable):
- **`'no description'` placeholders** (4,539): 67 low-priority columns remain without descriptions
- **`OptoSection_opto_connected` as int** (590): stores 0/1, could be boolean
- **All-NaN `intertrain_interval_in_ms`** (236): expected for single-pulse opto sessions

`dandi validate` passed for all 2,336 files (exit code 0).

---
