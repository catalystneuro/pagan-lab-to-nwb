# Conversion & Upload Progress — Pagan Lab → DANDI:001550

**Last updated:** 2026-04-15

---

## Overall Status

| | Count |
|---|---|
| Local NWB files converted | **16,113** |
| Files on DANDI (001550) | **16,113** |
| Total source .mat files | ~16,206 |
| Intentionally excluded | ~93 |

**Upload is complete.** All 16,113 converted NWB files are on DANDI.

**Note:** The nwbinspector crash (IndexError on empty HDF5 datasets — issue #5 in `conversion_issues.md`) affects 8 specific files that are on DANDI, uploaded with `--validation ignore`. These are valid files; the crash is a known nwbinspector 0.7.1 bug.

---

## Per-Protocol Breakdown

| Protocol | Source .mat | Converted NWB | Excluded | Exclusion reason |
|---|---|---|---|---|
| TaskSwitch6 | 1,565 | 1,548 | 17 | See §Exclusions below |
| TaskSwitch4 | 13,841 | 13,777 | 64 | 61 "Ended at" aborts, 3 zero-trial |
| TaskSwitch2 | 201 | 201 | 0 | All converted after null-byte + mixed-type fixes |
| TaskSwitch3 | 112 | 111 | 1 | "Ended at" abort |
| TaskSwitch4double | 102 | 96 | 6 | "Ended at" aborts |
| TaskSwitch4new | 87 | 86 | 1 | "Ended at" abort |
| TaskSwitch4repeat | 14 | 13 | 1 | "Ended at" abort |
| ProAnti3Marino | 174 | 173 | 1 | "Ended at" abort |
| PBups | 75 | 73 | 2 | "Ended at" aborts |
| ProAnti3 | 16 | 16 | 0 | Complete |
| TaskSwitch | 19 | 19 | 0 | Complete |
| **Total** | **~16,206** | **16,113** | **~93** | |

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

## TaskSwitch6 — Exclusion Breakdown (17 files)

All 17 excluded TaskSwitch6 sessions are confirmed:

| Subject | File | Reason |
|---|---|---|
| P189 | 210529b, 210529a, 210523b, 210518b, 210516b, 210516a, 210328a | "Ended at" abort |
| P187 | 190831b, 190801b, 190627b | "Ended at" abort |
| P131 | 190807a, 191023a | "Ended at" abort |
| P127 | 190610a | "Ended at" abort |
| P116 | 190719a, 190607a | "Ended at" abort |
| P124 | 190726a | AssertionError: empty ragged column data (data anomaly, issue #10) |
| P131 | 190617a | TypeError: 'int' object is not iterable (data anomaly, issue #10) |

**15 of 17 are "Ended at" aborts.** The remaining 2 (P124 and P131_190617a) are
data-specific anomalies documented in `conversion_issues.md §10`.

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
