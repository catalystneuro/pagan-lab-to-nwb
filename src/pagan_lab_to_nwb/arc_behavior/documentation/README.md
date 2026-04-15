# arc_behavior — Conversion Documentation

This folder contains reference documents produced during the BControl → NWB conversion
for [DANDI:001550](https://dandiarchive.org/dandiset/001550).

## Files

### `data_manifest.md`
Field-by-field map from BControl `.mat` source to NWB destination. Describes which
`ndx-structured-behavior` types are used (states, events, actions, trials) and how to
read each field back from the converted files. Start here if you need to understand the
NWB file structure.

### `conversion_issues.md`
Log of every bug, data quirk, and edge case encountered during conversion, with root
cause, fix applied, and current status. Organized as numbered issues. Useful for
understanding why certain files were excluded and what workarounds are in the code.

### `conversion_progress.md`
Running tally of converted and uploaded files by protocol, including the DANDI upload
status. Also contains the NWBInspector summary and per-protocol exclusion breakdowns.

### `nwbinspector_report.md`
Detailed NWBInspector 0.7.1 report for all converted files (`--config dandi`). Explains
which violations are actionable, which are by design, and which are inspector bugs.
References `conversion_issues.md` for the known empty-table crash.
