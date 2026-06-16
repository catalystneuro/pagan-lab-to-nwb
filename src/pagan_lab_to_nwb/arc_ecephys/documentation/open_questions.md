
# Arc Ephys Conversion — Open Questions

Questions to resolve with the lab before finalizing the NWB conversion.
Placeholder values are used in the prototype; update the code once confirmed.

---

## 1. `tim` row labels ✓ RESOLVED (2026-04-21)

Confirmed by lab:
```
0: end of previous trial  → previous_trial_end
1: trial ready             → trial_ready
2: start of the cue        → cue_start
3: rat pokes in            → poke_in
4: rat pokes out           → poke_out
5: rat makes choice        → choice_time
6: end of trial            → trial_end
```
**Action taken:** Updated `_TIM_COLUMN_NAMES` and `_TIM_COLUMN_DESCRIPTIONS` in `_dati_mat.py`.

---

## 2. Tetrode → brain region mapping (critical for electrode metadata)

NWB requires a `location` field for every `ElectrodeGroup`. Do you have a mapping
of tetrode number (1–32) to brain region (e.g. PFC, striatum, ACC)?

**Placeholder used:** `location = "unknown"` for all tetrode groups.

Expected answer: a table or dict, e.g.:
```
tetrodes 1–8:   PFC
tetrodes 9–16:  Striatum
tetrodes 17–24: ACC
```
If this varies by rat/session, please indicate how it is recorded (e.g. surgery notes,
a spreadsheet, embedded in the SpikeGadgets configuration file).

---

## 3. Video synchronization ✓ RESOLVED (2026-04-21)

**Lab confirmed:** There is no sync signal at the moment.

**Action taken:** Uniform timestamps derived from nominal frame rate (~19.98 fps) will be
retained as the final approach for current data. Q10 (Spyglass timestamps) is resolved
by the same answer — Spyglass VideoFile.make() may warn about epoch overlap but will not fail.

---

## 4. `rrr4` PSTH tensor — store or skip?

The `dati_@*.mat` file contains `rrr4` (shape: n_units × n_trials × 501 time bins),
a manually processed trial-aligned neural response tensor.

This is a derived artifact (not raw data). Options:
- **Store** it in NWB `processing["neural_responses"]` as a labeled array (increases file size significantly).
- **Skip** it and rely on the raw spike times in the `Units` table for any downstream analysis.

**Placeholder:** currently stored. Let us know if it should be excluded.

---

## 5. Raw SpikeGadgets data availability ✓ RESOLVED (2026-04-21)

**Lab confirmed:** Princeton backup exists but access is currently unavailable.
For the initial release, the spike-sorted `.mat` files (`spikes_@*.mat`) are the
authoritative source and will be published on DANDI.

**Future plan:** The lab wants the conversion script to make it easy to add `.rec` files
when full Princeton access is restored.

**Action taken (2026-05-19):** Implemented `SpyglassSpikeGadgetsRecordingInterface`
(`interfaces/spyglass_spikegadgets_recording_interface.py`), wired into
`ArcEcephysNWBConverter` ahead of `SpikeSorting`, and exposed via the
`spikegadgets_file_path` parameter on `session_to_nwb()`. When provided, it writes
`ElectricalSeriesRaw` to acquisition so Spyglass can populate the `Raw` table on insertion.
**Untested** — no `.rec` files are available yet, so this path has not been run end-to-end
or verified against a real Spyglass insertion.

---

## 7. SpikeGadgets acquisition device fields — PARTIALLY RESOLVED (2026-04-21)

**Lab confirmed:** Device is the Horizontal Headstage 128-Channel Datalogger (HH128) by
SpikeGadgets. Amplifier and ADC circuit are integrated inside the same unit.

**Lab confirmed:** Device is HH128; amplifier and ADC circuit are both integrated inside it.
Using "Horizontal Headstage 128-Channel Datalogger" as the value for both fields.

**Action taken:** Updated `metadata.yaml` and `insert_session.py`:
- `name`: "HH128"
- `system`: "SpikeGadgets"
- `amplifier`: "Horizontal Headstage 128-Channel Datalogger"
- `adc_circuit`: "Horizontal Headstage 128-Channel Datalogger"
- `description`: updated to mention HH128 full name

---

## 8. Tetrode contact size (Spyglass: `Probe.contact_size`)

The `Probe` object in ndx-franklab-novela requires a `contact_size` (float, in mm).
What is the wire diameter / contact size for the tetrodes used in this experiment?

**Placeholder used:** `0.0125` mm (12.5 µm, a standard nichrome tetrode wire diameter).

**Important:** this must be a real numeric value, not `float("nan")`/`None`. An
earlier version of this code used `float("nan")` (which becomes `NULL` in the
Spyglass `Probe.Electrode` table on first insert). On any *subsequent* insert
against the same `probe_id`, Spyglass's `IngestionMixin._unequal_vals()` treats a
freshly-generated `nan` as always-unequal to the stored `NULL`, triggering an
unresolvable "accept divergence" prompt (`EOFError` in non-interactive runs). See
`spyglass_notes.md` for the full incident writeup. `0.0125` was verified to
round-trip exactly through MySQL `FLOAT`, so it is safe as a long-term placeholder
until the lab confirms the real value.

---

## 9. Camera device metadata ✓ PARTIALLY RESOLVED (2026-04-21)

**Lab confirmed:**
- `model`: "HDE-Security-Camera-Vision-Pinhole" (cheap IR pinhole camera)
- `lens`: unknown — lab could not find lens specification for this camera
- `meters_per_pixel`: still unknown (depends on cage size / camera distance)

**Future plan:** Lab is transitioning to Raspberry Pi cameras for new recordings, and
will have full metadata then. Conversion script should make it easy to specify these
fields per-session in `metadata.yaml`.

**Action taken:** Updated `metadata.yaml` with confirmed model. `lens` and
`meters_per_pixel` remain as placeholders. `camera_name = "top_camera"` unchanged (Q11).

---

## 10. Video synchronization — confirmation needed (Spyglass: real timestamps)

Spyglass `VideoFile.make()` uses `ImageSeries.timestamps` for epoch matching. See also Q3.

Until the sync method is confirmed, the conversion uses uniform timestamps derived from
the nominal frame rate (~19.98 fps). This may cause Spyglass `VideoFile.make()` to warn
about epoch overlap but will not cause ingestion failure.

**Action needed:** confirm synchronization method (TTL pulses, sync LED, or none).

---

## 11. `camera_name` registration in Spyglass database

Spyglass requires `CameraDevice.camera_name` to already exist in the `sgc.CameraDevice`
DataJoint table before ingestion. The conversion writes `camera_name = "top_camera"` as a
placeholder.

**Action needed:** check what `camera_name` values are registered in the lab's Spyglass
database and update `metadata.yaml` accordingly.

---

## 6. Behavior file pairing ✓ RESOLVED (2026-04-22)

**Lab confirmed:** the `180110a` file was included by mistake. The correct behavior
file is `data_@TaskSwitch4_Marino_P100_181010a.mat` (date matches spikes/dati: `181010a`).

**Action taken:** the lab provided the correct file; it is now in `Ephys_data_examples`
and `convert_session.py` already references `181010a` for the P100 prototype session.
