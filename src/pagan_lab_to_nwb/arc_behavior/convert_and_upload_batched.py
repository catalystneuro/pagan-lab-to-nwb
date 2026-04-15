"""Batch conversion + DANDI upload for any BControl protocol.

For each batch of .mat files:
  1. Convert .mat → NWB  (skips files whose NWB already exists on disk)
  2. Run ``dandi validate`` on every new NWB file (PyNWB schema + nwbinspector).
     Files that fail genuine validation are logged and NOT uploaded.
  3. Upload validated files.  ``--validation ignore`` is passed to ``dandi upload``
     only for files that already passed step 2; this avoids a known nwbinspector 0.7.1
     bug (IndexError on empty HDF5 datasets) that crashes the upload's internal
     re-validation but does not indicate a real file error.

Usage
-----
# Dry-run — print batch plan without converting:
.venv/bin/python src/pagan_lab_to_nwb/arc_behavior/convert_and_upload_batched.py --protocol ProAnti3 --dry-run

# Convert + upload all TaskSwitch6 sessions:
.venv/bin/python src/pagan_lab_to_nwb/arc_behavior/convert_and_upload_batched.py --protocol TaskSwitch6

# Resume from batch 5:
.venv/bin/python src/pagan_lab_to_nwb/arc_behavior/convert_and_upload_batched.py --protocol TaskSwitch6 --start-batch 5

# Override default data / output directories:
.venv/bin/python src/pagan_lab_to_nwb/arc_behavior/convert_and_upload_batched.py \\
    --protocol TaskSwitch2 \\
    --data-dir /Volumes/T9/data/Pagan_latest_data_share \\
    --output-dir /Users/weian/data/Pagan_nwbfiles_to_upload/001550

Notes
-----
The skip-if-exists check compares the expected NWB output path against disk.  If
``session_to_nwb`` produces a different filename for a given protocol than the pattern
``sub-{subject}_ses-{protocol}-{date}.nwb``, the check will miss already-converted
files and attempt re-conversion.  Because ``overwrite=False`` is set this is harmless
(the attempt returns immediately), but it will slow down the batch.  Verify the output
filename for a new protocol with a ``--dry-run`` + manual spot-check before a full run.
"""

import argparse
import re
import subprocess
import sys
import traceback
from pathlib import Path

from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parents[3]
_VENV_DANDI = _REPO_ROOT / ".venv" / "bin" / "dandi"

_DEFAULT_DATA_DIR = Path("/Volumes/T9/data/Pagan_latest_data_share")
_DEFAULT_OUTPUT_DIR = Path("/Users/weian/data/Pagan_nwbfiles_to_upload/001550")

# ── Import conversion helpers (requires running with .venv/bin/python) ────────
sys.path.insert(0, str(_REPO_ROOT / "src"))
from pagan_lab_to_nwb.arc_behavior.convert_session import (  # noqa: E402
    _TASK_PARAMS_YAML_PATH,
    session_to_nwb,
)

# Signature of the known nwbinspector 0.7.1 crash on empty HDF5 datasets.
# This is an inspector bug, not a file error.  Files that fail *only* due to this
# message are still valid and safe to upload.
_INSPECTOR_CRASH_PATTERN = re.compile(
    r"IndexError.*out of range for empty dimension|"
    r"check_table_values_for_dict.*IndexError|"
    r"check_col_not_nan.*IndexError",
    re.IGNORECASE,
)

# Stem parser for BControl filenames: data_@{protocol}_{experimenter}_{subject}_{date}.mat
_STEM_PATTERN = re.compile(r"^(?P<protocol>[^_]+)_(?P<experimenter>[^_]+)_(?P<subject_id>[^_]+)_(?P<date_str>.+)$")


def _get_mat_files(data_dir: Path, protocol: str) -> list[Path]:
    """Return sorted list of .mat files for the given protocol."""
    return sorted(data_dir.rglob(f"data_@{protocol}_*.mat"))


def _nwb_path_for(mat_file: Path, output_dir: Path) -> Path | None:
    """Return the expected NWB output path for a .mat file, or None if unparseable."""
    stem = mat_file.stem.replace("data_@", "")
    m = _STEM_PATTERN.match(stem)
    if not m:
        return None
    subject_id = m.group("subject_id")
    protocol = m.group("protocol")
    date_str = m.group("date_str")
    nwb_name = f"sub-{subject_id}_ses-{protocol}-{date_str}.nwb"
    return output_dir / f"sub-{subject_id}" / nwb_name


def convert_batch(batch_files: list[Path], output_dir: Path, error_log: Path) -> list[Path]:
    """Convert a batch of .mat files. Returns list of successfully created NWB paths."""
    nwb_paths = []
    for mat_file in tqdm(batch_files, desc="  Converting", leave=False):
        try:
            nwb_path = session_to_nwb(
                file_path=mat_file,
                nwb_folder_path=output_dir,
                task_params_file_path=_TASK_PARAMS_YAML_PATH,
                stub_test=False,
                overwrite=False,
            )
            nwb_paths.append(nwb_path)
        except Exception:
            with error_log.open("a") as f:
                f.write(f"\n--- Conversion error: {mat_file.name} ---\n")
                f.write(traceback.format_exc())
    return nwb_paths


def validate_batch(nwb_paths: list[Path], validate_error_log: Path) -> tuple[list[Path], list[Path]]:
    """Run ``dandi validate`` on each NWB file.

    Returns
    -------
    valid_paths : list[Path]
        Files that passed validation (exit 0) or whose only failures are the
        known nwbinspector 0.7.1 IndexError crash (inspector bug, not file error).
    invalid_paths : list[Path]
        Files with genuine validation errors — do not upload.
    """
    valid_paths = []
    invalid_paths = []

    for nwb_path in tqdm(nwb_paths, desc="  Validating", leave=False):
        result = subprocess.run(
            [str(_VENV_DANDI), "validate", str(nwb_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            valid_paths.append(nwb_path)
            continue

        # Non-zero exit: check whether the only errors are the known inspector crash.
        combined_output = result.stdout + result.stderr
        stripped = _INSPECTOR_CRASH_PATTERN.sub("", combined_output)
        has_real_error = bool(re.search(r"\bERROR\b|\bFAIL\b|\bInvalid\b|\bschema\b", stripped, re.IGNORECASE))

        if not has_real_error:
            valid_paths.append(nwb_path)
        else:
            invalid_paths.append(nwb_path)
            with validate_error_log.open("a") as f:
                f.write(f"\n--- Validation error: {nwb_path.name} ---\n")
                f.write(combined_output)

    return valid_paths, invalid_paths


def upload_files(nwb_paths: list[Path], output_dir: Path) -> bool:
    """Upload pre-validated NWB files to DANDI.

    Uses ``--validation ignore`` because validation was already performed by
    ``dandi validate`` in the previous step.  This avoids the nwbinspector 0.7.1
    IndexError bug in the upload's internal re-validation.

    Returns True if the upload command exited 0.
    """
    if not nwb_paths:
        print("  No files to upload.")
        return True
    rel_paths = [str(p.relative_to(output_dir)) for p in nwb_paths]
    cmd = [str(_VENV_DANDI), "upload", "--validation", "ignore"] + rel_paths
    result = subprocess.run(cmd, cwd=str(output_dir))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Batch-convert, validate, and upload sessions for a given BControl protocol."
    )
    parser.add_argument(
        "--protocol",
        required=True,
        help=(
            "Protocol name as it appears in .mat filenames, e.g. TaskSwitch6, ProAnti3, "
            "TaskSwitch2, TaskSwitch3, TaskSwitch4double, TaskSwitch4new, "
            "TaskSwitch4repeat, ProAnti3Marino, PBups, TaskSwitch"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        help=f"Root directory containing BControl .mat files (default: {_DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Directory where NWB files are saved and uploaded from (default: {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--batch-size", type=int, default=150, help="Files per batch (default: 150)")
    parser.add_argument("--start-batch", type=int, default=0, help="Batch index to start from (default: 0)")
    parser.add_argument("--dry-run", action="store_true", help="Print batch plan without converting")
    args = parser.parse_args()

    protocol = args.protocol
    output_dir: Path = args.output_dir
    convert_error_log = output_dir / f"conversion_errors_{protocol}.txt"
    validate_error_log = output_dir / f"validation_errors_{protocol}.txt"

    all_files = _get_mat_files(args.data_dir, protocol)
    if not all_files:
        print(f"No .mat files found for protocol '{protocol}' in {args.data_dir}")
        sys.exit(1)

    total = len(all_files)
    n_batches = (total + args.batch_size - 1) // args.batch_size

    print(f"{protocol}: {total} .mat files → {n_batches} batches of {args.batch_size}")
    print(f"Output dir : {output_dir}")
    print(f"Conv errors: {convert_error_log}")
    print(f"Val errors : {validate_error_log}")
    print()

    if args.dry_run:
        for i in range(n_batches):
            batch = all_files[i * args.batch_size : (i + 1) * args.batch_size]
            subjects = {f.parent.name for f in batch}
            print(f"  Batch {i:3d}: {len(batch):3d} files | subjects: {', '.join(sorted(subjects))}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    total_converted = 0
    total_invalid = 0
    total_conv_errors = 0

    for batch_idx in range(args.start_batch, n_batches):
        batch_files = all_files[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]
        subjects = sorted({f.parent.name for f in batch_files})

        # Skip files whose NWB already exists on disk
        to_convert = [f for f in batch_files if not (_nwb_path_for(f, output_dir) or Path("/nonexistent")).exists()]
        already_done = len(batch_files) - len(to_convert)

        print(
            f"\n{'─'*60}\n"
            f"Batch {batch_idx + 1}/{n_batches}  "
            f"({len(batch_files)} files, {already_done} already converted)  "
            f"subjects: {', '.join(subjects)}"
        )

        if not to_convert:
            print("  All files already converted and uploaded. Skipping.")
            continue

        # Step 1: Convert
        nwb_paths = convert_batch(to_convert, output_dir, convert_error_log)
        conv_errors = len(to_convert) - len(nwb_paths)
        total_converted += len(nwb_paths)
        total_conv_errors += conv_errors
        print(f"  Converted : {len(nwb_paths)}/{len(to_convert)}  (errors: {conv_errors})")

        if not nwb_paths:
            continue

        # Step 2: Validate
        print(f"  Validating {len(nwb_paths)} files with dandi validate...")
        valid_paths, invalid_paths = validate_batch(nwb_paths, validate_error_log)
        total_invalid += len(invalid_paths)
        print(
            f"  Validated : {len(valid_paths)} OK  |  {len(invalid_paths)} genuine errors"
            + (f" (see {validate_error_log.name})" if invalid_paths else "")
        )

        # Step 3: Upload validated files
        if valid_paths:
            print(f"  Uploading {len(valid_paths)} validated files to DANDI...")
            ok = upload_files(valid_paths, output_dir)
            if not ok:
                print("  WARNING: dandi upload reported errors — check dandi logs.")

        print(
            f"  Cumulative: {total_converted} converted | "
            f"{total_conv_errors} conv errors | {total_invalid} validation errors"
        )

    print(f"\n{'='*60}")
    print(f"Done.  Protocol: {protocol}")
    print(f"  Total converted       : {total_converted}")
    print(f"  Conversion errors     : {total_conv_errors}")
    print(f"  Genuine val. errors   : {total_invalid}")
    if total_conv_errors:
        print(f"  Conv error log : {convert_error_log}")
    if total_invalid:
        print(f"  Val error log  : {validate_error_log}")


if __name__ == "__main__":
    main()
