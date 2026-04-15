"""Functions for adding OptoSection data to the NWB file."""

import numpy as np
from ndx_ophys_devices import (
    Effector,
    FiberInsertion,
    ViralVector,
    ViralVectorInjection,
)
from ndx_optogenetics import (
    ExcitationSource,
    ExcitationSourceModel,
    OpticalFiber,
    OpticalFiberModel,
    OptogeneticEffectors,
    OptogeneticEpochsTable,
    OptogeneticExperimentMetadata,
    OptogeneticSitesTable,
    OptogeneticViruses,
    OptogeneticVirusInjections,
)
from pynwb.file import NWBFile
from pynwb.ogen import OptogeneticSeries, OptogeneticStimulusSite


def add_optogenetic_series_to_nwbfile(
    nwbfile: NWBFile,
    saved_history: dict,
    parsed_events: list[dict],
    metadata: dict,
    stub_test: bool = False,
) -> None:
    """Add optogenetics data to the NWB file for sessions with active laser stimulation.

    This is a no-op for sessions where no trial has ``OptoSection_opto_connected == 1``.

    Two ``OptogeneticSeries`` (left/right hemisphere) are written to ``nwbfile.stimulus``
    as step-functions.  Per-trial opto columns are added to ``nwbfile.trials``.
    Rich metadata (virus, fiber, injection coordinates) is stored via ``ndx-optogenetics``.

    All hardware specs, coordinates, and descriptions are read from
    ``metadata["Optogenetics"]`` (populated from arc_behavior/metadata.yaml).

    Power values in ``OptogeneticSeries`` are in watts (SI).  Conversion rule:
    raw Cerebro internal unit >= cerebro_threshold → power_in_W; otherwise 0 W (laser off).
    """
    opto_connected = saved_history.get("OptoSection_opto_connected", [])
    if not opto_connected or not any(c != 0 for c in opto_connected):
        return  # No active stimulation in this session

    n_trials = len(opto_connected)
    if stub_test:
        n_trials = min(n_trials, 100)

    opto_type = saved_history.get("OptoSection_opto_type", ["Full Trial"] * n_trials)
    opto_left_power = saved_history.get("OptoSection_opto_left_power", [0] * n_trials)
    opto_right_power = saved_history.get("OptoSection_opto_right_power", [0] * n_trials)

    opto_meta = metadata["Optogenetics"]
    power_cal = opto_meta["power_calibration"]
    threshold = power_cal["cerebro_threshold"]
    power_W = power_cal["power_in_W"]
    power_mW = power_cal["power_in_mW"]
    opto_windows = {k: tuple(v) for k, v in opto_meta["stimulation_windows"].items()}

    # ── Per-trial opto columns on TrialsTable ─────────────────────────────────
    n_table = len(nwbfile.trials)
    tc = opto_meta["trials_columns"]
    nwbfile.trials.add_column(
        name="OptoSection_opto_connected",
        description=tc["opto_connected"]["description"],
        data=[int(opto_connected[i]) for i in range(n_table)],
    )
    _opto_type_full = list(opto_type) + ["Full Trial"] * n_table
    nwbfile.trials.add_column(
        name="OptoSection_opto_type",
        description=tc["opto_type"]["description"],
        data=[str(_opto_type_full[i]) for i in range(n_table)],
    )
    # NOTE: opto_left_power / opto_right_power are intentionally NOT added as trials columns.
    # The hemisphere-resolved power is fully covered by optogenetic_series_left /
    # optogenetic_series_right (watts, step function).  Adding the raw Cerebro units
    # alongside would be redundant given the binary conversion rule.
    # See data_manifest.md §10 for the rationale and a code snippet to recover per-trial
    # hemisphere stimulation from the OptogeneticSeries.

    # ── cpoke start times (t=0 reference for opto windows) ───────────────────
    cpoke_starts = []
    for i in range(n_trials):
        try:
            cpoke_state = parsed_events[i].get("states", {}).get("cpoke", [])
            cpoke_starts.append(float(np.asarray(cpoke_state).flat[0]) if len(cpoke_state) else None)
        except (IndexError, KeyError, TypeError, AttributeError):
            cpoke_starts.append(None)

    # ── Step-function timeseries ──────────────────────────────────────────────
    left_timestamps, left_data = [], []
    right_timestamps, right_data = [], []

    for i in range(n_trials):
        if not opto_connected[i] or cpoke_starts[i] is None:
            continue
        otype = opto_type[i] if i < len(opto_type) else "Full Trial"
        win_start, win_stop = opto_windows.get(otype, (0.0, 1.3))
        t_on = cpoke_starts[i] + win_start
        t_off = cpoke_starts[i] + win_stop
        raw_lp = float(opto_left_power[i]) if i < len(opto_left_power) else 0.0
        raw_rp = float(opto_right_power[i]) if i < len(opto_right_power) else 0.0
        lp = power_W if raw_lp >= threshold else 0.0
        rp = power_W if raw_rp >= threshold else 0.0
        left_timestamps += [t_on, t_off]
        left_data += [lp, 0.0]
        right_timestamps += [t_on, t_off]
        right_data += [rp, 0.0]

    if not left_timestamps:
        return

    # ── ndx-optogenetics: rich structured metadata ────────────────────────────
    em = opto_meta["excitation_source_model"]
    cerebro_model = ExcitationSourceModel(
        name=em["name"],
        source_type=em["source_type"],
        excitation_mode=em["excitation_mode"],
        manufacturer=em["manufacturer"],
        wavelength_range_in_nm=em["wavelength_range_in_nm"],
        description=em["description"],
    )
    es = opto_meta["excitation_source"]
    cerebro = ExcitationSource(
        name=es["name"],
        manufacturer=es["manufacturer"],
        power_in_W=es["power_in_W"],
        description=es["description"],
    )
    nwbfile.add_device_model(cerebro_model)
    nwbfile.add_device(cerebro)

    # Plain pynwb Device for the laser — required by OptogeneticStimulusSite.device
    ld = opto_meta["laser_device"]
    cerebro_laser_device = nwbfile.create_device(
        name=ld["name"],
        description=ld["description"],
        manufacturer=ld["manufacturer"],
    )

    fm = opto_meta["optical_fiber_model"]
    fiber_model = OpticalFiberModel(
        name=fm["name"],
        numerical_aperture=fm["numerical_aperture"],
        core_diameter_in_um=fm["core_diameter_in_um"],
        manufacturer=fm["manufacturer"],
        description=fm["description"],
    )
    nwbfile.add_device_model(fiber_model)

    # OpticalFiber requires a FiberInsertion group (ndx-ophys-devices 0.3.x)
    fl = opto_meta["optical_fibers"]["left"]
    fiber_left = OpticalFiber(
        name=fl["name"],
        description=fl["description"],
        fiber_insertion=FiberInsertion(
            name="fiber_insertion",
            insertion_position_ap_in_mm=fl["insertion_position_ap_in_mm"],
            insertion_position_ml_in_mm=fl["insertion_position_ml_in_mm"],
        ),
    )
    fr = opto_meta["optical_fibers"]["right"]
    fiber_right = OpticalFiber(
        name=fr["name"],
        description=fr["description"],
        fiber_insertion=FiberInsertion(
            name="fiber_insertion",
            insertion_position_ap_in_mm=fr["insertion_position_ap_in_mm"],
            insertion_position_ml_in_mm=fr["insertion_position_ml_in_mm"],
        ),
    )
    nwbfile.add_device(fiber_left)
    nwbfile.add_device(fiber_right)

    eff = opto_meta["effector"]
    chr2_effector = Effector(
        name=eff["name"],
        label=eff["label"],
        description=eff["description"],
    )

    st = opto_meta["sites_table"]
    sites_table = OptogeneticSitesTable(
        name=st["name"],
        description=st["description"],
    )
    sites_table.add_row(optical_fiber=fiber_left, excitation_source=cerebro, effector=chr2_effector)
    sites_table.add_row(optical_fiber=fiber_right, excitation_source=cerebro, effector=chr2_effector)

    vv = opto_meta["viral_vector"]
    virus = ViralVector(
        name=vv["name"],
        construct_name=vv["construct_name"],
        manufacturer=vv["manufacturer"],
        titer_in_vg_per_ml=float("nan"),  # not reported in paper
        description=vv["description"],
    )

    il = opto_meta["injections"]["left"]
    inj_left = ViralVectorInjection(
        name=il["name"],
        location=il["location"],
        hemisphere=il["hemisphere"],
        reference=il["reference"],
        ap_in_mm=il["ap_in_mm"],
        ml_in_mm=il["ml_in_mm"],
        dv_in_mm=float("nan"),  # not reported as single value; injected over 1.5 mm tract
        volume_in_uL=il["volume_in_uL"],
        viral_vector=virus,
        description=il["description"],
    )
    ir = opto_meta["injections"]["right"]
    inj_right = ViralVectorInjection(
        name=ir["name"],
        location=ir["location"],
        hemisphere=ir["hemisphere"],
        reference=ir["reference"],
        ap_in_mm=ir["ap_in_mm"],
        ml_in_mm=ir["ml_in_mm"],
        dv_in_mm=float("nan"),  # not reported as single value; injected over 1.5 mm tract
        volume_in_uL=ir["volume_in_uL"],
        viral_vector=virus,
        description=ir["description"],
    )

    stim = opto_meta["stimulation"]
    exp_metadata = OptogeneticExperimentMetadata(
        stimulation_software=stim["software"],
        optogenetic_sites_table=sites_table,
        optogenetic_effectors=OptogeneticEffectors(effectors=[chr2_effector]),
        optogenetic_viruses=OptogeneticViruses(viral_vectors=[virus]),
        optogenetic_virus_injections=OptogeneticVirusInjections(viral_vector_injections=[inj_left, inj_right]),
    )
    nwbfile.add_lab_meta_data(exp_metadata)

    # ── OptogeneticEpochsTable ────────────────────────────────────────────────
    et = opto_meta["epochs_table"]
    epochs_table = OptogeneticEpochsTable(
        name=et["name"],
        description=et["description"],
        target_tables={"optogenetic_sites": sites_table},
    )
    for i in range(n_trials):
        if not opto_connected[i] or cpoke_starts[i] is None:
            continue
        otype = opto_type[i] if i < len(opto_type) else "Full Trial"
        win_start, win_stop = opto_windows.get(otype, (0.0, 1.3))
        duration_ms = (win_stop - win_start) * 1000.0
        epochs_table.add_row(
            start_time=cpoke_starts[i] + win_start,
            stop_time=cpoke_starts[i] + win_stop,
            stimulation_on=True,
            pulse_length_in_ms=duration_ms,
            period_in_ms=duration_ms,
            number_pulses_per_pulse_train=stim["number_pulses_per_pulse_train"],
            number_trains=stim["number_trains"],
            intertrain_interval_in_ms=float("nan"),
            power_in_mW=power_mW,
            wavelength_in_nm=stim["wavelength_in_nm"],
            optogenetic_sites=[0, 1],  # both left and right hemisphere sites
        )
    nwbfile.add_time_intervals(epochs_table)

    # ── PyNWB OptogeneticStimulusSite + OptogeneticSeries ────────────────────
    ssl = opto_meta["stimulus_sites"]["left"]
    left_site = OptogeneticStimulusSite(
        name=ssl["name"],
        device=cerebro_laser_device,
        description=ssl["description"],
        excitation_lambda=ssl["excitation_lambda"],
        location=ssl["location"],
    )
    ssr = opto_meta["stimulus_sites"]["right"]
    right_site = OptogeneticStimulusSite(
        name=ssr["name"],
        device=cerebro_laser_device,
        description=ssr["description"],
        excitation_lambda=ssr["excitation_lambda"],
        location=ssr["location"],
    )
    nwbfile.add_ogen_site(left_site)
    nwbfile.add_ogen_site(right_site)

    left_order = np.argsort(left_timestamps)
    right_order = np.argsort(right_timestamps)

    series_meta = opto_meta["series"]
    nwbfile.add_stimulus(
        OptogeneticSeries(
            name="optogenetic_series_left",
            data=np.array(left_data)[left_order],
            timestamps=np.array(left_timestamps)[left_order],
            site=left_site,
            description=series_meta["description_template"].format(side="left"),
            comments=series_meta["comments"],
        )
    )
    nwbfile.add_stimulus(
        OptogeneticSeries(
            name="optogenetic_series_right",
            data=np.array(right_data)[right_order],
            timestamps=np.array(right_timestamps)[right_order],
            site=right_site,
            description=series_meta["description_template"].format(side="right"),
            comments=series_meta["comments"],
        )
    )
