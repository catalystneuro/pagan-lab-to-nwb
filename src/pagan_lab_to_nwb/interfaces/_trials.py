"""Functions for building TrialsTable and TaskArgumentsTable."""

import json
from warnings import warn

import numpy as np
from ndx_structured_behavior import TrialsTable
from pynwb.file import NWBFile


def _to_json_serializable(obj):
    """Convert numpy scalars/arrays to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def add_trials_to_nwbfile(
    nwbfile: NWBFile,
    parsed_events: list[dict],
    starting_state: str,
    metadata: dict,
) -> None:
    """Build TrialsTable and attach it to the NWB file.

    Requires ``task_recording`` to already be present in ``nwbfile.acquisition``
    (call :func:`add_task_recording_to_nwbfile` first).
    """
    task_recording = nwbfile.acquisition["task_recording"]
    states_table = task_recording.states
    events_table = task_recording.events
    actions_table = task_recording.actions

    trials_table = TrialsTable(
        description=metadata["Behavior"]["TrialsTable"]["description"],
        states_table=states_table,
        events_table=events_table,
        actions_table=actions_table,
    )

    trial_start_times = [events["states"][starting_state][0][1] for events in parsed_events]
    trial_stop_times = [events["states"][starting_state][1][0] for events in parsed_events]

    for start, stop in zip(trial_start_times, trial_stop_times):
        states = states_table[:]
        states_idx = states[(states["start_time"] >= start) & (states["stop_time"] <= stop)].index

        events = events_table[:]
        events_idx = events[(events["timestamp"] >= start) & (events["timestamp"] <= stop)].index

        actions = actions_table[:]
        actions_idx = actions[(actions["timestamp"] >= start) & (actions["timestamp"] <= stop)].index

        trials_table.add_trial(
            start_time=start,
            stop_time=stop,
            states=states_idx.tolist(),
            events=events_idx.tolist(),
            actions=actions_idx.tolist(),
        )

    nwbfile.trials = trials_table


def add_task_arguments(
    nwbfile: NWBFile,
    saved: dict,
    arguments_to_exclude: list[str] | None = None,
    arguments_metadata: dict | None = None,
    stub_test: bool = False,
) -> None:
    """Add session-level task parameters from ``saved`` to the NWB file.

    Scalar parameters go into ``TaskArgumentsTable``; array parameters whose
    length matches the trial count are added as trial columns.
    """
    if arguments_to_exclude is None:
        arguments_to_exclude = [
            "raw_events",
            "prepare_next_trial_set",
            "parsed_events",
            "current_assembler",
            "my_gui",
            "my_xyfig",
            "ThisStimulus",
            "PokesPlotSection",
            "pokesplot",
            # SessionDefinition GUI-state fields (non-scientific):
            "SessionDefinition_my_gui_info",
            "SessionDefinition_last_change_tracker",
            "SessionDefinition_last_callback_placeholder",
            "SessionDefinition_session_show",
            "SessionDefinition_display_type",
            "SessionDefinition_show_train",
            "SessionDefinition_show_complete",
            "SessionDefinition_show_eod",
            "SessionDefinition_show_name",
            "SessionDefinition_show_vars",
            "SessionDefinition_btnbk_hdr",
            "SessionDefinition_listblock",
            "SessionDefinition_paramblock",
            "SessionDefinition_var_hdr",
            "SessionDefinition_var_names",
            "SessionDefinition_list_hdr",
            "SessionDefinition_subparam_hdr",
            "SessionDefinition_last_change",
            "SessionDefinition_old_style_parsing_flag",
            # SoundInterface: feedback-sound configuration; not scientifically relevant.
            "SoundInterface_",
        ]

    # Lazy import to avoid circular dependency:
    # interfaces._trials → arc_behavior.utils → arc_behavior.__init__ → nwbconverter → interfaces
    from pagan_lab_to_nwb.arc_behavior.utils import (
        get_description_from_arguments_metadata,
    )

    if (trials := nwbfile.trials) is None:
        warn("No trials found in the NWB file. Skipping task arguments addition.")
        return

    num_completed_trials = saved.get("ProtocolsSection_n_completed_trials", len(trials))
    num_started_trials = saved.get("ProtocolsSection_n_started_trials", len(trials))
    num_incomplete_trials = num_started_trials - num_completed_trials

    if "task" not in nwbfile.lab_meta_data:
        raise ValueError("Task metadata not found in NWB file.")
    task_arguments = nwbfile.get_lab_meta_data("task").task_arguments

    task_arguments_to_add = [col for col in saved.keys() if not any(skip in col for skip in arguments_to_exclude)]

    array_type_arguments = {}
    for argument_name in task_arguments_to_add:
        argument_description = get_description_from_arguments_metadata(arguments_metadata, argument_name)
        argument_value = saved[argument_name]

        if isinstance(argument_value, (np.ndarray, list)):
            if len(argument_value):
                array_type_arguments[argument_name] = argument_value
            continue

        if isinstance(argument_value, dict):
            try:
                task_arguments.add_row(
                    argument_name=argument_name,
                    argument_description=argument_description,
                    expression=json.dumps(argument_value, default=_to_json_serializable),
                    expression_type="json",
                    output_type="text",
                )
            except (TypeError, ValueError) as e:
                warn(f"Could not JSON-serialize dict argument '{argument_name}': {e}. Skipping.")
            continue

        if isinstance(argument_value, int):
            expression_type, output_type = "integer", "numeric"
        elif isinstance(argument_value, float):
            expression_type, output_type = "float", "numeric"
        elif isinstance(argument_value, str):
            expression_type, output_type = "string", "text"
            if len(argument_value) == num_completed_trials:
                array_type_arguments[argument_name] = list(argument_value)
                continue
            if len(argument_value) == num_started_trials and num_incomplete_trials == 1:
                array_type_arguments[argument_name] = list(argument_value[:num_completed_trials])
                continue
        else:
            raise Exception(f"Unknown type '{type(argument_value)}' for argument '{argument_name}'.")

        task_arguments.add_row(
            argument_name=argument_name,
            argument_description=argument_description,
            expression=str(argument_value),
            expression_type=expression_type,
            output_type=output_type,
        )

    # Add array-type arguments either as trial columns or as task arguments
    for argument_name, argument_values in array_type_arguments.items():
        argument_description = get_description_from_arguments_metadata(arguments_metadata, argument_name)

        if isinstance(argument_values, list):
            try:
                argument_values = np.array(argument_values)
            except ValueError:
                argument_values = np.array(argument_values, dtype=object)

        # Object-dtype arrays: JSON-dump as a task argument
        if argument_values.dtype == object:
            try:
                task_arguments.add_row(
                    argument_name=argument_name,
                    argument_description=argument_description,
                    expression=json.dumps(argument_values.tolist(), default=_to_json_serializable),
                    expression_type="json",
                    output_type="text",
                )
            except (TypeError, ValueError) as e:
                warn(f"Could not JSON-serialize argument '{argument_name}': {e}. Skipping.")
            continue

        if len(argument_values) and isinstance(argument_values[0], str):
            argument_values = argument_values[:100] if stub_test else argument_values
            if len(argument_values) == num_completed_trials:
                trials.add_column(name=argument_name, description=argument_description, data=argument_values)
        elif np.isnan(argument_values).all():
            continue
        else:
            argument_values = argument_values[:100] if stub_test else argument_values
            if len(argument_values) == num_completed_trials + 2:
                warn(f"Argument '{argument_name}' has 2 extra entries, removing them.")
                trials.add_column(
                    name=argument_name,
                    description=argument_description,
                    data=argument_values[:num_completed_trials],
                )
            elif len(argument_values) == num_started_trials and num_incomplete_trials == 1:
                trials.add_column(
                    name=argument_name,
                    description=argument_description,
                    data=argument_values[:num_completed_trials],
                )
            elif len(argument_values) == num_completed_trials - 1:
                trials.add_column(
                    name=argument_name,
                    description=argument_description,
                    data=np.append(argument_values, np.nan),
                )
            elif len(argument_values) == num_completed_trials:
                trials.add_column(name=argument_name, description=argument_description, data=argument_values)
            elif len(argument_values) < 127:  # spyglass column-count limit
                task_arguments.add_row(
                    argument_name=argument_name,
                    argument_description=argument_description,
                    expression=str(argument_values),
                    expression_type="array",
                    output_type="numeric",
                )
            else:
                warn(
                    f"Argument '{argument_name}' has {len(argument_values)} entries, "
                    f"expected {num_completed_trials}. Skipping."
                )
