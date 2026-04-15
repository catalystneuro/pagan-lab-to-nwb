"""Functions for building ndx-structured-behavior task recording tables."""

import numpy as np
import pandas as pd
from ndx_structured_behavior import (
    ActionsTable,
    ActionTypesTable,
    EventsTable,
    EventTypesTable,
    StatesTable,
    StateTypesTable,
    Task,
    TaskArgumentsTable,
    TaskRecording,
)
from pynwb.file import NWBFile


def create_states(
    parsed_events: list[dict],
    starting_state: str,
    metadata: dict,
) -> tuple[StateTypesTable, StatesTable]:
    """Create StateTypesTable and StatesTable from parsed BControl events."""
    state_types = StateTypesTable(description=metadata["Behavior"]["StateTypesTable"]["description"])
    states_table = StatesTable(
        description=metadata["Behavior"]["StatesTable"]["description"],
        state_types_table=state_types,
    )

    first_trial_states = parsed_events[0]["states"]
    for state_name in first_trial_states:
        if not isinstance(first_trial_states[state_name], np.ndarray):
            continue
        state_types.add_row(state_name=state_name, check_ragged=False)

    state_rows = []
    for trial_events in parsed_events:
        states = trial_events["states"]
        state_names = state_types.state_name[:]
        for state_name in state_names:
            state_times = states[state_name] if state_name in states else []
            if len(state_times) == 0:
                continue

            state_times = np.asarray(state_times)
            # Special handling for starting state with possible NaNs
            if state_name == starting_state:
                not_nan = ~np.isnan(state_times)
                starting_state_times = state_times[not_nan]
                if len(starting_state_times) > 2:
                    raise ValueError(
                        f"Unexpected shape for starting state '{state_name}', shape ({state_times.shape}) "
                        f"values of {state_times[:]}. Expected shape is (2,) or (2, 2) with NaNs handled."
                    )
                start_time, stop_time = starting_state_times
                state_rows.append({"state_name": state_name, "start_time": start_time, "stop_time": stop_time})
                continue

            # Single interval: shape (2,)
            if state_times.shape == (2,):
                start_time, stop_time = state_times
                if not np.isnan(start_time):
                    state_rows.append({"state_name": state_name, "start_time": start_time, "stop_time": stop_time})
                continue

            # Special case: shape (2, 2) and first row contains NaN
            if state_times.shape == (2, 2) and np.any(np.isnan(state_times[0])):
                not_nan = ~np.isnan(state_times)
                flat_times = state_times[not_nan]
                if len(flat_times) >= 2:
                    start_time, stop_time = flat_times[:2]
                    state_rows.append({"state_name": state_name, "start_time": start_time, "stop_time": stop_time})
                continue

            # General case: iterate over intervals
            for state_time in state_times:
                if len(state_time) != 2:
                    raise ValueError(f"Unexpected shape for state '{state_name}': {state_time.shape}.")
                start_time, stop_time = state_time
                if not np.isnan(start_time):
                    state_rows.append({"state_name": state_name, "start_time": start_time, "stop_time": stop_time})

    if state_rows:
        df = pd.DataFrame(state_rows).sort_values(by="start_time")
        for _, row in df.iterrows():
            state_type = state_types.state_name[:].index(row["state_name"])
            states_table.add_row(
                state_type=state_type,
                start_time=row["start_time"],
                stop_time=row["stop_time"],
                check_ragged=False,
            )

    return state_types, states_table


def create_events(
    parsed_events: list[dict],
    metadata: dict,
) -> tuple[EventTypesTable, EventsTable]:
    """Create EventTypesTable and EventsTable from parsed BControl events."""
    event_types = EventTypesTable(description=metadata["Behavior"]["EventTypesTable"]["description"])
    events_table = EventsTable(
        description=metadata["Behavior"]["EventsTable"]["description"],
        event_types_table=event_types,
    )

    first_trial_events = parsed_events[0]["pokes"]
    for event_name in first_trial_events:
        if not isinstance(first_trial_events[event_name], np.ndarray):
            continue
        event_types.add_row(event_name=event_name, check_ragged=False)

    event_rows = []
    for trial_events in parsed_events:
        pokes = trial_events["pokes"]
        event_names = event_types.event_name[:]
        for event_name in event_names:
            event_times = pokes[event_name]
            if len(event_times) == 0:
                continue
            value = pokes["starting_state"].get(event_name, "out")
            if event_times.shape == (2,):
                if not np.isnan(event_times[0]):
                    event_rows.append(
                        {
                            "event_name": event_name,
                            "timestamp": event_times[0],
                            "duration": event_times[1] - event_times[0],
                            "value": value,
                        }
                    )
            else:
                for event_time in event_times:
                    if not np.isnan(event_time[0]):
                        event_rows.append(
                            {
                                "event_name": event_name,
                                "timestamp": event_time[0],
                                "duration": event_time[1] - event_time[0],
                                "value": value,
                            }
                        )

    if event_rows:
        df = pd.DataFrame(event_rows).sort_values(by="timestamp")
        for _, row in df.iterrows():
            event_type = event_types.event_name[:].index(row["event_name"])
            events_table.add_row(
                event_type=event_type,
                timestamp=row["timestamp"],
                duration=row["duration"],
                value=row["value"],
                check_ragged=False,
            )

    return event_types, events_table


def create_actions(
    parsed_events: list[dict],
    metadata: dict,
) -> tuple[ActionTypesTable, ActionsTable]:
    """Create ActionTypesTable and ActionsTable from parsed BControl events."""
    action_types = ActionTypesTable(description=metadata["Behavior"]["ActionTypesTable"]["description"])
    actions_table = ActionsTable(
        description=metadata["Behavior"]["ActionsTable"]["description"],
        action_types_table=action_types,
    )

    first_trial_actions = parsed_events[0]["waves"]
    for action_name in first_trial_actions:
        if not isinstance(first_trial_actions[action_name], np.ndarray):
            continue
        action_types.add_row(action_name=action_name, check_ragged=False)

    action_rows = []
    for trial_events in parsed_events:
        waves = trial_events["waves"]
        action_names = action_types.action_name[:]
        for action_name in action_names:
            action_times = waves[action_name]
            if len(action_times) == 0:
                continue
            value = waves["starting_state"].get(action_name, "out")
            if action_times.shape == (2,):
                if not np.isnan(action_times[0]):
                    action_rows.append(
                        {
                            "action_name": action_name,
                            "timestamp": action_times[0],
                            "duration": action_times[1] - action_times[0],
                            "value": value,
                        }
                    )
            else:
                for action_time in action_times:
                    if not np.isnan(action_time[0]):
                        action_rows.append(
                            {
                                "action_name": action_name,
                                "timestamp": action_time[0],
                                "duration": action_time[1] - action_time[0],
                                "value": value,
                            }
                        )

    if action_rows:
        df = pd.DataFrame(action_rows).sort_values(by="timestamp")
        for _, row in df.iterrows():
            action_type = action_types.action_name[:].index(row["action_name"])
            actions_table.add_row(
                action_type=action_type,
                timestamp=row["timestamp"],
                duration=row["duration"],
                value=row["value"],
                check_ragged=False,
            )

    return action_types, actions_table


def add_task_recording_to_nwbfile(
    nwbfile: NWBFile,
    parsed_events: list[dict],
    starting_state: str,
    metadata: dict,
) -> None:
    """Build Task + TaskRecording and attach them to the NWB file."""
    state_types_table, states_table = create_states(parsed_events, starting_state, metadata)
    action_types_table, actions_table = create_actions(parsed_events, metadata)
    event_types_table, events_table = create_events(parsed_events, metadata)

    task_arguments_table = TaskArgumentsTable(description=metadata["Behavior"]["TaskArgumentsTable"]["description"])
    task = Task(
        event_types=event_types_table,
        state_types=state_types_table,
        action_types=action_types_table,
        task_arguments=task_arguments_table,
    )
    nwbfile.add_lab_meta_data(task)

    recording = TaskRecording(events=events_table, states=states_table, actions=actions_table)
    nwbfile.add_acquisition(recording)
