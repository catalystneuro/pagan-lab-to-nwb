"""Primary class for converting experiment-specific behavior."""

import re
from datetime import datetime
from pathlib import Path
from warnings import warn

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
    TrialsTable,
)
from pydantic import validate_call
from pynwb.device import Device
from pynwb.file import NWBFile

from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.tools import get_module
from neuroconv.utils import DeepDict, get_base_schema, get_schema_from_hdmf_class
from pagan_lab_to_nwb.arc_behavior.utils import get_description_from_arguments_metadata


# TODO: implement this interface in NeuroConv
class BControlBehaviorInterface(BaseDataInterface):
    """Interface for converting BControl behavioral data files to NWB format."""

    display_name = "BControl Behavior"
    keywords = ("behavior", "states", "events", "actions", "trials", "task recording")
    associated_suffixes = (".mat",)
    info = "Interface for behavior data from BControl to an NWB file."

    @validate_call
    def __init__(self, file_path: Path, starting_state: str = "state_0", verbose: bool = False):
        """
        Data interface for writing BControl behavioral data to an NWB file.

        Writes behavior data using the ndx-structured-behavior extension.

        Parameters
        ----------
        file_path : Path or str
            The path to the BControl data file to be converted.
        starting_state : str, default: "state_0"
            The name of the starting state for the trials. This is used to identify the start time of each trial.
        verbose : bool, default: False
        """

        super().__init__(file_path=file_path)
        self.verbose = verbose
        self.starting_state = starting_state

    def _read_file(self):
        """Read the BControl .mat file and extract the 'saved' and 'saved_history' data."""
        from pymatreader import read_mat

        if hasattr(self, "saved") and hasattr(self, "saved_history"):
            return

        # Read the .mat file
        mat_data = read_mat(self.source_data["file_path"])

        # Extract relevant data from the mat file
        if "saved" not in mat_data and "saved_history" not in mat_data:
            raise ValueError(
                f"The provided .mat file does not contain the expected 'saved' or 'saved_history' fields. The keys: {list(mat_data.keys())}."
            )
        self.saved = mat_data["saved"]
        self.saved_history = mat_data["saved_history"]

    def _get_parsed_events(self, stub_test: bool = False) -> list[dict]:
        """
        Get parsed events from the 'ProtocolsSection_parsed_events' key from 'saved_history'.

        Parameters
        ----------
        stub_test : bool, default: False
            If True, only a subset of trials will be processed for testing purposes.

        Returns
        -------
        list[dict]
            A list of parsed events, where each event is a dictionary containing state, poke, and wave information.
        """
        self._read_file()
        if "ProtocolsSection_parsed_events" not in self.saved_history:
            raise ValueError(
                "The saved_history does not contain 'ProtocolsSection_parsed_events'. "
                "Please ensure the BControl data file is correctly formatted."
            )
        parsed_events = self.saved_history["ProtocolsSection_parsed_events"]
        if not isinstance(parsed_events, list):
            raise ValueError(
                f"Expected 'ProtocolsSection_parsed_events' to be a list, but got {type(parsed_events)}. "
                "Please check the format of the BControl data file."
            )
        num_trials = len(parsed_events)
        if stub_test:
            num_trials = min(num_trials, 100)
            parsed_events = parsed_events[:num_trials]
        return parsed_events

    def get_trial_times(self, stub_test: bool = False) -> (list[float], list[float]):
        """
        Get the start and end times of trials from the parsed events.
        This method extracts the start and end times of trials based on the starting state.

        Parameters
        ----------
        stub_test : bool, default: False
            If True, only a subset of trials will be processed for testing purposes.

        Returns
        -------
        tuple[list[float], list[float]]
            A tuple containing two lists:
            - The start times of the trials.
            - The end times of the trials.
        """
        parsed_events = self._get_parsed_events(stub_test=stub_test)

        trial_start_times = [events["states"][self.starting_state][0][1] for events in parsed_events]
        trial_end_times = [events["states"][self.starting_state][1][0] for events in parsed_events]

        return trial_start_times, trial_end_times

    def get_metadata_schema(self) -> dict:
        metadata_schema = super().get_metadata_schema()
        metadata_schema["properties"]["Behavior"] = get_base_schema(tag="Behavior")
        device_schema = get_schema_from_hdmf_class(Device)
        metadata_schema["properties"]["Behavior"].update(
            required=[
                "Device",
                "StateTypesTable",
                "StatesTable",
                "ActionTypesTable",
                "ActionsTable",
                "EventTypesTable",
                "EventsTable",
                "TrialsTable",
                "TaskArgumentsTable",
            ],
            properties=dict(
                Device=device_schema,
                StateTypesTable=dict(type="object", properties=dict(description={"type": "string"})),
                StatesTable=dict(type="object", properties=dict(description={"type": "string"})),
                ActionTypesTable=dict(type="object", properties=dict(description={"type": "string"})),
                ActionsTable=dict(type="object", properties=dict(description={"type": "string"})),
                EventTypesTable=dict(type="object", properties=dict(description={"type": "string"})),
                EventsTable=dict(type="object", properties=dict(description={"type": "string"})),
                TrialsTable=dict(type="object", properties=dict(description={"type": "string"})),
                TaskArgumentsTable=dict(type="object", properties=dict(description={"type": "string"})),
            ),
        )
        return metadata_schema

    def get_metadata(self) -> DeepDict:
        metadata = super().get_metadata()

        default_device_metadata = dict(
            name="BControl",
            manufacturer="Example Manufacturer",  # TODO: ask from lab
        )
        metadata["Behavior"] = dict(
            Device=default_device_metadata,
            StateTypesTable=dict(description="Contains the name of the states in the task."),
            StatesTable=dict(description="Contains the start and end times of each state in the task."),
            ActionsTable=dict(description="Contains the onset times of the task output actions."),
            ActionTypesTable=dict(description="Contains the name of the task output actions."),
            EventTypesTable=dict(description="Contains the name of the events in the task."),
            EventsTable=dict(description="Contains the onset times of events in the task."),
            TrialsTable=dict(description="Contains the start and end times of each trial in the task."),
            TaskArgumentsTable=dict(description="Contains the task arguments for the task."),
        )

        self._read_file()
        # extract session_start_time from the protocol title
        protocol_title = [key for key in self.saved.keys() if "prot_title" in key]
        if len(protocol_title) == 1:
            protocol_title = self.saved[protocol_title[0]]
            # Extract session_start_time
            # 'TaskSwitch6 - on rig brodyrigws32.princeton.edu : Marino, P131.  Started at 11:41, Ended at 13:19'
            # extract the datetime from "Started at" from the title and date from "SavingSection_SaveTime"
            match = re.search(r"Started at (\d{2}:\d{2})", protocol_title)
            # lookup file save date and combine with the time from the protocol title
            if "SavingSection_SaveTime" in self.saved and match:
                save_time = self.saved["SavingSection_SaveTime"]  # '15-Aug-2019 13:19:41'
                time_str = match.group(1)
                # Extract date part (e.g., '15-Aug-2019') from save_time
                date_str = save_time.split()[0]
                # Combine date and time
                try:
                    session_start_time = datetime.strptime(f"{date_str} {time_str}", "%d-%b-%Y %H:%M")
                    metadata["NWBFile"]["session_start_time"] = session_start_time
                except ValueError as e:
                    warn(
                        f"Failed to parse session start time from protocol title '{protocol_title}' and save time '{save_time}': {e}"
                    )

        return metadata

    def create_states(self, metadata: dict, stub_test: bool = False) -> tuple[StateTypesTable, StatesTable]:
        """
        Create states and state types tables from the parsed events.
         This method extracts state information from the parsed events and creates
         the corresponding StateTypesTable and StatesTable.

        Parameters
         ----------
         metadata : dict
             Metadata dictionary containing information about the behavior data.
         stub_test : bool, default: False
             If True, only a subset of trials will be processed for testing purposes.

        Returns
        -------
        tuple[StateTypesTable, StatesTable]
            A tuple containing the StateTypesTable and StatesTable.
        """
        state_types = StateTypesTable(description=metadata["Behavior"]["StateTypesTable"]["description"])
        states_table = StatesTable(
            description=metadata["Behavior"]["StatesTable"]["description"],
            state_types_table=state_types,
        )

        parsed_events = self._get_parsed_events(stub_test=stub_test)
        first_trial_states = parsed_events[0]["states"]
        for state_name in first_trial_states:
            # Check if the state is a valid state with recorded times
            if not isinstance(first_trial_states[state_name], np.ndarray):
                continue
            state_types.add_row(state_name=state_name, check_ragged=False)

        state_rows = []
        for trial_events in parsed_events:
            states = trial_events["states"]
            state_names = state_types.state_name[:]
            for state_name in state_names:
                state_times = states[state_name]
                if len(state_times) == 0:
                    continue

                state_times = np.asarray(state_times)
                # Special handling for starting state with possible NaNs
                if state_name == self.starting_state:
                    not_nan = ~np.isnan(state_times)
                    starting_state_times = state_times[not_nan]
                    if len(starting_state_times) > 2:
                        raise ValueError(
                            f"Unexpected shape for starting state '{state_name}': {state_times.shape}. "
                            f"Expected shape is (2,) or (2, 2) with NaNs handled."
                        )
                    start_time, stop_time = starting_state_times
                    state_rows.append(
                        {
                            "state_name": state_name,
                            "start_time": start_time,
                            "stop_time": stop_time,
                        }
                    )
                    continue

                # Single interval: shape (2,)
                if state_times.shape == (2,):
                    start_time, stop_time = state_times
                    if not np.isnan(start_time):
                        state_rows.append(
                            {
                                "state_name": state_name,
                                "start_time": start_time,
                                "stop_time": stop_time,
                            }
                        )
                    continue

                # Special case: shape (2, 2) and first row contains NaN
                if state_times.shape == (2, 2) and np.any(np.isnan(state_times[0])):
                    not_nan = ~np.isnan(state_times)
                    flat_times = state_times[not_nan]
                    if len(flat_times) >= 2:
                        start_time, stop_time = flat_times[:2]
                        state_rows.append(
                            {
                                "state_name": state_name,
                                "start_time": start_time,
                                "stop_time": stop_time,
                            }
                        )
                    continue

                # General case: iterate over intervals
                for state_time in state_times:
                    if len(state_time) != 2:
                        raise ValueError(f"Unexpected shape for state '{state_name}': {state_time.shape}. ")
                    start_time, stop_time = state_time
                    if not np.isnan(start_time):
                        state_rows.append(
                            {
                                "state_name": state_name,
                                "start_time": start_time,
                                "stop_time": stop_time,
                            }
                        )

        # Sort by start_time
        if state_rows:
            states = pd.DataFrame(state_rows)
            states = states.sort_values(by="start_time")
            for _, row in states.iterrows():
                state_type = state_types.state_name[:].index(row["state_name"])
                states_table.add_row(
                    state_type=state_type,
                    start_time=row["start_time"],
                    stop_time=row["stop_time"],
                    check_ragged=False,
                )

        return state_types, states_table

    def create_events(self, metadata: dict, stub_test: bool = False) -> tuple[EventTypesTable, EventsTable]:
        """
        Create events and event types tables from the parsed events.

        This method extracts event information from the parsed events and creates
        the corresponding EventTypesTable and EventsTable.

        Parameters
        ----------
        metadata : dict
            Metadata dictionary containing information about the behavior data.
        stub_test : bool, default: False
            If True, only a subset of trials will be processed for testing purposes.

        Returns
        -------
        tuple[EventTypesTable, EventsTable]
            A tuple containing the EventTypesTable and EventsTable.
        """
        event_types = EventTypesTable(description=metadata["Behavior"]["EventTypesTable"]["description"])
        events_table = EventsTable(
            description=metadata["Behavior"]["EventsTable"]["description"], event_types_table=event_types
        )

        parsed_events = self._get_parsed_events(stub_test=stub_test)

        # Add event types
        first_trial_events = parsed_events[0]["pokes"]
        for event_name in first_trial_events:
            if not isinstance(first_trial_events[event_name], np.ndarray):
                continue
            event_types.add_row(
                event_name=event_name,
                check_ragged=False,
            )

        # Collect all event rows
        event_rows = []
        for trial_events in parsed_events:
            pokes = trial_events["pokes"]
            event_names = event_types.event_name[:]
            for event_name in event_names:
                event_times = pokes[event_name]
                if len(event_times) == 0:
                    continue

                value = pokes["starting_state"].get(event_name, "out")
                # Single interval: shape (2,)
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
                # General case: iterate over intervals
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

        # Sort by timestamp
        if event_rows:
            events = pd.DataFrame(event_rows)
            events = events.sort_values(by="timestamp")
            for _, row in events.iterrows():
                event_type = event_types.event_name[:].index(row["event_name"])
                events_table.add_row(
                    event_type=event_type,
                    timestamp=row["timestamp"],
                    duration=row["duration"],
                    value=row["value"],
                    check_ragged=False,
                )
        return event_types, events_table

    def create_actions(self, metadata: dict, stub_test: bool = False) -> tuple[ActionTypesTable, ActionsTable]:
        """
        Create actions and action types tables from the parsed events.

        This method extracts action information from the parsed events and creates
        the corresponding ActionTypesTable and ActionsTable.

        Parameters
        ----------
        metadata : dict
            Metadata dictionary containing information about the behavior data.
        stub_test : bool, default: False
            If True, only a subset of trials will be processed for testing purposes.

        Returns
        -------
        tuple[ActionTypesTable, ActionsTable]
            A tuple containing the ActionTypesTable and ActionsTable.
        """
        action_types = ActionTypesTable(description=metadata["Behavior"]["ActionTypesTable"]["description"])
        actions_table = ActionsTable(
            description=metadata["Behavior"]["ActionTypesTable"]["description"], action_types_table=action_types
        )

        parsed_events = self._get_parsed_events(stub_test=stub_test)

        first_trial_actions = parsed_events[0]["waves"]
        for action_name in first_trial_actions:
            if not isinstance(first_trial_actions[action_name], np.ndarray):
                continue
            action_types.add_row(
                action_name=action_name,
                check_ragged=False,
            )

        # Collect all action rows
        action_rows = []
        for trial_events in parsed_events:
            waves = trial_events["waves"]
            action_names = action_types.action_name[:]
            for action_name in action_names:
                action_times = waves[action_name]
                if len(action_times) == 0:
                    continue

                value = waves["starting_state"].get(action_name, "out")
                # Single interval: shape (2,)
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

        # Sort by timestamp
        if action_rows:
            actions = pd.DataFrame(action_rows)
            actions = actions.sort_values(by="timestamp")
            for _, row in actions.iterrows():
                action_type = action_types.action_name[:].index(row["action_name"])
                actions_table.add_row(
                    action_type=action_type,
                    timestamp=row["timestamp"],
                    duration=row["duration"],
                    value=row["value"],
                    check_ragged=False,
                )
        return action_types, actions_table

    def add_task_recording_to_nwbfile(self, nwbfile: NWBFile, metadata: dict, stub_test: bool = False) -> None:
        """
        Add events, states, actions, and task arguments to the NWB file as a TaskRecording.

        Parameters
        ----------
        nwbfile : NWBFile
            The NWB file to which the task recording will be added.
        metadata : dict
            Metadata dictionary containing information about the behavior data.
        stub_test : bool, default: False
            If True, only a subset of trials will be processed for testing purposes.

        """

        # Create the task recording tables
        state_types_table, states_table = self.create_states(stub_test=stub_test, metadata=metadata)
        action_types_table, actions_table = self.create_actions(stub_test=stub_test, metadata=metadata)
        event_types_table, events_table = self.create_events(stub_test=stub_test, metadata=metadata)

        task_arguments_metadata = metadata["Behavior"]["TaskArgumentsTable"]
        task_arguments_table = TaskArgumentsTable(description=task_arguments_metadata["description"])

        task = Task(
            event_types=event_types_table,
            state_types=state_types_table,
            action_types=action_types_table,
            task_arguments=task_arguments_table,
        )
        # Add the task
        nwbfile.add_lab_meta_data(task)

        # Add the tables to the task recording
        recording = TaskRecording(events=events_table, states=states_table, actions=actions_table)
        nwbfile.add_acquisition(recording)

    def add_trials_to_nwbfile(self, nwbfile: NWBFile, metadata: dict, stub_test: bool = False) -> None:
        """Add trials to the NWB file."""

        if "task_recording" not in nwbfile.acquisition:
            self.add_task_recording_to_nwbfile(nwbfile=nwbfile, metadata=metadata, stub_test=stub_test)
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

        trial_start_times, trial_stop_times = self.get_trial_times(stub_test=stub_test)
        for start, stop in zip(trial_start_times, trial_stop_times):
            states = states_table[:]
            states_index_mask = (states["start_time"] >= start) & (states["stop_time"] <= stop)
            states_index_ranges = states[states_index_mask].index

            events = events_table[:]
            events_index_mask = (events["timestamp"] >= start) & (events["timestamp"] <= stop)
            events_index_ranges = events[events_index_mask].index

            actions = actions_table[:]
            actions_index_mask = (actions["timestamp"] >= start) & (actions["timestamp"] <= stop)
            actions_index_ranges = actions[actions_index_mask].index
            trials_table.add_trial(
                start_time=start,
                stop_time=stop,
                states=states_index_ranges.tolist(),
                events=events_index_ranges.tolist(),
                actions=actions_index_ranges.tolist(),
            )

        nwbfile.trials = trials_table

    def add_task_arguments(
        self,
        nwbfile: NWBFile,
        arguments_to_exclude: list[str] = None,
        arguments_metadata: dict = None,
        stub_test: bool = False,
    ) -> None:
        """
        Add task arguments to the NWB file.

        Parameters
        ----------
        nwbfile : NWBFile
            The NWB file to which the task arguments will be added.
        arguments_to_exclude : list[str], optional
            List of argument names to exclude from the task arguments. If None, defaults to a predefined list.
        arguments_metadata : dict, optional
            Metadata for the task arguments.
        stub_test : bool, default: False
            If True, only a subset of trials will be processed for testing purposes.
        """
        if arguments_to_exclude is None:
            arguments_to_exclude = [
                "raw_events",
                "prepare_next_trial_set",
                "parsed_events",
                "current_assembler",
                "comments",
                "my_gui",
                "my_xyfig",
                "ThisStimulus",
            ]

        if (trials := nwbfile.trials) is None:
            warn("No trials found in the NWB file. Skipping task arguments addition.")
            return

        num_trials = len(trials)
        all_arguments = list(self.saved.keys())
        array_type_arguments = dict()

        if "task" not in nwbfile.lab_meta_data:
            raise ValueError("Task metadata not found in NWB file.")
        task = nwbfile.get_lab_meta_data("task")
        task_arguments = task.task_arguments

        task_arguments_to_add = [col for col in all_arguments if not any(skip in col for skip in arguments_to_exclude)]
        for argument_name in task_arguments_to_add:
            # Get description for task arguments
            argument_description = get_description_from_arguments_metadata(arguments_metadata, argument_name)

            argument_value = self.saved[argument_name]
            if isinstance(argument_value, np.ndarray) or isinstance(argument_value, list):
                # Array types are added to trials directly
                if len(argument_value):
                    array_type_arguments[argument_name] = argument_value
                continue
            if isinstance(argument_value, dict):
                continue
            if isinstance(argument_value, int):
                expression_type = "integer"  # The type of the expression
                output_type = "numeric"  # The type of the output
            elif isinstance(argument_value, float):
                expression_type = "float"
                output_type = "numeric"
            elif isinstance(argument_value, str):
                expression_type = "string"
                output_type = "text"
                # check for char array
                if len(argument_value) == num_trials:
                    # add as list of characters
                    array_type_arguments[argument_name] = [a for a in argument_value]
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

        # Add array type arguments to trials
        for argument_name, argument_values in array_type_arguments.items():
            # Get description for array type arguments
            argument_description = get_description_from_arguments_metadata(arguments_metadata, argument_name)

            if isinstance(argument_values, list):
                argument_values = np.array(argument_values)
            if len(argument_values) and isinstance(argument_values[0], str):
                argument_values = argument_values[:100] if stub_test else argument_values
                if len(argument_values) == num_trials:
                    trials.add_column(
                        name=argument_name,
                        description=argument_description,
                        data=argument_values,
                    )
            elif np.isnan(argument_values).all():
                continue
            else:
                argument_values = argument_values[:100] if stub_test else argument_values
                if len(argument_values) == num_trials - 1:
                    # add np.nan for the incomplete trial
                    argument_values = np.append(argument_values, np.nan)
                if len(argument_values) == num_trials:
                    trials.add_column(
                        name=argument_name,
                        description=argument_description,
                        data=argument_values,
                    )

        # TODO: add event times
        event_times_columns = [col for col in array_type_arguments.keys() if "times" in col.lower()]
        print("Need to add event times columns:", event_times_columns)

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: dict,
        arguments_to_exclude: list[str] = None,
        arguments_metadata: dict = None,
        stub_test: bool = False,
    ) -> None:
        self.add_trials_to_nwbfile(nwbfile=nwbfile, metadata=metadata, stub_test=stub_test)
        self.add_task_arguments(
            nwbfile=nwbfile,
            stub_test=stub_test,
            arguments_to_exclude=arguments_to_exclude,
            arguments_metadata=arguments_metadata,
        )
        get_module(
            nwbfile=nwbfile, name="behavior", description="Behavior module"
        )  # Ensure the behavior module exists for spyglass compatibility
