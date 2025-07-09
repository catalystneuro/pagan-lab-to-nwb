"""Primary class for converting experiment-specific behavior."""

import re
from datetime import datetime
from pathlib import Path
from warnings import warn

import numpy as np
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
            The name of the starting state for the trials. This is used to identify the start of each trial.
        verbose : bool, default: False
        """

        super().__init__(file_path=file_path)
        self.verbose = verbose
        self.starting_state = starting_state

    def _read_data(self):
        """Read the BControl data file and return the data."""
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
        """Get parsed events from the saved history."""
        self._read_data()
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

        self._read_data()
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
        """Create states and state types tables from the parsed events.
         This method extracts state information from the parsed events and creates
         the corresponding StateTypesTable and StatesTable.

        Parameters
         ----------
         metadata : dict
             Metadata dictionary containing information about the behavior data.
         stub_test : bool, default: False
             If True, only a subset of trials will be processed for testing purposes.
        """
        state_types = StateTypesTable(description=metadata["Behavior"]["StateTypesTable"]["description"])
        states_table = StatesTable(
            description=metadata["Behavior"]["StatesTable"]["description"],
            state_types_table=state_types,
        )

        parsed_events = self._get_parsed_events(stub_test=stub_test)
        first_parsed_events = parsed_events[0]
        for state_name in first_parsed_events["states"]:
            if not isinstance(first_parsed_events["states"][state_name], np.ndarray):
                continue
            state_types.add_row(
                state_name=state_name,
                check_ragged=False,
            )

        for trial_events in parsed_events:
            states = trial_events["states"]
            state_names = state_types.state_name[:]
            for state_name in state_names:
                if state_name == self.starting_state:
                    state_start_time = trial_events["states"][self.starting_state][0][1]  # start time of the trial
                    state_stop_time = trial_events["states"][self.starting_state][1][0]  # end time of the trial
                    states_table.add_row(
                        state_type=state_names.index(state_name),
                        start_time=state_start_time,
                        stop_time=state_stop_time,
                        check_ragged=False,
                    )
                elif len(states[state_name]) == 0:
                    # print(f"Skipping state {state_name} with no recorded times.")
                    continue
                elif len(states[state_name].shape) == 1:
                    if np.isnan(states[state_name][0]):
                        continue
                    states_table.add_row(
                        state_type=state_names.index(state_name),
                        start_time=states[state_name][0],
                        stop_time=states[state_name][1],
                        check_ragged=False,
                    )
                else:
                    for state_time in states[state_name]:
                        if np.isnan(state_time[0]):
                            continue
                        states_table.add_row(
                            state_type=state_names.index(state_name),
                            start_time=state_time[0],
                            stop_time=state_time[1],
                            check_ragged=False,
                        )

        return state_types, states_table

    def create_events(self, metadata: dict, stub_test: bool = False) -> tuple[EventTypesTable, EventsTable]:
        # todo: add metadata for event types and events tables
        event_types = EventTypesTable(description=metadata["Behavior"]["EventTypesTable"]["description"])
        events_table = EventsTable(
            description=metadata["Behavior"]["EventsTable"]["description"], event_types_table=event_types
        )

        parsed_events = self._get_parsed_events(stub_test=stub_test)

        for event_name in parsed_events[0]["pokes"]:
            if not isinstance(parsed_events[0]["pokes"][event_name], np.ndarray):
                continue
            event_types.add_row(
                event_name=event_name,
                check_ragged=False,
            )

        for trial_events in parsed_events:
            pokes = trial_events["pokes"]
            event_names = event_types.event_name[:]
            for event_name in event_names:
                event_type = event_types.event_name[:].index(event_name)
                if len(pokes[event_name]) == 0:
                    # print(f"Skipping event {event_name} with no recorded times.")
                    continue
                elif len(pokes[event_name].shape) == 1:
                    if np.isnan(pokes[event_name][0]):
                        continue
                    events_table.add_row(
                        event_type=event_type,
                        timestamp=pokes[event_name][0],
                        duration=pokes[event_name][1] - pokes[event_name][0],
                        value="In",  # enter state
                        check_ragged=False,
                    )
                else:
                    for poke_time in pokes[event_name]:
                        if np.isnan(poke_time[0]):
                            continue
                        events_table.add_row(
                            event_type=event_type,
                            timestamp=poke_time[0],
                            duration=poke_time[1] - poke_time[0],
                            value="In",  # value feels redundant here, but it is required by the EventsTable
                            check_ragged=False,
                        )
        return event_types, events_table

    def create_actions(self, metadata: dict, stub_test: bool = False) -> tuple[ActionTypesTable, ActionsTable]:
        action_types = ActionTypesTable(description=metadata["Behavior"]["ActionTypesTable"]["description"])
        actions_table = ActionsTable(
            description=metadata["Behavior"]["ActionTypesTable"]["description"], action_types_table=action_types
        )

        parsed_events = self._get_parsed_events(stub_test=stub_test)

        for action_name in parsed_events[0]["waves"]:
            if not isinstance(parsed_events[0]["waves"][action_name], np.ndarray):
                continue
            action_types.add_row(
                action_name=action_name,
                check_ragged=False,
            )

        for trial_events in parsed_events:
            waves = trial_events["waves"]
            action_names = action_types.action_name[:]
            for action_name in action_names:
                if len(waves[action_name]) == 0:
                    # print(f"Skipping action {action_name} with no recorded times.")
                    continue
                elif len(waves[action_name].shape) == 1:
                    actions_table.add_row(
                        action_type=action_types.action_name[:].index(action_name),
                        timestamp=waves[action_name][0],
                        duration=waves[action_name][1] - waves[action_name][0],
                        value="In",  # enter state
                        check_ragged=False,
                    )
                else:
                    for wave_time in waves[action_name]:
                        actions_table.add_row(
                            action_type=action_types.action_name[:].index(action_name),
                            timestamp=wave_time[0],
                            duration=wave_time[1] - wave_time[0],
                            value="In",  # value feels redundant here, but it is required by the ActionsTable
                            check_ragged=False,
                        )
        return action_types, actions_table

    def create_task_arguments(self) -> TaskArgumentsTable:

        task_arguments = TaskArgumentsTable(description="Task arguments for the task.")

        all_columns = list(self.saved.keys())

        columns_to_skip = [
            "raw_events",
            "parsed_events",
            "current_assembler",
            "comments",
            "my_gui",
            "my_xyfig",
            "ThisStimulus",
        ]
        all_columns = [col for col in all_columns if not any(skip in col for skip in columns_to_skip)]
        for argument_name in all_columns:
            argument_value = self.saved[argument_name]
            # expression = type(argument_value).__name__
            argument_description = "no description"  # TODO: extract this from .m if available
            if isinstance(argument_value, int):
                expression_type = "integer"  # The type of the expression
                output_type = "numeric"  # The type of the output
            elif isinstance(argument_value, float):
                expression_type = "float"
                output_type = "numeric"
            elif isinstance(argument_value, str):
                expression_type = "string"
                output_type = "text"
            elif isinstance(argument_value, np.ndarray) or isinstance(argument_value, list):
                continue  # Skip arrays for now, as they are not well defined in the context of task arguments
                expression_type = "array"
                output_type = "numeric"
            else:
                expression_type = "unknown"
                output_type = "unknown"

            task_arguments.add_row(
                argument_name=argument_name,
                argument_description=argument_description,
                expression=str(argument_value),
                expression_type=expression_type,
                output_type=output_type,
            )

        return task_arguments

    def add_task(self, nwbfile: NWBFile, metadata: dict, stub_test: bool = False) -> None:

        state_types_table, states_table = self.create_states(stub_test=stub_test, metadata=metadata)
        action_types_table, actions_table = self.create_actions(stub_test=stub_test, metadata=metadata)
        event_types_table, events_table = self.create_events(stub_test=stub_test, metadata=metadata)

        task_arguments_table = self.create_task_arguments()

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

    def add_trials(self, nwbfile: NWBFile, metadata: dict, stub_test: bool = False) -> None:
        """Add trials to the NWB file."""

        if "task_recording" not in nwbfile.acquisition:
            self.add_task(nwbfile=nwbfile, metadata=metadata, stub_test=stub_test)
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
        trial_start_times, trial_stop_times = self.get_trial_times()

        for start, stop in zip(trial_start_times, trial_stop_times):
            states_table_df = states_table[:]
            states_index_mask = (states_table_df["start_time"] >= start) & (states_table_df["stop_time"] <= stop)
            states_index_ranges = states_table_df[states_index_mask].index

            events_table_df = events_table[:]
            events_index_mask = (events_table_df["timestamp"] >= start) & (events_table_df["timestamp"] <= stop)
            events_index_ranges = events_table_df[events_index_mask].index

            actions_table_df = actions_table[:]
            actions_index_mask = (actions_table_df["timestamp"] >= start) & (actions_table_df["timestamp"] <= stop)
            actions_index_ranges = actions_table_df[actions_index_mask].index
            trials_table.add_trial(
                start_time=start,
                stop_time=stop,
                states=states_index_ranges.tolist(),
                events=events_index_ranges.tolist(),
                actions=actions_index_ranges.tolist(),
            )

        nwbfile.trials = trials_table

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict, stub_test: bool = False) -> None:
        self.add_trials(nwbfile=nwbfile, metadata=metadata, stub_test=stub_test)
        get_module(
            nwbfile=nwbfile, name="behavior", description="Behavior module"
        )  # Ensure the behavior module exists for spyglass compatibility
