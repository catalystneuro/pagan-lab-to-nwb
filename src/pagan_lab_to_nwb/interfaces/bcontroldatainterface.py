"""Primary class for converting experiment-specific behavior."""

import re
from datetime import datetime
from pathlib import Path

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
from pynwb.file import NWBFile

from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.tools import get_module
from neuroconv.utils import DeepDict


# TODO: implement this interface in NeuroConv
class BControlBehaviorInterface(BaseDataInterface):
    """Interface for converting BControl behavioral data files to NWB format."""

    display_name = "BControl Behavior"
    keywords = ("behavior", "states", "events", "actions", "trials", "task recording")
    associated_suffixes = (".mat",)
    info = "Interface for behavior data from BControl to an NWB file."

    @validate_call
    def __init__(self, file_path: Path, verbose: bool = False):
        """
        Data interface for writing BControl behavioral data to an NWB file.

        Writes behavior data using the ndx-structured-behavior extension.

        Parameters
        ----------
        file_path : Path or str
            The path to the BControl data file to be converted.
        verbose : bool, default: False
        """

        self.verbose = verbose
        super().__init__(file_path=file_path)

    def _read_data(self):
        """Read the BControl data file and return the data."""
        from pymatreader import read_mat

        if hasattr(self, "saved") and hasattr(self, "saved_history"):
            return

        # Read the .mat file
        mat_data = read_mat(self.source_data["file_path"])

        # Extract relevant data from the mat file
        # This is a placeholder; actual extraction logic will depend on the structure of the BControl data
        self.saved = mat_data.get("saved", None)
        self.saved_history = mat_data.get("saved_history", None)

    def get_trial_times(self, stub_test: bool = False) -> (list[float], list[float]):
        parsed_events = self.saved_history["ProtocolsSection_parsed_events"]  # list of n trials
        num_trials = len(parsed_events)
        if stub_test:
            num_trials = min(num_trials, 100)
            parsed_events = parsed_events[:num_trials]

        trial_start_times = [events["states"]["state_0"][0][1] for events in parsed_events]
        trial_end_times = [events["states"]["state_0"][1][0] for events in parsed_events]

        return trial_start_times, trial_end_times

    def get_metadata(self) -> DeepDict:
        metadata = super().get_metadata()

        default_device_metadata = dict(
            name="BControl",
            manufacturer="Example Manufacturer",  # TODO: ask from lab
        )
        metadata["Behavior"] = dict(Device=default_device_metadata)

        self._read_data()
        # extract session_start_time from the protocol title
        if self.saved is not None:
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
                    session_start_time = datetime.strptime(f"{date_str} {time_str}", "%d-%b-%Y %H:%M")
                    metadata["NWBFile"]["session_start_time"] = session_start_time

        return metadata

    def create_states(self, stub_test: bool = False) -> tuple[StateTypesTable, StatesTable]:
        # todo: add metadata for event types and events tables
        state_types = StateTypesTable(description="State Types Table")
        states_table = StatesTable(description="State Table", state_types_table=state_types)

        parsed_events = self.saved_history["ProtocolsSection_parsed_events"]
        num_trials = self.saved["ProtocolsSection_n_completed_trials"]
        if stub_test:
            num_trials = min(num_trials, 100)
            parsed_events = parsed_events[:num_trials]

        if num_trials == 1:
            parsed_events = [parsed_events]

        for state_name in parsed_events[0]["states"]:
            if not isinstance(parsed_events[0]["states"][state_name], np.ndarray):
                continue

            state_types.add_row(
                state_name=state_name,
                check_ragged=False,
            )

        for trial_events in parsed_events:
            states = trial_events["states"]
            state_names = state_types.state_name[:]
            for state_name in state_names:
                if state_name == "state_0":
                    state_start_time = trial_events["states"]["state_0"][0][1]  # start time of the trial
                    state_stop_time = trial_events["states"]["state_0"][1][0]  # end time of the trial
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

    def create_events(self, stub_test: bool = False) -> tuple[EventTypesTable, EventsTable]:
        # todo: add metadata for event types and events tables
        event_types = EventTypesTable(description="Event Types Table")
        events_table = EventsTable(description="Events Table", event_types_table=event_types)

        parsed_events = self.saved_history["ProtocolsSection_parsed_events"]
        num_trials = len(parsed_events)
        if stub_test:
            stub_trials = min(num_trials, 100)
            parsed_events = parsed_events[:stub_trials]

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
                # TODO: event type is an integer and not region TypeError: EventsTable.add_row: incorrect type for 'event_type' (got 'DynamicTableRegion', expected 'int'), missing argument 'value'
                event_type = event_types.event_name[:].index(event_name)
                if len(pokes[event_name]) == 0:
                    print(f"Skipping event {event_name} with no recorded times.")
                    continue
                # todo: skip nan values
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

    def create_actions(self, stub_test: bool = False) -> tuple[ActionTypesTable, ActionsTable]:
        action_types = ActionTypesTable(description="Action Types Table")
        actions_table = ActionsTable(description="Actions Table", action_types_table=action_types)

        parsed_events = self.saved_history["ProtocolsSection_parsed_events"]
        num_trials = len(parsed_events)
        if stub_test:
            stub_trials = min(num_trials, 100)
            parsed_events = parsed_events[:stub_trials]

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
                    print(f"Skipping action {action_name} with no recorded times.")
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

        state_types_table, states_table = self.create_states(stub_test=stub_test)
        action_types_table, actions_table = self.create_actions(stub_test=stub_test)
        event_types_table, events_table = self.create_events(stub_test=stub_test)

        task_arguments_table = self.create_task_arguments()

        task = Task(
            event_types=event_types_table,
            state_types=state_types_table,
            action_types=action_types_table,
            task_arguments=task_arguments_table,
        )
        # Add the task
        nwbfile.add_lab_meta_data(task)

        # To add these tables to acquisitions in an NWBFile, they are stored within TaskRecording.
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
            description="Trials Table",  # TODO: extract from metadata
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
