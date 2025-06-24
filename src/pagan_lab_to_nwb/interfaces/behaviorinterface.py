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
)
from pydantic import validate_call
from pynwb.file import NWBFile

from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.utils import DeepDict


# TODO: implement this interface in NeuroConv
class BControlBehaviorInterface(BaseDataInterface):
    """Interface for converting BControl behavioral data files to NWB format."""

    display_name = "BControl Behavior"
    keywords = ("behavior", "states", "events", "actions", "trials", "task recording")
    associated_suffixes = (".mat",)
    info = "Interface for behavior data from BControl to an NWB file."

    @validate_call
    def __init__(self, file_path: Path | str, verbose: bool = False):
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

        # Extract session_start_time
        # 'TaskSwitch6 - on rig brodyrigws32.princeton.edu : Marino, P131.  Started at 11:41, Ended at 13:19'
        # extract the datetime from "Started at" from the title and date from "SavingSection_SaveTime"
        prot_title = mat_data["saved"]["TaskSwitch6_prot_title"]

        # Extract relevant data from the mat file
        # This is a placeholder; actual extraction logic will depend on the structure of the BControl data
        self.saved = mat_data.get("saved", None)
        self.saved_history = mat_data.get("saved_history", None)

    def get_trial_times(self) -> (list[float], list[float]):
        parsed_events = self.saved_history["ProtocolsSection_parsed_events"]  # list of n trials

        trial_start_times = [events["states"]["state_0"][0][1] for events in parsed_events]
        trial_end_times = [events["states"]["state_0"][1][0] for events in parsed_events]

        return trial_start_times, trial_end_times

    # def add_trials(self, nwbfile: NWBFile, metadata: dict) -> None:
    #     """Add trials to the NWB file."""
    #     # This is a placeholder; actual implementation will depend on the structure of the BControl data
    #     trials_table = TrialsTable(name="Trials")
    #     nwbfile.add_acquisition(trials_table)

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

    def create_states(self) -> tuple[StateTypesTable, StatesTable]:
        # todo: add metadata for event types and events tables
        state_types = StateTypesTable(description="State Types Table")
        states_table = StatesTable(description="State Table", state_types_table=state_types)

        parsed_events = self.saved_history["ProtocolsSection_parsed_events"]
        num_trials = self.saved["ProtocolsSection_n_completed_trials"]

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
                state_type_region = state_types.create_region(
                    name=state_name,
                    region=[state_names.index(state_name)],
                    description=f"The reference for {state_name} in the state types table.",
                )
                if state_name == "state_0":
                    state_start_time = trial_events["states"]["state_0"][0][1]  # start time of the trial
                    state_stop_time = trial_events["states"]["state_0"][1][0]  # end time of the trial
                    states_table.add_row(
                        state_type=state_type_region,
                        start_time=state_start_time,
                        stop_time=state_stop_time,
                        check_ragged=False,
                    )
                elif len(states[state_name]) == 0:
                    # print(f"Skipping state {state_name} with no recorded times.")
                    continue
                elif len(states[state_name].shape) == 1:
                    states_table.add_row(
                        state_type=state_type_region,
                        start_time=states[state_name][0],
                        stop_time=states[state_name][1],
                        check_ragged=False,
                    )
                else:
                    for state_time in states[state_name]:
                        states_table.add_row(
                            state_type=state_type_region,
                            start_time=state_time[0],
                            stop_time=state_time[1],
                            check_ragged=False,
                        )

        return state_types, states_table

    def create_events(self) -> tuple[EventTypesTable, EventsTable]:
        # todo: add metadata for event types and events tables
        event_types = EventTypesTable(description="Event Types Table")
        events_table = EventsTable(description="Events Table", event_types_table=event_types)

        parsed_events = self.saved_history["ProtocolsSection_parsed_events"]
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

    def create_actions(self) -> tuple[ActionTypesTable, ActionsTable]:
        action_types = ActionTypesTable(description="Action Types Table")
        actions_table = ActionsTable(description="Actions Table", action_types_table=action_types)

        parsed_events = self.saved_history["ProtocolsSection_parsed_events"]
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
        pass

    def add_task(self, nwbfile: NWBFile, metadata: dict) -> None:

        state_types_table, states_table = self.create_states()
        action_types_table, actions_table = self.create_actions()
        event_types_table, events_table = self.create_events()

        # task_arguments_table = self.create_task_arguments()

        task = Task(
            event_types=event_types_table,
            state_types=state_types_table,
            action_types=action_types_table,
            # task_arguments=task_arguments_table,
        )
        # Add the task
        nwbfile.add_lab_meta_data(task)

        # To add these tables to acquisitions in an NWBFile, they are stored within TaskRecording.
        recording = TaskRecording(events=events_table, states=states_table, actions=actions_table)
        nwbfile.add_acquisition(recording)

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict):
        self.add_task(nwbfile=nwbfile, metadata=metadata)
