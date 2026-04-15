"""Primary class for converting BControl behavioral data to NWB format."""

import re
from datetime import datetime
from pathlib import Path
from warnings import warn

import numpy as np
from pydantic import validate_call
from pynwb.device import Device
from pynwb.file import NWBFile

from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.utils import DeepDict, get_base_schema, get_schema_from_hdmf_class
from pagan_lab_to_nwb.interfaces._optogenetics import add_optogenetic_series_to_nwbfile
from pagan_lab_to_nwb.interfaces._stimulus import add_stimulus_to_trials
from pagan_lab_to_nwb.interfaces._task_recording import add_task_recording_to_nwbfile
from pagan_lab_to_nwb.interfaces._trials import (
    add_task_arguments,
    add_trials_to_nwbfile,
)


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
            The name of the starting state for the trials.
        verbose : bool, default: False
        """
        super().__init__(file_path=file_path)
        self.verbose = verbose
        self.starting_state = starting_state

    def _read_file(self):
        """Read the BControl .mat file and cache ``saved`` and ``saved_history``."""
        from pymatreader import read_mat

        if hasattr(self, "saved") and hasattr(self, "saved_history"):
            return

        mat_data = read_mat(self.source_data["file_path"])
        if "saved" not in mat_data and "saved_history" not in mat_data:
            raise ValueError(
                f"The provided .mat file does not contain the expected 'saved' or 'saved_history' fields. "
                f"Keys: {list(mat_data.keys())}."
            )
        self.saved = mat_data["saved"]
        self.saved_history = mat_data["saved_history"]

    def _get_parsed_events(self, stub_test: bool = False) -> list[dict]:
        """Return the list of per-trial parsed-events dicts from ``saved_history``.

        Parameters
        ----------
        stub_test : bool, default: False
            If True, cap at 100 trials for fast testing.
        """
        self._read_file()
        if "ProtocolsSection_parsed_events" not in self.saved_history:
            raise ValueError(
                "The saved_history does not contain 'ProtocolsSection_parsed_events'. "
                "Please ensure the BControl data file is correctly formatted."
            )
        num_completed_trials = self.saved.get("ProtocolsSection_n_completed_trials", None)
        if int(num_completed_trials) == 0:
            raise ValueError(
                "Expected at least 1 completed trial, got 0 for 'ProtocolsSection_n_completed_trials'. "
                "Please check the BControl data file."
            )
        parsed_events = self.saved_history["ProtocolsSection_parsed_events"]
        if int(num_completed_trials) == 1 and isinstance(parsed_events, dict):
            parsed_events = [parsed_events]
        if not isinstance(parsed_events, list):
            raise ValueError(
                f"Expected 'ProtocolsSection_parsed_events' to be a list, got {type(parsed_events)}. "
                "Please check the format of the BControl data file."
            )
        if stub_test:
            parsed_events = parsed_events[: min(len(parsed_events), 100)]
        return parsed_events

    def get_trial_times(self, stub_test: bool = False) -> tuple[list[float], list[float]]:
        """Return (start_times, stop_times) for all trials.

        Parameters
        ----------
        stub_test : bool, default: False
            If True, only the first 100 trials are returned.
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

        # Behavior table descriptions and device metadata are defined in metadata.yaml
        # (arc_behavior/metadata.yaml) and merged into this dict by convert_session.py.
        # Only the key structure is seeded here so downstream code can safely navigate
        # metadata["Behavior"] before the YAML merge happens.
        metadata["Behavior"] = dict(
            Device=dict(name="BControl", manufacturer=""),
            StateTypesTable=dict(description=""),
            StatesTable=dict(description=""),
            ActionsTable=dict(description=""),
            ActionTypesTable=dict(description=""),
            EventTypesTable=dict(description=""),
            EventsTable=dict(description=""),
            TrialsTable=dict(description=""),
            TaskArgumentsTable=dict(description=""),
        )

        self._read_file()

        # Extract session start time from the protocol title
        protocol_title_keys = [k for k in self.saved.keys() if "prot_title" in k]
        if len(protocol_title_keys) == 1:
            protocol_title = self.saved[protocol_title_keys[0]]
            # Expected format: '... Started at HH:MM, Ended at HH:MM'
            match = re.search(r"Started at (\d{2}:\d{2})", protocol_title)
            if not match:
                raise ValueError(
                    f"Could not extract session start time from protocol title: '{protocol_title}'. "
                    "Expected format is 'Started at HH:MM'. "
                    "Sessions without this substring are excluded from conversion."
                )
            if "SavingSection_SaveTime" in self.saved:
                save_time = self.saved["SavingSection_SaveTime"]  # '15-Aug-2019 13:19:41'
                date_str = save_time.split()[0]
                try:
                    session_start_time = datetime.strptime(f"{date_str} {match.group(1)}", "%d-%b-%Y %H:%M")
                    metadata["NWBFile"]["session_start_time"] = session_start_time
                except ValueError as e:
                    warn(
                        f"Failed to parse session start time from protocol title '{protocol_title}' "
                        f"and save time '{save_time}': {e}"
                    )

        # Extract experimenter comments
        notes_parts = []
        comments_arr = self.saved.get("CommentsSection_comments", np.array([]))
        if isinstance(comments_arr, np.ndarray) and comments_arr.size > 0:
            notes_parts.append("\n".join(s.replace("\x00", "").strip() for s in comments_arr.tolist()))
        overall_arr = self.saved.get("CommentsSection_overall_comments", np.array([]))
        if isinstance(overall_arr, np.ndarray) and overall_arr.size > 0:
            notes_parts.append(
                "Overall comments:\n" + "\n".join(s.replace("\x00", "").strip() for s in overall_arr.tolist())
            )
        if notes_parts:
            metadata["NWBFile"]["notes"] = "\n\n".join(notes_parts)

        return metadata

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: dict,
        arguments_to_exclude: list[str] | None = None,
        arguments_metadata: dict | None = None,
        stub_test: bool = False,
    ) -> None:
        self._read_file()
        parsed_events = self._get_parsed_events(stub_test=stub_test)

        add_task_recording_to_nwbfile(nwbfile, parsed_events, self.starting_state, metadata)
        add_trials_to_nwbfile(nwbfile, parsed_events, self.starting_state, metadata)
        add_stimulus_to_trials(nwbfile, self.saved_history, metadata)
        add_task_arguments(
            nwbfile,
            self.saved,
            arguments_to_exclude=arguments_to_exclude,
            arguments_metadata=arguments_metadata,
            stub_test=stub_test,
        )
        add_optogenetic_series_to_nwbfile(nwbfile, self.saved_history, parsed_events, metadata, stub_test=stub_test)
