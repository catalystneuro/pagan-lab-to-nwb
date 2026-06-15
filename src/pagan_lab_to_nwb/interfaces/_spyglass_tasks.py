"""Shared helper for the Spyglass-required ``processing["tasks"]`` table.

Spyglass's ``common_task.TaskEpoch.make()`` reads task metadata from a
``DynamicTable`` named after the BControl protocol inside
``nwbfile.processing["tasks"]``. Both the spike-sorting and video interfaces
need to register this table for the session's protocol; this helper makes
that idempotent so whichever interface runs first creates the table and a
later call for the same protocol is a no-op.
"""

import numpy as np
from hdmf.common import DynamicTable
from pynwb.file import NWBFile

from neuroconv.tools import get_module


def add_spyglass_task_table(
    nwbfile: NWBFile,
    protocol: str,
    camera_id: list[int] | np.ndarray | None = None,
    task_epochs: list[int] | None = None,
) -> None:
    """Add the Spyglass ``tasks`` processing-module table for ``protocol`` (no-op if present)."""
    tasks_module = get_module(nwbfile, name="tasks", description="tasks module")

    if protocol in tasks_module.data_interfaces:
        return

    task_table = DynamicTable(name=protocol, description=f"{protocol} behavioral task")
    task_table.add_column(name="task_name", description="Name of the task")
    task_table.add_column(name="task_description", description="Description of the task")
    task_table.add_column(name="task_type", description="Type of task (required by Spyglass Task table)")
    task_table.add_column(name="task_subtype", description="Subtype of task (required by Spyglass Task table)")
    task_table.add_column(name="task_environment", description="Recording environment")
    task_table.add_column(name="camera_id", description="Camera IDs used during this task")
    task_table.add_column(name="task_epochs", description="Epoch indices for this task")
    task_table.add_row(
        task_name=protocol,
        task_description="Auditory decision-making task-switching paradigm (BControl)",
        task_type="auditory decision-making",
        task_subtype="task-switching",
        task_environment="behavioral_box",
        camera_id=camera_id if camera_id is not None else np.array([], dtype=np.int32),
        task_epochs=task_epochs if task_epochs is not None else [1],
    )
    tasks_module.add(task_table)
