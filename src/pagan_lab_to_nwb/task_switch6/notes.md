# Notes concerning the conversion of the TaskSwitch6 protocol

## Task mapping to NWB

Using the [ndx-structured-behavior](https://github.com/rly/ndx-structured-behavior) extension, the TaskSwitch6 protocol can be represented in NWB as follows:

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#ffffff', 'primaryBorderColor': '#144E73', 'lineColor': '#D96F32'}}}%%

classDiagram
    direction TB

    class TaskRecording {
        <<NWBDataInterface>>
        --------------------------------------
        groups
        --------------------------------------
        events : EventsTable
        states : StatesTable
        actions : ActionsTable
    }

    class Task {
        <<LabMetaData>>
        --------------------------------------
        datasets
        --------------------------------------
        TaskProgram : optional
        TaskSchema : optional
        --------------------------------------
        groups
        --------------------------------------
        event_types : EventTypesTable
        state_types : StateTypesTable
        action_types : ActionTypesTable
        task_arguments : TaskArgumentsTable
    }

    class TrialsTable {
        <<TimeIntervals>>
        --------------------------------------
        datasets
        --------------------------------------
        states : DynamicTableRegion
        states_index : VectorIndex, optional
        events : DynamicTableRegion
        events_index : VectorIndex, optional
        actions : DynamicTableRegion
        actions_index : VectorIndex, optional
    }

    class StateTypesTable {
        <<DynamicTable>>
        --------------------------------------
        datasets
        --------------------------------------
        state_name : VectorData
    }

    class StatesTable {
        <<TimeIntervals>>
        --------------------------------------
        datasets
        --------------------------------------
        state_type : DynamicTableRegion
    }

    class EventTypesTable {
        <<DynamicTable>>
        --------------------------------------
        datasets
        --------------------------------------
        event_name : VectorData
    }

    class EventsTable {
        <<DynamicTable>>
        --------------------------------------
        datasets
        --------------------------------------
        timestamp : VectorData, float32
        event_type : DynamicTableRegion
        value : VectorData, text
    }

    class ActionsTable {
        <<DynamicTable>>
        --------------------------------------
        datasets
        --------------------------------------
        timestamp : VectorData, float32
        action_type : DynamicTableRegion
        value : VectorData, text
    }

    class ActionTypesTable {
        <<DynamicTable>>
        --------------------------------------
        datasets
        --------------------------------------
        action_name : VectorData
    }

    class TaskArgumentsTable {
        <<DynamicTable>>
        --------------------------------------
        datasets
        --------------------------------------
        argument_name : VectorData, text
        argument_description : VectorData, text
        expression : VectorData, text
        expression_type : VectorData, text
        output_type : VectorData, text
    }

    Task *-- EventTypesTable : contains
    Task *-- StateTypesTable : contains
    Task *-- ActionTypesTable : contains
    Task *-- TaskArgumentsTable : contains
    TaskRecording *-- EventsTable : contains
    TaskRecording *-- StatesTable : contains
    TaskRecording *-- ActionsTable : contains

```
