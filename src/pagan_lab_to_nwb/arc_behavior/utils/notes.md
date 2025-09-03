# Task Parameter Description Extraction from MATLAB Files

This document explains how to extract parameter descriptions from MATLAB (.m) files used in BControl protocols and generate a YAML file containing these descriptions. The YAML file can then be used during NWB conversion to provide meaningful descriptions for task arguments.

## Overview

BControl MATLAB protocols often define task parameters using functions like `MenuParam()`, `NumeditParam()`, `TaskParam()` etc.
These parameter definitions frequently include descriptions in `TooltipString` or `label` fields.
The utilities `parse_all_matlab_files()` and `write_yaml()` allow you to extract these descriptions automatically.

## How to Use

### 1. Parse MATLAB Files and Generate YAML

```python
from pathlib import Path
from pagan_lab_to_nwb.arc_behavior.utils import parse_all_matlab_files, write_yaml

# Specify the folder containing the .m files
protocol_code_folder = Path('/path/to/Protocol "TaskSwitch6"/Protocol_code')

# Parse all .m files in the folder
parameters = parse_all_matlab_files(protocol_code_folder)

# Write the extracted descriptions to a YAML file
yaml_path = protocol_code_folder / "protocol_parameters.yaml"
write_yaml(parameters, yaml_path)
```

### 2. Use the Generated YAML in NWB Conversion

```python
from pathlib import Path
from neuroconv.utils import load_dict_from_file

from pagan_lab_to_nwb.arc_behavior.nwbconverter import ArcBehaviorNWBConverter

# Load the parameter descriptions
yaml_file_path = Path('/path/to/protocol_parameters.yaml')
arguments_metadata = load_dict_from_file(yaml_file_path)

# Use in conversion options
conversion_options = dict(Behavior=dict(arguments_metadata=load_dict_from_file(yaml_file_path)))

# Path to BControl data file
file_path = Path('/path/to/Protocol "TaskSwitch6"/data_@TaskSwitch6_Nuria_H7015_250516a.mat')
# Define as source data for conversion
source_data = dict(Behavior=dict(file_path=file_path))
# Pass to converter
converter = ArcBehaviorNWBConverter(source_data=source_data)
# Get metadata from the converter
metadata = converter.get_metadata()
# Run conversion with metadata and options
converter.run_conversion(
    metadata=metadata,
    nwbfile_path=Path('/path/to/output.nwb'),
    conversion_options=conversion_options,
    overwrite=True
)
```

## Example YAML Output

The generated YAML file organizes parameter descriptions by section (derived from the .m filename) and parameter name:

```yaml
ProtocolsSection:
  parsed_events:
    description: Events parsed from the raw events
  training_stage:
    description: the current training stage
  block_dur:
    description: duration of each block in seconds

StimulusSection:
  volume:
    description: Sound volume (0-1)
  tone_frequency:
    description: Frequency of the tone in Hz
```

## Manual Editing

The generated YAML file can be manually edited to:

1. Add descriptions for parameters that don't have `TooltipString` or `label` defined
2. Correct or improve automatically extracted descriptions
3. Add descriptions for parameters that aren't defined using the standard pattern
4. Remove unnecessary parameters

After editing, the YAML file can be used in the conversion process as shown above.

## How It Works

The extraction process:

1. Scans all .m files in the specified directory
2. For each file, extracts all `*Param(...)` blocks (e.g., `MenuParam`, `NumeditParam`)
3. Parses each block to find the parameter name and any associated `TooltipString` or `label`
4. Organizes the results by section (filename) and parameter name
5. Writes the structured data to a YAML file

The section name is determined from the filename. For example, parameters from `ProtocolsSection.m` will be under the `ProtocolsSection` key in the YAML.
