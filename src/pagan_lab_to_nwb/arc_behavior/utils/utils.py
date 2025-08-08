import re
from pathlib import Path
from typing import Dict, List

import yaml
from pydantic import DirectoryPath, FilePath


def read_matlab_file(file_path: FilePath) -> str:
    """
    Read MATLAB .m file and remove line continuations.

    Parameters
    ----------
    file_path : FilePath
        Path to the MATLAB .m file.

    Returns
    -------
    str
        Content of the MATLAB file with line continuations removed.
    """
    with file_path.open("r") as f:
        content = f.read()
    return re.sub(r"\.\.\.\s*", " ", content)


def extract_param_blocks(text: str) -> List[str]:
    """
    Extract all *Param(...) blocks from the MATLAB code.

    Parameters
    ----------
    text : str
        The content of the MATLAB file as a string.

    Returns
    -------
    List[str]
        A list of strings, each representing a *Param block.
    """
    pattern = re.compile(r"\b\w*Param\s*\((.*?)\);", re.DOTALL)
    return pattern.findall(text)


def parse_param_block(block: str) -> Dict[str, str]:
    """
    Extract the parameter name and its description (TooltipString or label)
    from a *Param block.

    Parameters
    ----------
    block : str
        A single *Param(...) block as a string.

    Returns
    -------
    Dict[str, str]
        A dictionary with the parameter name as the key and its description as the value.
    """
    args = re.findall(r"'(.*?)'", block)
    if len(args) < 2:
        return {}

    if args[0].strip().lower() == "obj":
        param_name = args[1].strip()
    else:
        param_name = args[0].strip()

    description = None
    if "TooltipString" in args:
        idx = args.index("TooltipString")
        if idx + 1 < len(args):
            description = args[idx + 1].strip()
    elif "label" in args:
        idx = args.index("label")
        if idx + 1 < len(args):
            description = args[idx + 1].strip()

    if description:
        return {param_name: {"description": description}}
    else:
        return {}


def parse_matlab_file(file_path: FilePath) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Parse a single MATLAB .m file and return a dict of parameters.

    Parameters
    ----------
    file_path : FilePath
        Path to the MATLAB .m file.

    Returns
    -------
    Dict[str, Dict[str, Dict[str, str]]]
        A dictionary with section names as keys and dictionaries of parameters as values.
    """
    section_name = file_path.stem
    section_params = {}

    print(f"Processing file: '{file_path.name}'")
    content = read_matlab_file(file_path=file_path)
    blocks = extract_param_blocks(text=content)

    for block in blocks:
        param_entry = parse_param_block(block=block)
        section_params.update(param_entry)

    return {section_name: section_params}


def parse_all_matlab_files(folder_path: DirectoryPath) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Parse all MATLAB .m files in the given folder.

    Parameters
    ----------
    folder_path : DirectoryPath
        Path to the folder containing MATLAB .m files.

    Returns
    -------
    Dict[str, Dict[str, Dict[str, str]]]
        A dictionary with section names as keys and dictionaries of parameters as values.
    """
    results = {}
    for file_path in folder_path.glob("*.m"):
        section_data = parse_matlab_file(file_path)
        results.update(section_data)
    return results


def write_yaml(data: Dict, output_path: Path) -> None:
    """
    Write the results dictionary to a YAML file.

    Parameters
    ----------
    data : Dict
        The dictionary containing the parsed parameters.
    output_path : Path
        The path where the YAML file will be saved.
    """
    with output_path.open("w") as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)
    print(f"YAML saved to: '{output_path}'")


def get_description_from_arguments_metadata(
    arguments_metadata: Dict[str, Dict[str, Dict[str, str]]], argument_name: str
) -> str:
    """
    Get the description of a specific argument from the arguments metadata.

    Parameters
    ----------
    arguments_metadata : Dict[str, Dict[str, Dict[str, str]]]
        The metadata dictionary containing argument descriptions.
    argument_name : str
        The name of the argument in the format "SectionName_ArgumentName".

    Returns
    -------
    str
        The description of the argument, or "no description" if not found.
    """
    description = "no description"
    if arguments_metadata is None:
        return description
    try:
        section_name, arg_name = argument_name.split("_", maxsplit=1)
        if section_name in arguments_metadata and arg_name in arguments_metadata[section_name]:
            description = arguments_metadata[section_name][arg_name]["description"]
    except ValueError:
        raise ValueError(f"Argument name '{argument_name}' is not in the expected format 'SectionName_ArgumentName'.")
    return description


# -------------------------------
# If running directly (CLI use)
# -------------------------------
if __name__ == "__main__":
    protocol_code_folder_path = Path('/Users/weian/data/Pagan/Protocol "TaskSwitch6"/Protocol_code')
    yaml_file_path = protocol_code_folder_path / "task_switch6_params.yaml"

    all_results = parse_all_matlab_files(protocol_code_folder_path)
    write_yaml(all_results, yaml_file_path)
