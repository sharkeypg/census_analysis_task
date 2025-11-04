import pandas as pd
from typing import List, Dict

def extract_schema_lines(metadata_filepath: str) -> List[str]:
    """
    Find schema in metadata file
    """

    with open(metadata_filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find the start of the last schema section
    # The section starts after the line containing '- 50000, 50000+.'
    start_idx = 0
    for i, line in enumerate(lines):
        if "- 50000, 50000+." in line:
            start_idx = i + 1
            break

    # Extract schema lines (skip empty lines)
    schema_lines = [line.strip() for line in lines[start_idx:] if line.strip()]

    schema_lines.append("salary: -50000, 50000")
    return schema_lines

def extract_schema_as_dict(schema_lines: List[str]) -> Dict:
    """
    Extract schema and values as dictionary
    """

    schema_dict = {}

    for line in schema_lines:
        if ":" in line:
            col, values = line.split(":", 1)
            col = col.strip()
            values = values.strip().rstrip(".")  # remove ending period
            if values.lower() == "continuous":
                schema_dict[col] = [values]
            elif values.lower() == "ignore":
                pass
            else:
                schema_dict[col] = [v.strip() for v in values.split(",")]

    return schema_dict

def extract_schema(metadata_filepath: str) -> Dict:
    """
    Extract data schema from metadata file
    """

    schema_lines = extract_schema_lines(metadata_filepath)
    schema_dict = extract_schema_as_dict(schema_lines)
    return schema_dict


                                
