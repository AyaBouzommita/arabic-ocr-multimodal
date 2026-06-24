import os
import json
from pathlib import Path
from jsonschema import Draft7Validator, ValidationError

# Get root directory of the project (2 levels up from pipeline/utils/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Alternatively, find the project root from workspace path
SCHEMAS_DIR = BASE_DIR / "schemas"


class SchemaValidationError(Exception):
    """Custom exception raised when payload validation against a JSON schema fails."""

    pass


def _load_schema(schema_name: str) -> dict:
    """Helper to load a schema file by name."""
    schema_path = SCHEMAS_DIR / schema_name
    if not schema_path.exists():
        # Fallback to current working directory if path is different (e.g. running in test context)
        fallback_path = Path.cwd() / "schemas" / schema_name
        if fallback_path.exists():
            schema_path = fallback_path
        else:
            raise FileNotFoundError(
                f"Schema file not found at {schema_path} or {fallback_path}"
            )

    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_data(data: dict, schema_name: str) -> None:
    """Generic function to validate data against a schema.

    Raises:
        SchemaValidationError: If the data does not conform to the schema.
    """
    try:
        schema = _load_schema(schema_name)
    except Exception as e:
        raise SchemaValidationError(f"Could not load schema {schema_name}: {e}")

    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)

    if errors:
        error_msgs = []
        for error in errors:
            path = " -> ".join([str(p) for p in error.path]) or "root"
            error_msgs.append(f"[{path}]: {error.message}")
        raise SchemaValidationError("Validation failed:\n" + "\n".join(error_msgs))


def validate_ocr_output(data: dict) -> None:
    """Validate data against ocr_output.schema.json."""
    validate_data(data, "ocr_output.schema.json")


def validate_detection_output(data: dict) -> None:
    """Validate data against detection_output.schema.json."""
    validate_data(data, "detection_output.schema.json")


def validate_fusion_input(data: dict) -> None:
    """Validate data against fusion_input.schema.json."""
    validate_data(data, "fusion_input.schema.json")
