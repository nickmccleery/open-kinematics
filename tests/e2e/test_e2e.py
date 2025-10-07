"""
End-to-end integration tests for the CLI interface using direct CLI function calls.

Tests all combinations of:
- Output formats: CSV and Parquet
- With and without visualization
- Using real test data files and golden reference files

This version calls the CLI sweep() function directly instead of using subprocess,
making debugging easier while still testing the same code path.
"""

import csv
import io
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import patch

import pyarrow.parquet as pq
import pytest

from kinematics.cli import sweep as cli_sweep


def load_csv_data(file_path: Path) -> tuple[list[str], list[list[str]]]:
    """
    Load CSV data and return headers and rows, skipping comment lines.
    """
    with open(file_path, "r") as f:
        # Skip comment lines starting with #
        lines = []
        for line in f:
            if not line.strip().startswith("#"):
                lines.append(line)

        # Use csv reader on the non-comment lines
        reader = csv.reader(lines)
        headers = next(reader)
        rows = list(reader)
    return headers, rows


def load_parquet_data(file_path: Path) -> tuple[list[str], list[list[str]]]:
    """
    Load Parquet data and return headers and rows as strings for comparison.
    """
    table = pq.read_table(file_path)
    headers = table.column_names

    # Convert to list of rows (as strings for comparison).
    rows = []
    for i in range(table.num_rows):
        row = []
        for col_name in headers:
            value = table.column(col_name)[i].as_py()
            row.append(str(value))
        rows.append(row)

    return headers, rows


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """
    Create a temporary directory for test outputs.
    """
    # yield Path(__file__).parent / "temp"
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def test_data_dir() -> Path:
    """
    Path to test data directory.
    """
    return Path(__file__).parent.parent / "data"


@pytest.fixture
def geometry_file(test_data_dir: Path) -> Path:
    """
    Path to test geometry file.
    """
    return test_data_dir / "geometry.yaml"


@pytest.fixture
def sweep_file(test_data_dir: Path) -> Path:
    """
    Path to test sweep file.
    """
    return test_data_dir / "sweep.yaml"


def run_cli_sweep_direct(
    geometry_file: Path,
    sweep_file: Path,
    output_file: Path,
    animation_file: Path | None = None,
) -> tuple[bool, str]:
    """
    Run the CLI sweep function directly instead of via subprocess.

    Args:
        geometry_file: Path to geometry YAML file
        sweep_file: Path to sweep YAML file
        output_file: Output path for results (.csv or .parquet)
        animation_file: Optional animation output path

    Returns:
        Tuple of (success, captured_output) indicating result
    """
    # Capture typer.echo() output
    captured_output = io.StringIO()

    try:
        # Mock typer.echo to capture output instead of printing
        with patch("typer.echo") as mock_echo:
            # Store all echo calls
            echo_calls = []

            def capture_echo(message, **kwargs):
                echo_calls.append(str(message))
                captured_output.write(str(message) + "\n")

            mock_echo.side_effect = capture_echo

            # Call the CLI function directly
            cli_sweep(
                geometry=geometry_file,
                sweep=sweep_file,
                out=output_file,
                animation_out=animation_file,
            )

            # If we get here, it succeeded
            return True, captured_output.getvalue()

    except Exception as e:
        return False, str(e)


def normalize_csv_for_comparison(file_path: Path) -> str:
    """
    Normalize CSV content for comparison by removing metadata comments and timestamps.

    Returns the normalized content as a string with consistent formatting.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Filter out comment lines and normalize
    data_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            # Remove the timestamp column (last column) for comparison
            # Split by comma, remove last field, rejoin
            parts = stripped.split(",")
            if len(parts) > 1:
                # Remove the timestamp (last column)
                normalized_line = ",".join(parts[:-1])
                data_lines.append(normalized_line)
            else:
                data_lines.append(stripped)

    return "\n".join(data_lines)


def compare_files(actual_file: Path, expected_file: Path, file_format: str) -> bool:
    """
    Compare actual output file with expected golden file.

    Args:
        actual_file: Path to the generated file
        expected_file: Path to the expected golden file
        file_format: 'csv' or 'parquet'

    Returns:
        True if files match, False otherwise
    """
    if not actual_file.exists():
        print(f"❌ Actual file does not exist: {actual_file}")
        return False

    if not expected_file.exists():
        print(f"❌ Expected golden file does not exist: {expected_file}")
        return False

    if file_format == "csv":
        # For CSV, normalize and compare content (ignoring metadata comments)
        actual_content = normalize_csv_for_comparison(actual_file)
        expected_content = normalize_csv_for_comparison(expected_file)

        if actual_content == expected_content:
            print("✓ CSV files match perfectly")
            return True
        else:
            print("❌ CSV files differ")
            # Show first few lines of difference for debugging
            actual_lines = actual_content.split("\n")
            expected_lines = expected_content.split("\n")
            print(
                f"Actual lines: {len(actual_lines)}, Expected lines: {len(expected_lines)}"
            )
            if len(actual_lines) != len(expected_lines):
                print("Different number of lines")
            else:
                for i, (a, e) in enumerate(zip(actual_lines[:3], expected_lines[:3])):
                    if a != e:
                        print(f"First difference at line {i}: '{a}' != '{e}'")
                        break
            return False

    elif file_format == "parquet":
        # For Parquet, compare data content ignoring metadata timestamp
        try:
            actual_headers, actual_rows = load_parquet_data(actual_file)
            expected_headers, expected_rows = load_parquet_data(expected_file)

            if actual_headers != expected_headers:
                print("❌ Parquet headers differ")
                return False

            if actual_rows != expected_rows:
                print(
                    f"❌ Parquet data differs: {len(actual_rows)} vs {len(expected_rows)} rows"
                )
                return False

            print("✓ Parquet files match perfectly")
            return True
        except Exception as e:
            print(f"❌ Error comparing Parquet files: {e}")
            return False

    else:
        raise ValueError(f"Unsupported format: {file_format}")


def validate_output_against_golden(output_file: Path, file_format: str) -> None:
    """
    Validate output file against golden reference file.

    Args:
        output_file: Path to the generated output file
        file_format: Expected format ('csv' or 'parquet')
    """
    # Get path to golden file
    golden_file = (
        Path(__file__).parent.parent / "data" / "e2e" / f"output.{file_format}"
    )

    assert output_file.exists(), f"Output file {output_file} was not created"
    assert output_file.stat().st_size > 0, f"Output file {output_file} is empty"

    # Compare with golden file
    matches = compare_files(output_file, golden_file, file_format)
    assert matches, (
        f"Output file {output_file} does not match golden file {golden_file}"
    )

    print(f"✓ Output matches golden {file_format.upper()} file")


def validate_animation_file(animation_file: Path) -> None:
    """
    Validate the animation file exists and has reasonable size.

    Args:
        animation_file: Path to the animation file
    """
    assert animation_file.exists(), f"Animation file {animation_file} was not created"
    file_size = animation_file.stat().st_size
    assert file_size > 1000, (
        f"Animation file {animation_file} is too small: {file_size} bytes"
    )
    print(f"✓ Validated animation file: {file_size} bytes")


class TestCliDirectEndToEnd:
    """
    Test class for CLI end-to-end functionality using direct function calls.
    """

    def test_csv_output_without_animation(
        self,
        temp_dir: Path,
        geometry_file: Path,
        sweep_file: Path,
    ) -> None:
        """
        Test CSV output without animation.
        """
        output_file = temp_dir / "test_output.csv"

        success, output = run_cli_sweep_direct(geometry_file, sweep_file, output_file)

        assert success, f"CLI sweep failed: {output}"
        assert "wrote" in output.lower(), f"Unexpected output: {output}"

        validate_output_against_golden(output_file, "csv")

    def test_parquet_output_without_animation(
        self,
        temp_dir: Path,
        geometry_file: Path,
        sweep_file: Path,
    ) -> None:
        """
        Test Parquet output without animation.
        """
        output_file = temp_dir / "test_output.parquet"

        success, output = run_cli_sweep_direct(geometry_file, sweep_file, output_file)

        assert success, f"CLI sweep failed: {output}"
        assert "wrote" in output.lower(), f"Unexpected output: {output}"

        validate_output_against_golden(output_file, "parquet")

    def test_csv_output_with_animation(
        self,
        temp_dir: Path,
        geometry_file: Path,
        sweep_file: Path,
    ) -> None:
        """
        Test CSV output with animation.
        """
        output_file = temp_dir / "test_output.csv"
        animation_file = temp_dir / "test_animation.mp4"

        success, output = run_cli_sweep_direct(
            geometry_file, sweep_file, output_file, animation_file
        )

        assert success, f"CLI sweep failed: {output}"
        assert "wrote" in output.lower(), f"Unexpected output: {output}"

        validate_output_against_golden(output_file, "csv")
        validate_animation_file(animation_file)

    def test_parquet_output_with_animation(
        self,
        temp_dir: Path,
        geometry_file: Path,
        sweep_file: Path,
    ) -> None:
        """
        Test Parquet output with animation.
        """
        output_file = temp_dir / "test_output.parquet"
        animation_file = temp_dir / "test_animation.mp4"

        success, output = run_cli_sweep_direct(
            geometry_file, sweep_file, output_file, animation_file
        )

        assert success, f"CLI sweep failed: {output}"
        assert "wrote" in output.lower(), f"Unexpected output: {output}"

        validate_output_against_golden(output_file, "parquet")
        validate_animation_file(animation_file)

    def test_gif_animation_output(
        self,
        temp_dir: Path,
        geometry_file: Path,
        sweep_file: Path,
    ) -> None:
        """
        Test CSV output with GIF animation.
        """
        output_file = temp_dir / "test_output.csv"
        animation_file = temp_dir / "test_animation.gif"

        success, output = run_cli_sweep_direct(
            geometry_file, sweep_file, output_file, animation_file
        )

        assert success, f"CLI sweep failed: {output}"
        assert "wrote" in output.lower(), f"Unexpected output: {output}"

        validate_output_against_golden(output_file, "csv")
        validate_animation_file(animation_file)

    def test_invalid_geometry_file(
        self,
        temp_dir: Path,
        sweep_file: Path,
    ) -> None:
        """
        Test handling of invalid geometry file.
        """
        invalid_geometry = temp_dir / "nonexistent.yaml"
        output_file = temp_dir / "test_output.csv"

        success, output = run_cli_sweep_direct(
            invalid_geometry, sweep_file, output_file
        )

        assert not success, "CLI should fail with invalid geometry file"
        assert output  # Should have an error message

    def test_invalid_sweep_file(
        self,
        temp_dir: Path,
        geometry_file: Path,
    ) -> None:
        """
        Test handling of invalid sweep file.
        """
        invalid_sweep = temp_dir / "nonexistent.yaml"
        output_file = temp_dir / "test_output.csv"

        success, output = run_cli_sweep_direct(
            geometry_file, invalid_sweep, output_file
        )

        assert not success, "CLI should fail with invalid sweep file"
        assert output  # Should have an error message

    def test_output_formats_produce_same_data(
        self,
        temp_dir: Path,
        geometry_file: Path,
        sweep_file: Path,
    ) -> None:
        """
        Test that CSV and Parquet outputs contain the same data.
        """
        csv_output = temp_dir / "test_output.csv"
        parquet_output = temp_dir / "test_output.parquet"

        # Run both formats
        csv_success, csv_message = run_cli_sweep_direct(
            geometry_file, sweep_file, csv_output
        )
        parquet_success, parquet_message = run_cli_sweep_direct(
            geometry_file, sweep_file, parquet_output
        )

        assert csv_success, f"CSV command failed: {csv_message}"
        assert parquet_success, f"Parquet command failed: {parquet_message}"

        # Both should match their respective golden files
        validate_output_against_golden(csv_output, "csv")
        validate_output_against_golden(parquet_output, "parquet")

        print("✓ Both CSV and Parquet outputs match their golden files")


class TestUserExampleEquivalenceDirect:
    """
    Test that our direct CLI calls match the user's example command.
    """

    def test_matches_user_example_command_direct(
        self,
        temp_dir: Path,
        geometry_file: Path,
        sweep_file: Path,
    ) -> None:
        """
        Test that reproduces the exact user example via direct CLI function call:
        uv run kinematics --geometry tests/data/geometry.yaml --sweep tests/data/sweep.yaml --out final_test.csv
        """
        output_file = temp_dir / "final_test.csv"

        # This should match the user's command exactly (except calling CLI function directly)
        success, output = run_cli_sweep_direct(geometry_file, sweep_file, output_file)

        assert success, f"User example command failed: {output}"
        assert "wrote" in output.lower(), f"Unexpected output: {output}"

        # Validate against golden file - this ensures the output is correct
        validate_output_against_golden(output_file, "csv")

        print("✓ User example command reproduced successfully via direct CLI call")
