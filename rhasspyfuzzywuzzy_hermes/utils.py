"""Utility methods for rhasspy-fuzzywuzzy-hermes"""
import io
import json
import logging
import subprocess
import typing
from pathlib import Path

_LOGGER = logging.getLogger("rhasspyfuzzywuzzy_hermes")


# -----------------------------------------------------------------------------


class CliConverter:
    """Command-line converter for intent recognition"""

    def __init__(self, name: str, command_path: Path):
        self.name = name
        self.command_path = command_path

    def __call__(self, *args, converter_args=None):
        """Runs external program to convert JSON values"""
        converter_args = converter_args or []
        proc = subprocess.Popen(
            [str(self.command_path)] + converter_args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )

        with io.StringIO() as input_file:
            for arg in args:
                json.dump(arg, input_file)

            stdout, _ = proc.communicate(input=input_file.getvalue())

            return [json.loads(line) for line in stdout.splitlines() if line.strip()]


def load_converters(converters_dir: Path,) -> typing.Dict[str, typing.Any]:
    """Load user-defined converters from a directory"""
    converters = {}

    if converters_dir.is_dir():
        _LOGGER.debug("Loading converters from %s", converters_dir)
        for converter_path in converters_dir.glob("**/*"):
            if not converter_path.is_file():
                continue

            # Retain directory structure in name
            converter_name = str(
                converter_path.relative_to(converters_dir).with_suffix("")
            )

            # Run converter as external program.
            # Input arguments are encoded as JSON on individual lines.
            # Output values should be encoded as JSON on individual lines.
            converter = CliConverter(converter_name, converter_path)

            # Key off name without file extension
            converters[converter_name] = converter

            _LOGGER.debug("Loaded converter %s from %s", converter_name, converter_path)

    return converters
