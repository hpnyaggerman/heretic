# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import sys
import unittest
from unittest.mock import patch

from heretic.config import Settings


class CommandLineTests(unittest.TestCase):
    def test_settings_ignore_the_process_arguments(self) -> None:
        with patch.object(sys, "argv", ["analyze.py", "--run", "x"]):
            settings = Settings(model="current")
            rebuilt = Settings.model_validate_json(settings.model_dump_json())
            restored = Settings.model_validate_init_only({"model": "stored"}, settings)

        self.assertEqual(settings.model, "current")
        self.assertEqual(rebuilt.model, "current")
        self.assertEqual(restored.model, "stored")

    def test_from_command_line_reads_the_process_arguments(self) -> None:
        with patch.object(sys, "argv", ["heretic", "--save-directory", "out"]):
            settings = Settings.from_command_line(model="current")

        self.assertEqual(settings.save_directory, "out")


if __name__ == "__main__":
    unittest.main()
