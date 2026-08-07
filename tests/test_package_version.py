import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


class TestPackageVersion(unittest.TestCase):
    def test_source_tree_import_without_generated_version_file(self):
        package_dir = Path(__file__).resolve().parents[1] / 'pcdet'
        init_path = package_dir / '__init__.py'
        expected_commit = subprocess.check_output(
            ['git', 'rev-parse', '--short=7', 'HEAD'],
            cwd=package_dir.parent,
            text=True,
        ).strip()
        module_name = 'pcdet_version_fallback_test'
        spec = importlib.util.spec_from_file_location(
            module_name,
            init_path,
            submodule_search_locations=[str(package_dir)],
        )
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {module_name: module}):
            spec.loader.exec_module(module)

        self.assertEqual(module.__version__, '0.6.0+py%s' % expected_commit)
