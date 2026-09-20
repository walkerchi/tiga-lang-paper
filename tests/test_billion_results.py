import copy
import json
from pathlib import Path
import unittest

from tools.build_billion_results import validate


class BillionValidation(unittest.TestCase):
    def setUp(self):
        self.rows = json.loads((Path(__file__).resolve().parents[1]/'data/billion/results.json').read_text())

    def test_results(self):
        validate(self.rows)

    def test_incomplete_trial(self):
        self.rows[-1]['status'] = 'interrupted'
        with self.assertRaises(AssertionError):
            validate(self.rows)

    def test_partial_oracle(self):
        self.rows[-1]['checked_values'] -= 32
        with self.assertRaises(AssertionError):
            validate(self.rows)

    def test_different_implementation(self):
        self.rows[-1]['provenance']['benchmark_sha256'] = 'wrong'
        with self.assertRaises(AssertionError):
            validate(self.rows)

    def test_duplicate_trial(self):
        self.rows[-1] = copy.deepcopy(self.rows[-2])
        with self.assertRaises(AssertionError):
            validate(self.rows)

    def test_incorrect_capacity_arithmetic(self):
        self.rows[-1]['resident_lower_bound_bytes'] //= 2
        with self.assertRaises(AssertionError):
            validate(self.rows)

    def test_changed_fixture(self):
        self.rows[-1]['ingest']['payload_sha256']['x.bin'] = 'wrong'
        with self.assertRaises(AssertionError):
            validate(self.rows)
