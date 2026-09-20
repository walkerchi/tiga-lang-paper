"""Reject missing, duplicated or semantically mismatched paper measurements."""
import copy
import json
from pathlib import Path
import unittest

from tools.build_results import validate

DATA = Path(__file__).resolve().parents[1] / 'data'


class ResultsValidation(unittest.TestCase):
    def setUp(self):
        self.gpu = json.loads((DATA / 'gpu-scaling.json').read_text())
        self.memory = json.loads((DATA / 'memory-results.json').read_text())

    def test_archived_data(self):
        validate(self.gpu, self.memory)

    def test_reject_wrong_feature_width(self):
        self.gpu['rows'][0]['features'] = 128
        with self.assertRaises(AssertionError):
            validate(self.gpu, self.memory)

    def test_reject_duplicate_trial(self):
        self.memory['rows'][0] = copy.deepcopy(self.memory['rows'][1])
        with self.assertRaises(AssertionError):
            validate(self.gpu, self.memory)

    def test_reject_budget_mismatch(self):
        self.memory['rows'][0]['budget_bytes'] *= 2
        with self.assertRaises(AssertionError):
            validate(self.gpu, self.memory)

    def test_reject_changed_timing_scope(self):
        self.memory['rows'][0]['total_s'] += 1
        with self.assertRaises(AssertionError):
            validate(self.gpu, self.memory)

    def test_reject_missing_gpu_sample(self):
        self.gpu['rows'][0]['samples_ms'].pop()
        with self.assertRaises(AssertionError):
            validate(self.gpu, self.memory)
