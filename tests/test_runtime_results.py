"""Protect the topology, timing samples and native-budget claims in the report."""
import copy
import unittest

from tools.build_runtime_results import read, validate


class RuntimeValidation(unittest.TestCase):
    def setUp(self):
        self.ranks, self.gpu = copy.deepcopy(read())

    def test_recorded_results(self):
        validate(self.ranks, self.gpu)

    def test_reject_different_sources(self):
        self.ranks[1]['source_hashes'] = {}
        with self.assertRaises(AssertionError):
            validate(self.ranks, self.gpu)

    def test_reject_missing_sample(self):
        self.ranks[0]['rows'][0]['samples_ms'].pop()
        with self.assertRaises(AssertionError):
            validate(self.ranks, self.gpu)

    def test_reject_wrong_budget(self):
        self.gpu['budget_mib'] = 32
        with self.assertRaises(AssertionError):
            validate(self.ranks, self.gpu)

    def test_reject_failed_paging(self):
        row = next(r for r in self.gpu['rows'] if r['mode'] == 'paged')
        row['status'] = 'budget_exceeded'
        with self.assertRaises(AssertionError):
            validate(self.ranks, self.gpu)

    def test_reject_repeated_step(self):
        self.gpu['rows'][0]['steps'][1]['step'] = 0
        with self.assertRaises(AssertionError):
            validate(self.ranks, self.gpu)
