"""The two-host NCCL gate requires both ranks and numerical VJP evidence."""
import copy
import json
from pathlib import Path
import unittest

from tools.build_runtime_results import validate_nccl

DATA = Path(__file__).resolve().parents[1] / 'data/runtime'


class NCCLResults(unittest.TestCase):
    def setUp(self):
        self.ranks = [json.loads((DATA / f'nccl-graph-rank{i}.json').read_text())
                      for i in (0, 1)]

    def test_recorded_gate(self):
        validate_nccl(self.ranks)

    def test_reject_repeated_rank(self):
        self.ranks[1] = copy.deepcopy(self.ranks[0])
        with self.assertRaises(AssertionError):
            validate_nccl(self.ranks)

    def test_reject_incorrect_gradient_despite_pass_flag(self):
        self.ranks[1]['records'][2]['gradient'] = [3.] * 4
        with self.assertRaises(AssertionError):
            validate_nccl(self.ranks)

    def test_reject_host_time_as_gpu_overlap(self):
        self.ranks[1]['records'][1]['trace']['measured_overlap_ms'] = 1.
        with self.assertRaises(AssertionError):
            validate_nccl(self.ranks)
