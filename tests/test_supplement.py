"""Check archived evidence and the boundaries used in the report."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/f'tools/{name}.py')
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Supplement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder=module('build_supplement')
        cls.records=cls.builder.validate()
        cls.summary=cls.builder.summarize(cls.records)

    def test_full_matrix_validates(self):
        self.assertEqual(len(self.records),54)

    def test_saved_summary_matches_records(self):
        saved=json.loads((ROOT/'generated/supplement-summary.json').read_text())['records']
        self.assertEqual(len(saved),len(self.summary))
        for row in saved:
            key=tuple(row[k] for k in ('workload','provider','nodes','width','phase'))
            self.assertEqual({k:v for k,v in row.items() if k not in
                              ('workload','provider','nodes','width','phase')},self.summary[key])

    def test_table_values_come_from_records(self):
        table=(ROOT/'generated/supplement-edgenn.tex').read_text()
        for f in (8,32):
            for p in ('tiga','eager'):
                row=self.summary[('edgenn',p,35947,f,'warm')]
                for key in ('forward_ms','backward_ms','total_ms','memory_mib'):
                    self.assertIn(f"{row[key]['median']:.2f}",table)

    def test_negative_findings_are_not_omitted(self):
        iga=self.summary[('edgenn','tiga',35947,32,'warm')]
        peer=self.summary[('edgenn','eager',35947,32,'warm')]
        self.assertGreater(iga['total_ms']['median'],peer['total_ms']['median'])
        self.assertLess(iga['memory_mib']['median'],peer['memory_mib']['median'])
        text=(ROOT/'sections/evaluation-supplement.tex').read_text()
        for phrase in ('total slower than Torch','not a single-pass on/off ablation',
                       'not downstream task accuracy','not a measured cumulative-time curve',
                       'first timed call, not cold library initialization'):
            self.assertIn(phrase,text)

    def test_page_size_records_and_summary(self):
        records=module('build_page_sensitivity').validate()
        self.assertEqual(len(records),9)
        table=(ROOT/'generated/page-sensitivity.tex').read_text()
        for row in json.loads((ROOT/'generated/page-sensitivity.json').read_text()):
            self.assertIn(f"{row['forward_s']:.2f}",table)
            self.assertIn(f"{row['peak_mib']:.1f}",table)


if __name__=='__main__':
    unittest.main()
