import copy
import importlib.util
import json
from pathlib import Path
import unittest
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('three_questions',ROOT/'tools/build_three_questions.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ThreeQuestionEvidence(unittest.TestCase):
    def test_billion_profile_is_measured_and_fully_checked(self):
        import build_paging_profile as paging
        result = paging.summary()
        self.assertEqual(result['edges'], 1_000_000_000)
        self.assertAlmostEqual(sum(result['exclusive_host_seconds'].values()),
                               result['host_seconds'])
        self.assertLess(result['kernel_seconds'], result['host_seconds'])

    def test_edge_nn_appendix_and_title_logo(self):
        main = (ROOT/'main.tex').read_text()
        self.assertIn('\\input{sections/appendix-edge-nn}', main)
        self.assertNotIn('\\input{sections/appendix-paging}', main)
        self.assertIn('figures/tiga-logo.pdf', main)
        example = (ROOT/'examples/edge_nn.py').read_text()
        for term in ('reducer = tg.sum()', 'torch.autograd.grad',
                     'torch.testing.assert_close', 'tg.nn.trace'):
            self.assertIn(term, example)

    def test_archive_checksums_and_shapes(self):
        module.validate()

    def test_comparison_cannot_drop_a_slow_provider(self):
        rows=module.read(module.DATA/'comparison.json')
        with self.assertRaises(AssertionError):
            module.validate_comparison(rows[:-1])

    def test_comparison_rejects_mixed_source(self):
        rows=module.read(module.DATA/'comparison.json')
        rows[-1]['benchmark_sha256']='unmatched'
        with self.assertRaises(AssertionError):
            module.validate_comparison(rows)

    def test_rank_critical_path_is_paired_maximum(self):
        ranks=[{'rows':[{'nodes':32,'mode':'automatic','samples_ms':s}]} for s in ([1,9,1],[8,2,8])]
        self.assertEqual(module.paired_samples(ranks,32,'automatic'),[8,9,8])

    def test_every_automatic_sample_has_additive_host_stages(self):
        for topology in ('local','remote'):
            for rank in (0,1):
                data=module.read(module.DATA/f'{topology}-rank{rank}.json')
                for record in data['rows']:
                    if record['mode']=='automatic':
                        for step in range(5):
                            phases=module.host_phases(record,step)
                            self.assertAlmostEqual(sum(phases),record['samples_ms'][step])

    def test_manuscript_uses_three_main_experiment_figures(self):
        text=(ROOT/'sections/evaluation-questions.tex').read_text()
        self.assertEqual(text.count('\\begin{figure}'),3)
        for name in ('q1-performance-memory','q2-capacity-cost','q3-distributed'):
            self.assertIn(name,text)
        main=(ROOT/'main.tex').read_text()
        self.assertNotIn('\\input{sections/evaluation}',main)
        self.assertNotIn('\\input{sections/related}',main)

    def test_storage_hierarchy_is_in_runtime_not_evaluation(self):
        runtime=(ROOT/'sections/runtime.tex').read_text()
        self.assertIn('\\input{figures/storage-hierarchy}',runtime)
        diagram=(ROOT/'figures/storage-hierarchy.tex').read_text()
        for term in ('On-chip storage','GPU device memory','Host DRAM',
                     'NVMe-backed persistent store','Active page + full output'):
            self.assertIn(term,diagram)
        for term in ('test GPUs', '5070', '4070', '1B-edge run', 'GiB'):
            self.assertNotIn(term,diagram)
        self.assertIn('sec:experimental-setup', runtime)

    def test_distributed_heading_preserves_actual_two_rank_scope(self):
        text=(ROOT/'sections/evaluation-questions.tex').read_text()
        self.assertIn('capacity, and distributed execution}',text)
        self.assertIn('Distributed execution and communication overhead}',text)
        self.assertIn('two-rank graph',text)
        for marker in ('Q1','Q2','Q3'):
            self.assertNotIn(marker,text)

    def test_revised_comparison_checksums_and_shapes(self):
        module.validate_revised_comparison()

    def test_revised_comparison_cannot_omit_128k(self):
        rows=module.read(module.Q1_DATA/'comparison.json')
        with self.assertRaises(AssertionError):
            module.validate_comparison([r for r in rows if r['workload']!='knn' or r['nodes']!=131072])

    def test_spatial_interface_and_paired_measurements(self):
        import build_spatial_results as spatial
        rows=spatial.summarize(spatial.validate())
        self.assertEqual([r['side'] for r in rows],[32,64,96])
        for r in rows:
            self.assertAlmostEqual(sum(r['host_phases_ms']),r['automatic_ms'])
            self.assertEqual(r['sent_bytes'],2*r['side']**2*16*4)

    def test_spatial_summary_accepts_only_the_public_method(self):
        import build_spatial_results as spatial
        ranks=copy.deepcopy(spatial.validate())
        for rank in ranks:
            rank['rows']=[q for q in rank['rows'] if q['mode']!='automatic']
        rows=spatial.summarize(ranks)
        self.assertEqual(rows[-1]['serialized_ms'],44.222053)
        self.assertTrue(all(r['automatic_ms'] is None for r in rows))


if __name__=='__main__': unittest.main()
