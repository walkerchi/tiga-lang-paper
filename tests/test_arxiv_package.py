"""Ensure the independent tech-report upload has only required source files."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('arxiv_package',ROOT/'tools/prepare_arxiv.py')
package=importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


class ArxivPackage(unittest.TestCase):
    def test_only_tech_report_dependencies(self):
        files=package.dependencies()
        for name in ('main.tex','references.bib','sections/statements.tex',
                     'examples/edge_nn.py','examples/causal_attention.py',
                     'generated/supplement-edgenn.tex','figures/supplement-execution.pdf'):
            self.assertIn(name,files)
        for name in files:
            self.assertFalse(name.startswith(('workshop/','styles/','data/','sections/archive/','build/')))
            self.assertNotIn(name,('iclr2027.tex','tiga-lang.pdf','tiga-lang-iclr2027.pdf'))

    def test_deterministic_archive_matches_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            target=package.build(directory)
            first=target.read_bytes()
            self.assertEqual(first,package.build(directory).read_bytes())
            with zipfile.ZipFile(target) as archive:
                self.assertEqual(archive.namelist(),package.dependencies())
                for name in archive.namelist():
                    self.assertEqual(archive.read(name),(ROOT/name).read_bytes())

    def test_disclosure_is_active_not_only_in_workshop(self):
        self.assertIn(r'\input{sections/statements}',(ROOT/'main.tex').read_text())
        statement=(ROOT/'sections/statements.tex').read_text()
        self.assertIn('AI coding and writing assistants',statement)
        self.assertIn('not generated',statement)
        self.assertIn('human author',statement)


if __name__=='__main__':
    unittest.main()
