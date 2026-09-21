"""Check citation and cross-reference integrity in the active manuscript."""
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


def active_sources():
    pending = [ROOT / 'main.tex']
    seen = set()
    result = []
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        text = path.read_text()
        result.append(text)
        for name in re.findall(r'\\input\{([^}]+)\}', text):
            source = ROOT / name
            if not source.suffix:
                source = source.with_suffix('.tex')
            pending.append(source)
    return '\n'.join(result)


class ManuscriptLinks(unittest.TestCase):
    def test_citations_resolve_to_unique_bibliography_entries(self):
        bibliography = (ROOT / 'references.bib').read_text()
        keys = re.findall(r'^@\w+\{([^,]+),', bibliography, re.MULTILINE)
        self.assertEqual(len(keys), len(set(keys)), 'Duplicate bibliography keys')
        citations = {
            key.strip()
            for group in re.findall(r'\\cite\{([^}]+)\}', active_sources())
            for key in group.split(',')
        }
        self.assertTrue(citations)
        self.assertFalse(citations - set(keys), citations - set(keys))

    def test_section_figure_and_table_references_resolve(self):
        source = active_sources()
        labels = re.findall(r'\\label\{([^}]+)\}', source)
        self.assertEqual(len(labels), len(set(labels)), 'Duplicate labels')
        references = set(re.findall(r'\\(?:ref|eqref|pageref)\{([^}]+)\}', source))
        self.assertFalse(references - set(labels), references - set(labels))

    def test_included_code_and_figures_exist(self):
        source = active_sources()
        assets = re.findall(
            r'\\(?:lstinputlisting|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}',
            source,
        )
        self.assertTrue(assets)
        for name in assets:
            with self.subTest(asset=name):
                self.assertTrue((ROOT / name).is_file(), name)


if __name__ == '__main__':
    unittest.main()
