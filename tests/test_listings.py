"""Keep every displayed code block explicitly typed and self-contained."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ListingStyle(unittest.TestCase):
    def test_all_blocks_declare_language(self):
        found = []
        for path in (ROOT / 'sections').glob('*.tex'):
            for match in re.finditer(r'\\begin\{lstlisting\}(\[[^\]]*\])?', path.read_text()):
                self.assertIsNotNone(match.group(1), str(path))
                self.assertIn('language=', match.group(1))
                found.append(match.group(1))
        self.assertEqual(len(found), 5)
        appendix = (ROOT/'sections/appendix-edge-nn.tex').read_text()
        listings = re.findall(r'\\lstinputlisting\[([^\]]*)\]', appendix)
        self.assertEqual(len(listings), 2)
        self.assertTrue(all('language=Python' in options for options in listings))

    def test_style_needs_no_external_highlighter(self):
        main = (ROOT / 'main.tex').read_text()
        style = (ROOT / 'listings.tex').read_text()
        self.assertIn(r'\input{listings}', main)
        self.assertNotIn('minted', main)
        for name in ('TigaIR', 'TigaShell'):
            self.assertIn(r'\lstdefinelanguage{' + name + '}', style)
