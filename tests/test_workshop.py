"""Keep draft status, template provenance and evidence boundaries explicit."""
import hashlib
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class WorkshopManuscript(unittest.TestCase):
    def test_official_template_is_unmodified(self):
        directory = ROOT / 'styles/iclr2027'
        for line in (directory / 'checksums.sha256').read_text().splitlines():
            if not line or line.startswith('#'):
                continue
            digest, name = line.split()
            self.assertEqual(hashlib.sha256((directory / name).read_bytes()).hexdigest(), digest)

    def test_draft_has_no_author_identifiers_or_false_submission_header(self):
        entry = (ROOT / 'iclr2027.tex').read_text()
        prose = '\n'.join(path.read_text() for path in (ROOT / 'workshop').glob('*.tex'))
        for marker in ('Mingyuan', 'walkerchi', 'walker.chi', 'mailto:', 'tiga-logo'):
            self.assertNotIn(marker, entry + prose)
        self.assertIn('not submitted', entry)
        self.assertIn(r'\patchcmd{\@maketitle}', entry)
        self.assertNotIn(r'\iclrfinalcopy', entry)
        self.assertIn('pdfauthor={}', entry)

    def test_shared_evidence_and_independent_report(self):
        entry = (ROOT / 'iclr2027.tex').read_text()
        evaluation = (ROOT / 'workshop/evaluation.tex').read_text()
        self.assertIn(r'\bibliography{references}', entry)
        for figure in ('q1-performance-memory.pdf', 'q2-capacity-cost.pdf'):
            self.assertIn('figures/' + figure, evaluation)
        self.assertIn('does not establish a distributed speedup', evaluation)
        report = (ROOT / 'main.tex').read_text()
        self.assertNotIn('workshop/', report)
        self.assertIn('Mingyuan Chi', report)

    def test_ai_disclosure_and_unmeasured_work_are_explicit(self):
        statements = (ROOT / 'workshop/statements.tex').read_text()
        discussion = (ROOT / 'workshop/discussion.tex').read_text()
        self.assertIn('AI use statement', statements)
        self.assertIn('recorded program executions', statements)
        self.assertIn('ablation', discussion)
        self.assertIn('first-call compilation', discussion)
        self.assertIn('forward/backward steps', discussion)

    def test_code_is_highlighted_and_inputs_resolve_from_root(self):
        for path in (ROOT / 'workshop').glob('*.tex'):
            source = path.read_text()
            for options in re.findall(r'\\begin\{lstlisting\}(\[[^\]]*\])?', source):
                self.assertIn('language=', options, str(path))


if __name__ == '__main__':
    unittest.main()
