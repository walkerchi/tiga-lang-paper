"""Keep scientific descriptions separate from internal editing history."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReaderContract(unittest.TestCase):
    def test_active_manuscript_has_no_internal_experiment_retrospective(self):
        paths = [ROOT / "main.tex", ROOT / "README.md", ROOT / "generated/q3-findings.tex",
                 *(ROOT / "sections").glob("*.tex")]
        forbidden = ("Three experiments", "three application-facing experiments",
                     "Earlier overlap probes", "earlier 10M diagnostics",
                     "previous ring", "private WireGuard", "test GPUs",
                     "there is no duplicate Related work section",
                     "An earlier interior/boundary overlap experiment",
                     "End-to-end timings include runtime work and waits, not isolated")
        for path in paths:
            text = path.read_text()
            for phrase in forbidden:
                self.assertNotIn(phrase, text, (path.name, phrase))

    def test_radius_membership_matches_the_inclusive_api(self):
        text = (ROOT / "sections/appendix-edge-nn.tex").read_text()
        self.assertIn(r"\|p_j-p_i\|_2\le r", text)

    def test_programming_example_uses_a_section_reference(self):
        self.assertIn(r"\label{sec:programming-model}",
                      (ROOT / "sections/model.tex").read_text())
        text = (ROOT / "sections/differentiation.tex").read_text()
        self.assertIn(r"\ref{sec:programming-model}", text)
        self.assertNotIn("example in Section 2", text)

    def test_distributed_conditions_remain_explicit(self):
        text = (ROOT / "sections/evaluation-questions.tex").read_text()
        for phrase in ("Socket", "half the destination rows", "synchronized host wall",
                       "maximum of paired rank durations"):
            self.assertIn(phrase, text)


if __name__ == "__main__":
    unittest.main()
