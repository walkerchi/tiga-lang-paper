"""Optional runtime checks; manuscript-only builds do not require Torch/Tiga."""
import importlib.util
from pathlib import Path
import unittest


AVAILABLE = all(importlib.util.find_spec(name) is not None
                for name in ("torch", "tiga"))


@unittest.skipUnless(AVAILABLE, "Requires Torch and the Tiga source/package")
class AttentionExample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / "examples/causal_attention.py"
        spec = importlib.util.spec_from_file_location("paper_attention", source)
        cls.example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.example)

    def test_cpu_forward_and_semantic_gradients(self):
        for nodes in (1, 7, 65):
            with self.subTest(nodes=nodes):
                self.example.check("cpu", nodes)

    def test_cuda_compiled_forward_and_semantic_gradients(self):
        if not self.example.torch.cuda.is_available():
            self.skipTest("Requires CUDA")
        for nodes in (1, 7, 65, 128):
            with self.subTest(nodes=nodes):
                self.example.check("cuda", nodes)


if __name__ == "__main__":
    unittest.main()
