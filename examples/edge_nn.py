"""Appendix A: differentiable EdgeNN and an independent Torch aggregation."""
import torch
from torch import nn
import tiga as tg

torch.manual_seed(7)
device = "cuda" if torch.cuda.is_available() else "cpu"
positions = torch.rand(128, 3, device=device, requires_grad=True)
x = torch.randn(128, 8, device=device, requires_grad=True)
mlp = nn.Sequential(nn.Linear(11, 16), nn.ReLU(), nn.Linear(16, 4)).to(device)


class EdgeNN(tg.MessagePassing):
    reducer = tg.sum()

    def __init__(self, module):
        super().__init__()
        self.mlp = tg.nn.trace(module)

    def edge(self, src, dst, edge):
        return self.mlp(edge.displacement, src.x)


graph = tg.Graph.radius(positions, cutoff=0.35)
program = EdgeNN(mlp)
y = program(graph=graph, src={"x": x}, dst={})
assert y.shape == (128, 4)
loss = y.square().mean()
variables = (x, positions, *mlp.parameters())
grads = torch.autograd.grad(loss, variables)

# Independent gather -> MLP -> index_add reference on the same relation.
row_ptr, col_idx = graph.resolve_csr()
rows = torch.repeat_interleave(
    torch.arange(128, device=device), row_ptr[1:] - row_ptr[:-1])
delta = positions[col_idx] - positions[rows]
messages = mlp(torch.cat((delta, x[col_idx]), dim=-1))
reference = torch.zeros_like(y).index_add(0, rows, messages)
reference_grads = torch.autograd.grad(reference.square().mean(), variables)
torch.testing.assert_close(y, reference, rtol=2e-4, atol=2e-5)
for actual, expected in zip(grads, reference_grads):
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)
print(f"EdgeNN forward and all gradients checked on {device}.")
