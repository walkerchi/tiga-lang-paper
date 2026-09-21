"""Appendix B: compiled attention forward and semantic gradient checks."""
import argparse

import torch
import tiga as tg


class CausalAttention(tg.MessagePassing):
    reducer = tg.online_softmax()

    def edge(self, src, dst, edge, scale):
        score = (dst.query * src.key).sum(dim=-1) * scale
        return self.reducer(score, src.value)


def torch_attention(query, key, value):
    # Public shape (N, H, D); matrix multiplication uses (H, N, D).
    q, k, v = (x.transpose(0, 1) for x in (query, key, value))
    scores = (q @ k.transpose(-1, -2)) * q.shape[-1]**-0.5
    future = torch.ones(q.shape[1], q.shape[1], device=q.device,
                        dtype=torch.bool).triu(1)
    weights = scores.masked_fill(future, -torch.inf).softmax(-1)
    return (weights @ v).transpose(0, 1)


def check(device, nodes=65):
    torch.manual_seed(7)
    heads, width = 2, 64
    dtype = torch.float16 if device == "cuda" else torch.float32
    # Head-major storage, viewed as ordinary (N, H, D) Torch tensors.
    query, key, value = (
        torch.randn(heads, nodes, width, device=device, dtype=dtype)
        .permute(1, 0, 2) for _ in range(3))
    graph = tg.Graph.triangular(nodes, device=device)
    program = CausalAttention()
    with torch.no_grad():
        output = program(graph=graph, src={"key": key, "value": value},
                         dst={"query": query}, scale=width**-0.5)
    expected = torch_attention(query.float(), key.float(), value.float())
    torch.testing.assert_close(output.float(), expected, rtol=4e-3, atol=4e-3)
    explanation = program.explain()
    if device == "cuda":
        # Fail if a fallback would conceal a missing compiler/provider.
        assert "gf-kernel-to-ttir-dense-streaming-reduction" in explanation

    # Explicit semantic path: streaming CUDA backward is not implemented.
    variables = tuple(x.float().detach().requires_grad_()
                      for x in (query, key, value))
    q, k, v = variables
    semantic = program.reference(
        graph=graph, src={"key": k, "value": v},
        dst={"query": q}, scale=width**-0.5)
    reference_variables = tuple(x.detach().clone().requires_grad_()
                                for x in variables)
    reference = torch_attention(*reference_variables)
    cotangent = torch.randn_like(reference)
    gradients = torch.autograd.grad(semantic, variables, cotangent)
    expected_gradients = torch.autograd.grad(
        reference, reference_variables, cotangent)
    torch.testing.assert_close(semantic, reference, rtol=2e-4, atol=2e-5)
    for actual, expected in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)

    print(f"Causal attention checked on {device}: N={nodes}, H={heads}, D={width}.")
    print("Forward: " + ("compiled streaming" if device == "cuda" else "CPU semantic"))
    print("Q/K/V gradients: explicit Torch semantic path.")
    return explanation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--nodes", type=int, default=65)
    args = parser.parse_args()
    if args.nodes < 1:
        parser.error("--nodes must be positive")
    check(args.device, args.nodes)
