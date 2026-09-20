"""Export a fixed-pair table from hash-verified historical Tiga measurements."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATCH = ("topology", "locality", "cache", "nodes", "edges", "features", "index_dtype", "dtype", "phase", "shape")


def export(project):
    archive = project / "benchmarks/evidence_snapshot"
    index = json.loads((archive / "index.json").read_text())
    hashes = {entry["path"]: entry["sha256"] for entry in index["files"]}
    manifest = json.loads((project / "benchmarks/evidence_manifest.json").read_text())
    rows, inputs = [], []
    for case in manifest["operations"]["weighted_aggregation"]["cases"]:
        relative = f"output/roofline/weighted_aggregation/{case}/roofline.json"
        raw = (archive / relative).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != hashes[relative]:
            raise ValueError(f"checksum mismatch: {relative}")
        inputs.append({"path": relative, "sha256": digest})
        data = json.loads(raw)["results"]
        candidates = [row for row in data if row["provider"] == "tiga.prepared_auto" and row["cache"] == "hot"]
        if not candidates:
            raise ValueError(f"missing candidates: {case}")
        for row in candidates:
            peers = [peer for peer in data if peer["provider"] == "torch.sparse.mm"
                     and all(peer.get(key) == row.get(key) for key in MATCH)]
            if len(peers) != 1 or row["nodes"] != 131072:
                raise ValueError(f"ambiguous or wrong-size comparison: {case}")
            candidate, baseline = row["milliseconds"], peers[0]["milliseconds"]
            rows.append((row["topology"], row["locality"], row["index_dtype"], row["features"], candidate, baseline, baseline / candidate))
    if len(rows) != 11:
        raise ValueError(f"selection changed: expected 11 rows, got {len(rows)}")
    lines = [r"\begin{tabular}{lllrrrr}", r"\toprule",
             r"Topology & Locality & Index & F & Tiga & Torch & Ratio \\", r"\midrule"]
    for topology, locality, index, features, candidate, baseline, ratio in rows:
        if not all(value.isalnum() for value in (topology, locality, index)):
            raise ValueError("unexpected LaTeX label")
        lines.append(f"{topology} & {locality} & {index} & {features} & {candidate:.3f} & {baseline:.3f} & {ratio:.2f}" + r"$\times$ \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    provenance = {"status": "historical, not remeasured for this report", "candidate": "tiga.prepared_auto", "baseline": "torch.sparse.mm", "cache": "hot", "inputs": inputs}
    return {"weighted-table.tex": "\n".join(lines) + "\n", "provenance.json": json.dumps(provenance, indent=2) + "\n"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    products = export(args.project.resolve())
    for name, content in products.items():
        path = ROOT / "generated" / name
        if args.check:
            if not path.exists() or path.read_text() != content:
                raise SystemExit(f"stale: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    print("Verified 11 historical comparisons and their source hashes.")


if __name__ == "__main__":
    main()
