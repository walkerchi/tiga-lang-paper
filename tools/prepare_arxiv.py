"""Build a deterministic, minimal tech-report source archive, never submit it."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def dependencies(root=ROOT, entry='main.tex'):
    root = Path(root).resolve()
    pending = [entry]
    found = set()
    while pending:
        name = pending.pop()
        path = root / name
        if not path.suffix:
            path = path.with_suffix('.tex')
        relative = path.resolve().relative_to(root).as_posix()
        if relative in found:
            continue
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Missing or symlinked dependency: {relative}')
        if not all(re.fullmatch(r'[A-Za-z0-9_+.,=\-]+', p) for p in Path(relative).parts):
            raise ValueError(f'Unsupported arXiv filename: {relative}')
        found.add(relative)
        if path.suffix != '.tex':
            continue
        text = re.sub(r'(?<!\\)%[^\n]*', '', path.read_text())
        pending.extend(re.findall(r'\\(?:input|include)\{([^}]+)\}', text))
        pending.extend(re.findall(
            r'\\(?:includegraphics|lstinputlisting)(?:\[[^\]]*\])?\{([^}]+)\}', text))
        for group in re.findall(r'\\bibliography\{([^}]+)\}', text):
            pending.extend(name.strip()+'.bib' for name in group.split(','))
    if any(p.startswith(('workshop/', 'styles/', 'sections/archive/', 'build/')) for p in found):
        raise ValueError('Unrelated manuscript or build files entered the source closure')
    return sorted(found)


def build(output, root=ROOT):
    root, output = Path(root), Path(output)
    names = dependencies(root)
    output.mkdir(parents=True, exist_ok=True)
    target = output/'tiga-lang-source.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            item = zipfile.ZipInfo(name, date_time=(2026,9,21,0,0,0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = 0o100644 << 16
            archive.writestr(item,(root/name).read_bytes())
    manifest = dict(entry='main.tex', files={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                                           for name in names},
                    archive_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'{target}: {len(names)} files, {target.stat().st_size} bytes')
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'build/arxiv')
    args = parser.parse_args()
    build(args.output)
