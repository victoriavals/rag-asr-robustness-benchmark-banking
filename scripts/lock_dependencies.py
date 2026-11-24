"""Freeze the tested Python 3.12 dependency graph from installed metadata, offline.

Run after deliberately upgrading the direct dependencies and running regression tests.
No packages are installed and no network requests are made by this script.
"""
from collections import defaultdict, deque
from importlib.metadata import distribution
from pathlib import Path
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ['streamlit', 'qdrant-client', 'jiwer', 'pydantic',
           'pydantic-settings', 'python-dotenv', 'soundfile', 'datasets',
           'numpy', 'pandas', 'plotly', 'httpx']


def freeze(roots):
    pending = deque(roots)
    versions, markers = {}, defaultdict(set)
    for name in roots:
        markers[canonicalize_name(name)].add('')
    while pending:
        name = canonicalize_name(pending.popleft())
        if name in versions:
            continue
        dist = distribution(name)
        versions[name] = dist.version
        for text in dist.requires or []:
            requirement = Requirement(text)
            if requirement.marker and not requirement.marker.evaluate({'extra': ''}):
                continue
            child = canonicalize_name(requirement.name)
            installed = distribution(child).version
            if requirement.specifier and installed not in requirement.specifier:
                raise RuntimeError(f'Tested environment does not satisfy {requirement.name}{requirement.specifier}')
            markers[child].add(str(requirement.marker) if requirement.marker else '')
            pending.append(child)
    lines = []
    for name, version in sorted(versions.items()):
        conditions = markers[name]
        marker = '' if '' in conditions else ' or '.join(f'({part})' for part in sorted(conditions))
        lines.append(f'{name}=={version}' + (f'; {marker}' if marker else ''))
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    header = '# Tested Python 3.12 environment. Generated offline by scripts/lock_dependencies.py.\n'
    runtime = freeze(RUNTIME)
    full = freeze(RUNTIME + ['pytest'])
    (ROOT / 'requirements.txt').write_text(header + '# Direct runtime dependencies: ' + ', '.join(RUNTIME) + '\n' + runtime, encoding='utf-8')
    # A single constraints graph for development also locks pytest transitive dependencies.
    (ROOT / 'requirements-dev.txt').write_text(header + '# Runtime plus offline regression test dependencies.\n' + full, encoding='utf-8')
