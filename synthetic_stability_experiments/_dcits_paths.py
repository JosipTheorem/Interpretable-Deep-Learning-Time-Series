"""Locate the sibling DCIts checkout without modifying it."""

import sys
from pathlib import Path


def configure_dcits_imports(script_file):
    """Make local experiment support and sibling DCIts source importable.

    The support directory is placed ahead of DCIts so ``src.utils`` resolves to
    this repository's experiment-specific utility module, while ``src.dcits``
    continues to resolve from the upstream DCIts checkout.
    """

    experiment_root = Path(script_file).resolve().parent
    repository_root = experiment_root.parent
    support_root = repository_root / "dcits_support"

    if not (support_root / "src" / "utils.py").is_file():
        raise FileNotFoundError(
            f"Could not find local DCIts utility support at {support_root}."
        )

    candidates = [
        repository_root.parent / "DCIts",
        Path.cwd() / "DCIts",
        Path.cwd().parent / "DCIts",
        Path.cwd(),
    ]
    dcits_root = next(
        (path.resolve() for path in candidates if (path / "src" / "dcits.py").is_file()),
        None,
    )

    if dcits_root is None:
        raise FileNotFoundError(
            "Could not find a DCIts checkout. Expected a sibling layout like "
            "workspace/DCIts and workspace/Interpretable-Deep-Learning-Time-Series."
        )

    for path in (dcits_root, support_root):
        path_string = str(path)
        if path_string in sys.path:
            sys.path.remove(path_string)
        sys.path.insert(0, path_string)

    return repository_root, dcits_root
