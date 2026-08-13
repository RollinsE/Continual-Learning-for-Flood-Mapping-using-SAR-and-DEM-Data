import ast
import re
from pathlib import Path

import yaml

from floods.pipeline_audit import audit_code_quality
from floods.version import __version__


ROOT = Path(__file__).parents[1]


def test_release_versions_are_consistent():
    assert __version__ == "0.15.20"
    assert 'version="0.15.20"' in (ROOT / "setup.py").read_text(encoding="utf-8")
    assert "**Release 0.15.20**" in (ROOT / "README.md").read_text(encoding="utf-8")
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert citation["version"] == "0.15.20"
    assert citation["license"] == "MIT"


def test_public_python_definitions_follow_standard_naming():
    invalid = []
    for base in (ROOT / "floods", ROOT / "streamlit_app", ROOT / "scripts"):
        for path in base.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not (node.name.startswith("__") or re.fullmatch(r"_?[a-z][a-z0-9_]*", node.name)):
                        invalid.append((path.relative_to(ROOT), node.lineno, node.name))
                elif isinstance(node, ast.ClassDef):
                    if not re.fullmatch(r"_?[A-Z][A-Za-z0-9]*", node.name):
                        invalid.append((path.relative_to(ROOT), node.lineno, node.name))
    assert invalid == []


def test_release_code_quality_audit_has_no_findings(tmp_path):
    findings = audit_code_quality(ROOT, tmp_path)
    assert findings.empty


def test_runtime_validation_does_not_use_optimisation_sensitive_asserts():
    remaining = []
    for path in (ROOT / "floods").rglob("*.py"):
        if path.name == "environment.py":
            # The environment doctor intentionally uses assertions as probe checks;
            # they are caught and reported rather than used for runtime validation.
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                remaining.append((path.relative_to(ROOT), node.lineno))
    assert remaining == []


def test_release_does_not_use_unsafe_yaml_loader() -> None:
    roots = [ROOT / "floods", ROOT / "scripts", ROOT / "streamlit_app"]
    offenders = []
    for root in roots:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "yaml.UnsafeLoader" in text or "yaml.unsafe_load" in text:
                offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"Unsafe YAML loaders found: {offenders}"


def test_public_software_identity_is_neutral():
    setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")
    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
    cli_text = (ROOT / "floods" / "cli.py").read_text(encoding="utf-8")
    app_text = (ROOT / "streamlit_app" / "app.py").read_text(encoding="utf-8")
    init_text = (ROOT / "floods" / "__init__.py").read_text(encoding="utf-8")
    environment_text = (ROOT / "floods" / "environment.py").read_text(encoding="utf-8")
    assert 'name="flood-extent-mapping"' in setup_text
    assert '"floodmap=floods.cli:main"' in setup_text
    assert '"mmflood=floods.cli:main"' not in setup_text
    assert '# Flood Extent Mapping' in readme_text
    assert 'prog="floodmap"' in cli_text
    assert 'page_title="Flood Extent Mapping"' in app_text
    assert 'st.title("Flood Extent Mapping")' in app_text
    assert 'MMFlood Flood Mapping' not in app_text
    assert 'MMFlood command-line' not in init_text
    assert 'MMFlood environment:' not in environment_text
