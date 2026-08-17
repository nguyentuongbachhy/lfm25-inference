from __future__ import annotations

import re
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one match, found {count}: {old!r}")
    path.write_text(text.replace(old, new, 1))


def patch_runner() -> None:
    path = Path("src/engine/runner.rs")
    replace_once(
        path,
        "    generation::{Sampler, SamplingConfig},\n",
        "    generation::{DEFAULT_SAMPLING_SEED, Sampler, SamplingConfig},\n",
    )
    replace_once(path, "                seed: 0x4c_46_4d_32,\n", "                seed: DEFAULT_SAMPLING_SEED,\n")
    replace_once(
        path,
        "                &token_ids,\n                &mut collector,\n",
        "                token_ids,\n                &mut collector,\n",
    )
    replace_once(
        path,
        "    if trimmed.starts_with('\\\"') {\n        if let Ok(text) = serde_json::from_str::<String>(trimmed) {\n            return Ok(Some(text));\n        }\n    }\n",
        "    if trimmed.starts_with('\\\"')\n        && let Ok(text) = serde_json::from_str::<String>(trimmed)\n    {\n        return Ok(Some(text));\n    }\n",
    )


def remove_lint_suppression() -> None:
    pattern = re.compile(r"(?m)^[ \t]*#\[allow\([^\n]*\)\]\n")
    for path in Path("src").rglob("*.rs"):
        text = path.read_text()
        updated = pattern.sub("", text)
        if updated != text:
            path.write_text(updated)


patch_runner()
remove_lint_suppression()
