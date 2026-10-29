"""
Universal Ingestion Engine for Bonsai.
Converts raw files, logs, documents, and plain-English goals into structured context blocks.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Union

class DocumentBlock:
    def __init__(self, source: str, content: str, metadata: Dict[str, Any] = None):
        self.source = source
        self.content = content.strip()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "content": self.content,
            "metadata": self.metadata
        }

def ingest_source(source_input: str) -> List[DocumentBlock]:
    """
    Ingests either a directory, a single file, or a direct string prompt.
    Returns a list of clean DocumentBlock objects.
    """
    path = Path(source_input)
    blocks: List[DocumentBlock] = []

    if path.exists():
        if path.is_file():
            _read_file(path, blocks)
        elif path.is_dir():
            for ext in ("*.txt", "*.md", "*.log", "*.json", "*.csv", "*.jsonl"):
                for file_path in path.rglob(ext):
                    if ".venv" in file_path.parts or "node_modules" in file_path.parts or ".git" in file_path.parts:
                        continue
                    _read_file(file_path, blocks)
    else:
        # Source input is a direct task goal or prompt
        blocks.append(DocumentBlock(
            source="user_goal",
            content=source_input,
            metadata={"type": "goal_prompt"}
        ))

    if not blocks:
        raise ValueError(f"No valid data or text found in source: {source_input}")

    return blocks

def _read_file(file_path: Path, blocks: List[DocumentBlock]):
    """Reads a single file safely and creates one or more DocumentBlocks."""
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            return

        # If it's a JSON file with a list or object
        if file_path.suffix.lower() == ".json":
            try:
                data = json.loads(text)
                if isinstance(data, list):
                    for idx, item in enumerate(data):
                        blocks.append(DocumentBlock(
                            source=f"{file_path.name}#{idx}",
                            content=json.dumps(item, indent=2) if isinstance(item, (dict, list)) else str(item),
                            metadata={"file": str(file_path), "index": idx}
                        ))
                    return
            except Exception:
                pass

        # If it's a JSONL file
        if file_path.suffix.lower() == ".jsonl":
            for line_no, line in enumerate(text.splitlines()):
                line = line.strip()
                if line:
                    blocks.append(DocumentBlock(
                        source=f"{file_path.name}:{line_no+1}",
                        content=line,
                        metadata={"file": str(file_path), "line": line_no+1}
                    ))
            return

        # For large markdown/text files with distinct section breaks (e.g. --- or ##)
        if "---" in text:
            parts = [p.strip() for p in text.split("---") if p.strip()]
            for idx, p in enumerate(parts):
                blocks.append(DocumentBlock(
                    source=f"{file_path.name}#section-{idx+1}",
                    content=p,
                    metadata={"file": str(file_path), "section": idx+1}
                ))
            return

        # Default: whole file as one block
        blocks.append(DocumentBlock(
            source=file_path.name,
            content=text,
            metadata={"file": str(file_path)}
        ))
    except Exception as e:
        print(f"Warning: Failed to read {file_path}: {e}")
