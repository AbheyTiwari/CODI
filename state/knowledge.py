"""Structured, run-scoped knowledge collected before and during execution."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FunctionKnowledge:
    name: str
    line: int
    args: list[str] = field(default_factory=list)
    returns: str | None = None
    decorators: list[str] = field(default_factory=list)
    docstring: str | None = None
    is_async: bool = False


@dataclass
class ClassKnowledge:
    name: str
    line: int
    bases: list[str] = field(default_factory=list)
    methods: list[FunctionKnowledge] = field(default_factory=list)
    docstring: str | None = None


@dataclass
class FileKnowledge:
    path: str
    language: str | None = None
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    functions: list[FunctionKnowledge] = field(default_factory=list)
    classes: list[ClassKnowledge] = field(default_factory=list)
    constants: list[str] = field(default_factory=list)
    globals: list[str] = field(default_factory=list)
    entrypoint: bool = False
    summary: str = ""
    source_read: bool = False


@dataclass
class KnowledgeBase:
    """Evidence, not hidden chain-of-thought, for one CODI run."""
    project: dict[str, Any] = field(default_factory=dict)
    files: dict[str, FileKnowledge] = field(default_factory=dict)
    search_results: dict[str, Any] = field(default_factory=dict)
    summaries: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    dependency_graph: dict[str, set[str]] = field(default_factory=dict)
    evidence: list[dict[str, str]] = field(default_factory=list)

    def record_inspection(self, payload: dict[str, Any]) -> None:
        path = str(payload.get("file") or payload.get("path") or "")
        if not path:
            return
        functions = [FunctionKnowledge(**item) for item in payload.get("functions", [])]
        classes = []
        for item in payload.get("classes", []):
            data = dict(item)
            data["methods"] = [FunctionKnowledge(**method) for method in data.get("methods", [])]
            classes.append(ClassKnowledge(**data))
        info = FileKnowledge(path=path, language=payload.get("language"), imports=payload.get("imports", []), exports=payload.get("exports", []), functions=functions, classes=classes, constants=payload.get("constants", []), globals=payload.get("globals", []), entrypoint=bool(payload.get("entrypoint")), summary=payload.get("summary", ""))
        self.files[path] = info
        self.dependency_graph[path] = set(info.imports)
        self.evidence.append({"kind": "inspect_file", "subject": path})

    def record_tool_output(self, tool: str, args: dict[str, Any], payload: Any) -> None:
        if tool == "inspect_project" and isinstance(payload, dict):
            self.project = payload
            self.evidence.append({"kind": "inspect_project", "subject": "project"})
        elif tool == "inspect_file" and isinstance(payload, dict):
            self.record_inspection(payload)
        elif tool == "search_codebase":
            query = str(args.get("query", "")); self.search_results[query] = payload; self.evidence.append({"kind": "search", "subject": query})
        elif tool in {"read_file", "read_file_numbered"}:
            path = str(args.get("path", ""))
            if path in self.files: self.files[path].source_read = True
            self.evidence.append({"kind": "read_file", "subject": path})

    def add_unknown(self, value: str) -> None:
        if value and value not in self.unknowns: self.unknowns.append(value)

    def summary_for_prompt(self, max_chars: int = 14000) -> str:
        lines = ["PROJECT KNOWLEDGE"]
        if self.project: lines.append(f"languages={self.project.get('languages', [])}; frameworks={self.project.get('frameworks', [])}")
        for path, info in self.files.items(): lines.append(f"{path}: imports={info.imports}; symbols={[f.name for f in info.functions] + [c.name for c in info.classes]}")
        if self.unknowns: lines.append("unknowns=" + "; ".join(self.unknowns))
        if self.risks: lines.append("risks=" + "; ".join(self.risks))
        return "\n".join(lines)[:max_chars]

    def plan_context(self) -> dict[str, Any]:
        return {"project": self.project, "files_inspected": list(self.files), "unknowns": self.unknowns, "risks": self.risks, "dependency_graph": {key: sorted(value) for key, value in self.dependency_graph.items()}, "evidence": self.evidence}
