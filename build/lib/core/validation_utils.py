import ast
import re


def build_framework_contamination_errors(content: str, requirements, path: str | None = None) -> list[str]:
    """Return deterministic framework-contamination issues for a file body."""
    if not content or not requirements:
        return []

    forbidden_patterns = []
    try:
        forbidden_patterns = list(requirements.framework_lock())
    except Exception:
        forbidden_patterns = []

    if not forbidden_patterns and not getattr(requirements, "must_not", []):
        return []

    framework = getattr(requirements, "framework", None) or "the target framework"
    errors: list[str] = []

    for pattern in forbidden_patterns:
        if re.search(re.escape(pattern), content, re.IGNORECASE):
            errors.append(
                f"FORBIDDEN: '{pattern}' found in {path or 'generated content'}. "
                f"This is a {framework} project. Remove ALL {pattern} references and rewrite using "
                f"{framework} only."
            )

    for constraint in getattr(requirements, "must_not", []) or []:
        search_term = constraint.lower().split()[0]
        if len(search_term) > 3 and re.search(search_term, content, re.IGNORECASE):
            errors.append(
                f"must_not violated: '{constraint}' detected in {path or 'generated content'}."
            )

    if path and path.endswith(".py") and content.strip():
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    module = ""
                    if isinstance(node, ast.ImportFrom) and node.module:
                        module = node.module.lower()
                    elif isinstance(node, ast.Import):
                        module = " ".join(a.name.lower() for a in node.names)

                    for pattern in forbidden_patterns:
                        pat_module = pattern.replace("from ", "").replace("import ", "").strip().lower()
                        if pat_module and pat_module in module:
                            errors.append(
                                f"AST check: forbidden import '{module}' in {path}. "
                                f"Replace with {framework} equivalent."
                            )
        except SyntaxError:
            pass

    return errors
