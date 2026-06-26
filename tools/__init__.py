# tools/__init__.py

from dataclasses import dataclass


@dataclass
class ToolInfo:
    name: str
    description: str = ""


def get_all_tools():
    """Return loaded tools for the CLI /tools command."""
    from config import MODE
    from tools.registry import registry

    if not registry.list_names():
        registry.load_all(mode=MODE)

    return [
        ToolInfo(name=item["name"], description=item.get("doc", ""))
        for item in registry.list_all()
    ]
