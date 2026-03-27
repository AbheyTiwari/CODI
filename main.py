import sys
import os
import time
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from dotenv import load_dotenv

load_dotenv()

from memory import session_memory
from indexer import index_codebase
from refiner import refine_prompt
from agent import create_agent
from logger import log

console = Console()

def print_banner():
    banner = r"""

   _____   ____   ____   _____
  / ____| / __ \ |  _ \ |_   _|
 | |     | |  | || | | |  | |
 | |     | |  | || | | |  | |
 | |____ | |__| || |_| | _| |_
  \_____| \____/ |____/ |_____|
    """
    console.print(Panel(Text(banner, style="bold cyan"), title="[bold magenta]Codi CLI Agent[/bold magenta]"))
    with console.status("[bold green]Waking up Codi...", spinner="star"):
        time.sleep(1)
        console.print("[cyan]✧[/cyan] Vector DB connected")
        time.sleep(0.5)
        console.print("[cyan]✧[/cyan] Cognitive Refiner online")
        time.sleep(0.5)
        console.print("[blue]★[/blue] Ready for input\n")

def print_help():
    help_text = """
**Built-in commands:**
- `/index <path>` : Index a codebase into ChromaDB
- `/mcp`          : List MCP servers and their status
- `/mcp on <n>`   : Enable an MCP server
- `/mcp off <n>`  : Disable an MCP server
- `/logs`         : Open live TUI telemetry dashboard
- `/clear`        : Wipe session memory
- `/history`      : Print conversation history
- `/help`         : Show this help message
- `/quit`         : Exit
"""
    console.print(Markdown(help_text))

def get_trimmed_history() -> str:
    """Return last 3 turns only — full history burns tokens fast."""
    full = session_memory.as_text()
    lines = full.split("\n")
    # Keep only last 12 lines (roughly 3 turns)
    trimmed = "\n".join(lines[-12:]) if len(lines) > 12 else full
    return trimmed

def main():
    print_banner()
    session = PromptSession(history=FileHistory(".agent_history"))
    agent_executor = create_agent()
    print_help()

    while True:
        try:
            user_input = session.prompt(">> ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue

        # Guard: reject inputs that look like pasted error output or bare prompts
        if (
            user_input.startswith(">>")                          # pasted prompt prefix
            or user_input.startswith("Graph execution failed")   # pasted error
            or user_input.startswith("Traceback")                # pasted traceback
            or (user_input.startswith("https://") and " " not in user_input)  # bare URL
            or user_input.startswith("Thinking...")              # pasted UI output
        ):
            console.print("[bold yellow]⚠ That looks like pasted output, not a command. Try rephrasing.[/bold yellow]")
            continue

        if user_input.startswith("/quit") or user_input.startswith("/exit"):
            console.print("[bold yellow]Exiting...[/bold yellow]")
            break

        elif user_input.startswith("/help"):
            print_help()
            continue

        elif user_input.startswith("/clear"):
            session_memory.clear()
            console.print("[bold green]Session memory cleared.[/bold green]")
            continue

        elif user_input.startswith("/history"):
            console.print(f"[bold cyan]History:[/bold cyan]\n{session_memory.as_text()}")
            continue

        elif user_input.startswith("/index"):
            parts = user_input.split(maxsplit=1)
            path = parts[1].strip() if len(parts) > 1 else "."
            abs_path = os.path.abspath(path)
            index_codebase(abs_path)
            try:
                os.chdir(abs_path)
                console.print(f"[bold green]Working directory: {abs_path}[/bold green]")
            except Exception as e:
                console.print(f"[bold red]Failed to cd: {e}[/bold red]")
            continue

        elif user_input == "/mcp":
            import json
            try:
                config = json.load(open("mcp_servers.json"))
                console.print("\n[bold cyan]MCP Servers:[/]\n")
                for name, cfg in config.items():
                    status = "[green]ON [/]" if cfg.get("enabled", False) else "[red]OFF[/]"
                    console.print(f"  {status} [bold]{name}[/] — {cfg.get('description', '')}")
                console.print()
            except Exception as e:
                console.print(f"[red]Could not read mcp_servers.json: {e}[/red]")
            continue

        elif user_input.startswith("/mcp on "):
            name = user_input.split(maxsplit=2)[2]
            import json
            config = json.load(open("mcp_servers.json"))
            if name in config:
                config[name]["enabled"] = True
                json.dump(config, open("mcp_servers.json", "w"), indent=2)
                console.print(f"[green]Enabled {name}. Restart to apply.[/]")
            else:
                console.print(f"[red]'{name}' not found.[/]")
            continue

        elif user_input.startswith("/mcp off "):
            name = user_input.split(maxsplit=2)[2]
            import json
            config = json.load(open("mcp_servers.json"))
            if name in config:
                config[name]["enabled"] = False
                json.dump(config, open("mcp_servers.json", "w"), indent=2)
                console.print(f"[yellow]Disabled {name}. Restart to apply.[/]")
            else:
                console.print(f"[red]'{name}' not found.[/]")
            continue

        elif user_input.startswith("/logs"):
            import subprocess
            subprocess.run(["python", "log_viewer.py"])
            continue

        # ── Natural language ──────────────────────────────────────
        refined_input = refine_prompt(user_input)
        if refined_input != user_input:
            console.print(f"[italic cyan]Refined:[/italic cyan] {refined_input}")

        session_memory.add("user", refined_input)

        try:
            console.print("[italic blue]Thinking...[/italic blue]")
            log("agent_start", {"input": refined_input[:200]})

            # Trimmed history — don't dump entire conversation into every call
            history_str = get_trimmed_history()

            response = agent_executor.invoke({
                "input": refined_input,
                "history": history_str
            })

            output = response.get("output", "No output returned.")
            log("agent_end", {"output": output[:200]})
            console.print(Markdown(output))
            session_memory.add("assistant", output)

        except Exception as e:
            log("agent_error", {"error": str(e)})
            console.print(f"[bold red]Error:[/bold red] {e}")

if __name__ == "__main__":
    main()