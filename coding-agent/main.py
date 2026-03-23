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

console = Console()

def print_banner():
    banner = """
   ______   ____   ____   ____ 
  /      | /    \\ |    \\ |    |
 |  ,----'|  o  | |  o  ||  |  
 |  |     |     | |     ||  |  
 |  `----.|  _  | |  O  ||  |  
  \\______| \\__| / |_____||____|
    """
    console.print(Panel(Text(banner, style="bold cyan"), title="[bold magenta]Codi CLI Agent[/bold magenta]"))
    with console.status("[bold green]Waking up Codi... initializing neural pathways...", spinner="star"):
        time.sleep(1)
        console.print("[cyan]✧[/cyan] Vector DB connected")
        time.sleep(0.5)
        console.print("[cyan]✧[/cyan] Cognitive Refiner online (Llama 3)")
        time.sleep(0.5)
        console.print("[blue]★[/blue] Ready for input\n")

def print_help():
    help_text = """
**Built-in commands:**
- `/index <path>`: Index a codebase into ChromaDB
- `/plan <task>` : Multi-step execution (Plan -> CSS -> HTML -> JS -> Verify)
- `/mcp`         : List MCP servers and their status
- `/mcp on/off <name>`: Enable/disable an MCP server
- `/clear`       : Wipe session memory
- `/history`     : Print conversation history
- `/help`        : Show this help message
- `/quit`        : Exit
"""
    console.print(Markdown(help_text))

def print_history():
    history_text = session_memory.as_text()
    console.print(f"[bold cyan]Conversation History:[/bold cyan]\n{history_text}")

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
            print_history()
            continue
        elif user_input.startswith("/index"):
            parts = user_input.split(maxsplit=1)
            path = parts[1].strip() if len(parts) > 1 else "."
            abs_path = os.path.abspath(path)
            index_codebase(abs_path)
            try:
                os.chdir(abs_path)
                console.print(f"[bold green]Working directory set to: {abs_path}[/bold green]")
            except Exception as e:
                console.print(f"[bold red]Failed to change directory: {e}[/bold red]")
            continue
        elif user_input.startswith("/plan"):
            parts = user_input.split(maxsplit=1)
            task = parts[1].strip() if len(parts) > 1 else ""
            if not task:
                console.print("[bold red]Usage: /plan <task description>[/bold red]")
                continue
            steps = [
                f"Plan only (no code yet): List every file you will create and every section each file will contain. Task: {task}",
                
                """Write ONLY the file style.css. It must include ALL of these:
- CSS custom properties at the top: --cyan: #00f5ff, --purple: #bf00ff, --dark: #050508
- Google Fonts import for 'Orbitron' (display) and 'Share Tech Mono' (monospace)  
- A scanline overlay using body::before with repeating-linear-gradient
- A CSS grid background using body::after
- text-shadow glows on headings using the cyan and purple variables
- @keyframes fadeUp animation for hero elements
- @keyframes blink-cursor for the typing effect cursor
- Feature cards with border: 1px solid rgba(0,245,255,0.15) and a ::before pseudo-element top border that appears on hover
- The terminal section must look like a real terminal: monospace font, #0a0a0a background, green prompt color #00ff88
- Write the complete file with no placeholders or TODO comments""",

                """Write ONLY the file index.html. Requirements:
- Import Google Fonts in the <head>
- Hero section must have: <h1 class="hero-title">CODI</h1>, a subtitle paragraph, and a <span id="typing-text"> inside a wrapper div
- Features section: each card must have a <div class="feature-icon">, <h3>, and <p> tag separately — not all text jammed in one div
- Terminal section: a .terminal-window div with a .terminal-bar header (three colored dots), and a .terminal-body with id="terminal-output"
- Link style.css in head and script.js before </body>
- Write the complete file""",

                """Write ONLY the file script.js. It must include ALL of these as separate functions:
- A typing effect function that cycles through the 6 phrases with character-by-character typing and a delete animation between phrases
- The typing cursor must blink using a CSS class toggle, not innerHTML hacks
- A terminal simulation that uses setTimeout chains to print each line with a realistic delay
- Each terminal line must be created as a new <div> element with a colored class: .line-user, .line-codi, .line-action
- A subtle card hover glow effect using mousemove event to track cursor position per card
- Write the complete file with no placeholders""",

                f"Run the shell command: ls -la '{abs_path}' to confirm all 3 files exist and have real file sizes (not 0 bytes)."
            ]
            for i, step in enumerate(steps, 1):
                console.print(f"\n[bold magenta]Step {i}/5[/bold magenta]")
                console.print("[italic gray]Refining prompt...[/italic gray]")
                refined = refine_prompt(step)
                session_memory.add("user", refined)
                try:
                    history_str = session_memory.as_text()
                    console.print("[italic blue]Agent is thinking...[/italic blue]")
                    response = agent_executor.invoke({"input": refined, "history": history_str})
                    output = response.get("output", "No output returned.")
                    session_memory.add("assistant", output)
                    console.print(Panel(Markdown(output), border_style="cyan"))
                except Exception as e:
                    console.print(f"[bold red]Error on step {i}:[/bold red] {e}")
                    break
            continue

        elif user_input == "/mcp":
            import json
            config = json.load(open("mcp_servers.json"))
            console.print("\n[bold cyan]MCP Servers:[/]\n")
            for name, cfg in config.items():
                status = "[green]ON [/]" if cfg.get("enabled", False) else "[red]OFF[/]"
                console.print(f"  {status} [bold]{name}[/] — {cfg['description']}")
            console.print("\n[dim]Edit mcp_servers.json to enable/disable servers.[/]\n")
            continue

        elif user_input.startswith("/mcp on "):
            name = user_input.split(maxsplit=2)[2]
            import json
            config = json.load(open("mcp_servers.json"))
            if name in config:
                config[name]["enabled"] = True
                json.dump(config, open("mcp_servers.json", "w"), indent=2)
                console.print(f"[green]Enabled {name}. Restart Codi to load new tools.[/]")
            else:
                console.print(f"[red]Server '{name}' not found.[/]")
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
                console.print(f"[red]Server '{name}' not found.[/]")
            continue

        # Process natural language input
        console.print("[italic gray]Refining prompt...[/italic gray]")
        refined_input = refine_prompt(user_input)
        
        if refined_input != user_input:
            console.print(f"[italic cyan]Refined prompt:[/italic cyan] {refined_input}")

        # Add user prompt to memory
        session_memory.add("user", refined_input)

        try:
            # Run the agent
            console.print("[italic blue]Agent is thinking...[/italic blue]")
            
            # We pass the refined input and current history to the agent
            history_str = session_memory.as_text()
            response = agent_executor.invoke({
                "input": refined_input,
                "history": history_str
            })
            
            output = response.get("output", "No output returned.")
            
            # Print response
            console.print(Markdown(output))
            
            # Add assistant response to memory
            session_memory.add("assistant", output)

        except Exception as e:
            console.print(f"[bold red]Error running agent:[/bold red] {e}")

if __name__ == "__main__":
    main()
