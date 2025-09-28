# 🪄 Magic Terminal

Magic Terminal is a cross-platform, AI-assisted command-line copilot that converts natural language instructions into safe, auditable shell commands. It streamlines installation, automation, diagnostics, and development workflows while providing human-readable previews, intelligent recovery, and multi-model failovers.

![Magic Terminal Demo](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)

---

## 🔥 Highlights

- **Natural Language Automation** — Describe desired outcomes (e.g., "install docker and start it") and Magic Terminal synthesizes the shell commands.
- **Safety First** — Every command is previewed with context, destructive patterns are flagged, and execution requires confirmation (unless auto-confirm is enabled).
- **Intelligent Recovery** — When commands fail, Magic Terminal analyzes stderr, suggests fixes, and retries with smarter alternatives.
- **Multi-LLM Backends** — Works with OpenAI, Grok (xAI), and local Ollama models; degrades gracefully to deterministic heuristics.
- **Cross-Platform Tooling** — Supports package managers, filesystem operations, process management, system monitoring, and development workflows on macOS, Linux, and Windows.
- **Persistent Experience** — Saves history, configuration, bookmarks, and aliases to the user's home directory.

---


## ⚡ Quick Start

```bash
# Clone repository
git clone https://github.com/Yogesh-developer/magic-terminal.git
cd magic-terminal

# Install dependencies
pip3 install -r requirements.txt

# Install Magic Terminal in editable mode
pip3 install -e .

# Launch the assistant
magic-terminal
```

If your shell cannot find the `magic-terminal` script, append the Python user-bin path to `PATH` (macOS example):

```bash
echo 'export PATH="$HOME/Library/Python/3.9/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

Aliases `magic` and `mt` are also installed for convenience.

---

## 📦 Installation Methods

- **PyPI install (users)**
  ```bash
  pip install magic-terminal-cli
  ```
- **Editable install (development)**
  ```bash
  pip3 install -e .
  ```
- **Direct GitHub install (users)**
  ```bash
  pip install git+https://github.com/Yogesh-developer/magic-terminal.git
  ```
- **Virtual environment isolation**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  pip install -e .
  ```

---

## 🤖 Configuring AI Backends

Magic Terminal automatically selects the first available backend in this order:

1. **OpenAI** (requires `OPENAI_API_KEY`)
2. **Grok / X.AI** (`XAI_API_KEY` or `GROK_API_KEY`)
3. **Ollama** (local models at `OLLAMA_URL`, default `http://localhost:11434`)
4. **Heuristic fallback**

Set environment variables in your shell profile:

```bash
export OPENAI_API_KEY="sk-..."          # Optional: OpenAI GPT
export XAI_API_KEY="xai-..."            # Optional: Grok
export GROK_API_KEY="xai-..."           # Alias for Grok
export OLLAMA_URL="http://localhost:11434"
export AI_TERMINAL_ALLOW_FALLBACK=1      # Allow heuristics when LLM fails
```

### Ollama Setup (local LLM)

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3:8b
ollama serve
```

Run `magic-terminal --setup` to verify connectivity and dependencies.

---

## 🕹️ Usage Guide

Launching the CLI:

```bash
magic-terminal
```

Example interactions:

```text
🪄 Magic-Terminal> install docker desktop
🪄 Magic-Terminal> show me running processes using too much memory
🪄 Magic-Terminal> create python backup_logs.py that compresses yesterday's logs
🪄 Magic-Terminal> delete all .tmp files in the current directory
```

Each request produces an execution plan similar to:

```text
🎯 Install Docker Desktop
📂 Working Directory: /Users/alex
⚡ Commands:
  brew install --cask docker
✅ Execute these commands? [y/N]:
```

Commands are executed only after confirmation (unless configured otherwise). Output is streamed live, and failures trigger automatic recovery attempts.

Exit the assistant with `exit`, `quit`, or `Ctrl+C`.

---

## 📚 Command Catalogue

Magic Terminal’s LLM-guided behavior covers a wide range of topics:

- **Package Management** — install/uninstall packages, update systems, search repositories.
- **File & Project Operations** — create templates, organize directories, scaffold projects.
- **Process & Service Control** — list processes, kill tasks, inspect services, analyze logs.
- **System Monitoring** — inspect CPU/memory usage, disk utilization, network stats, system info.
- **Development Tooling** — manage git workflows, run tests, configure environments, lint code.
- **Navigation & Productivity** — change directories intelligently, bookmark locations, manage aliases.

Because results are LLM-driven, review the command preview carefully before approving execution.

---

## 🔁 Intelligent Recovery

When a command fails, Magic Terminal:

1. Parses stderr and identifies known failure patterns.
2. Suggests alternative commands (e.g., macOS `top -o %MEM` ➜ `top -o mem -l 1`).
3. Executes alternatives automatically until a success or all options are exhausted.
4. Falls back to heuristic commands if the LLM backend is unavailable.

Recovery activity is logged to `~/.magic_terminal_logs/enhanced_terminal.log` for auditing.

---

## ⚙️ Configuration & Persistence

- **Configuration file**: `~/.magic_terminal_config.json`
  ```json
  {
    "auto_confirm_safe": false,
    "use_trash": true,
    "max_history": 1000,
    "bookmarks": {},
    "aliases": {},
    "preferred_package_manager": null
  }
  ```
- **Command history**: `~/.magic_terminal_history`
- **Logs**: `~/.magic_terminal_logs/`

Bookmarks, aliases, and preferences can be updated interactively or by editing the config file.

---

## 🛠️ Development Workflow

```bash
git clone https://github.com/Yogesh-developer/magic-terminal.git
cd magic-terminal

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -e .[dev]
pre-commit install

# Run automated tests
pytest
```

---

## ✅ Testing Checklist

- **Automated tests**: `pytest` (covers config migrations, HTTP retry logic, and safety auditing)
- **Manual scenarios**:
  - Offline mode (no API keys, Ollama down) ➜ verify heuristic fallback
  - Package installation on macOS/Linux/Windows ➜ confirm smart suggestions
  - Resource monitoring commands ➜ confirm platform-specific handling
- **Smoke test**: `python3 ai_terminal/main.py`
- **Formatting** *(optional)*: `black ai_terminal/`

---

## 🐞 Troubleshooting

- **Command not found (`magic-terminal`)** — add the user scripts directory to `PATH` (see Quick Start).
- **LLM response invalid / JSON parsing errors** — check logs and ensure the system prompt enforces JSON; fallback is automatic after three attempts.
- **`urllib3 NotOpenSSLWarning` on macOS** — macOS ships LibreSSL; install Python via Homebrew or ignore (warning only).
- **Permission denied** — Magic Terminal detects destructive commands and prompts for confirmation; rerun with `sudo` if needed.
- **Ollama connection errors** — verify `ollama serve` is running and `OLLAMA_URL` matches.

---

## ❓ FAQ

- **Does Magic Terminal require internet access?**
  No, if you use Ollama with local models. OpenAI and Grok require internet access.

- **Can I run it non-interactively?**
  Batch mode is on the roadmap. For now, use the interactive CLI or integrate `EnhancedAITerminal` directly in Python scripts.

- **How do I disable command confirmation?**
  Set `"auto_confirm_safe": true` in the configuration file (feature toggle coming to CLI).

- **Is Windows supported?**
  Yes. Package suggestions use Chocolatey (`choco`) or Winget, and PowerShell commands are generated when appropriate.

- **How do I uninstall?**
  ```bash
  pip3 uninstall magic-terminal-cli
  ```

---

## 🙌 Acknowledgements

{{ ... }}
- [OpenAI](https://platform.openai.com/) and [Grok](https://x.ai) for compatible API backends.
- [Rich](https://github.com/Textualize/rich) for terminal rendering utilities.
- [psutil](https://github.com/giampaolo/psutil) for system metrics collection.

---

## 📄 License

Magic Terminal is distributed under the MIT License. See [`LICENSE`](LICENSE) for full details.
