# 🪄 Magic Terminal

A magical terminal assistant that understands natural language and makes command-line operations effortless with intelligent automation.

![Magic Terminal Demo](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)

## ✨ Features

- 🧠 **Natural Language Processing** - Describe what you want in plain English
- 📦 **Package Management** - Install, update, and manage software across platforms
- 🔧 **Process Management** - Monitor, kill, and manage system processes
- 📁 **File Operations** - Create, delete, move files with templates and safety checks
- 🖥️ **System Monitoring** - Real-time resource usage and system information
- 🔄 **Intelligent Recovery** - Automatic error analysis and alternative suggestions
- 🛡️ **Safety Features** - Confirmation prompts and dangerous command detection
- 🌐 **Cross-Platform** - Works on Windows, macOS, and Linux
- 🎯 **Smart Fallbacks** - Works even when AI services are unavailable

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-terminal.git
cd ai-terminal

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Setup

```bash
# Run the setup wizard
ai-terminal --setup

# Or start directly
ai-terminal
```

## 🎮 Usage Examples

### Natural Language Commands

```bash
🤖 AI-Terminal> install firefox
🤖 AI-Terminal> show me running processes using too much memory
🤖 AI-Terminal> create a python script called hello.py
🤖 AI-Terminal> delete all .tmp files in this directory
🤖 AI-Terminal> update all my packages
🤖 AI-Terminal> show system resource usage
```

### Direct Commands

```bash
# Package management
ai-terminal install docker
ai-terminal update system

# File operations
ai-terminal create python my_script.py
ai-terminal delete old_files/

# System monitoring
ai-terminal monitor processes
ai-terminal show memory usage
```

## 🔧 Configuration

### Environment Variables

```bash
# Ollama server URL (default: http://localhost:11434)
export OLLAMA_URL="http://localhost:11434"

# Grok API key (optional)
export XAI_API_KEY="your-grok-api-key"

# Enable fallback mode
export AI_TERMINAL_ALLOW_FALLBACK=1
```

### Configuration File

The terminal creates a configuration file at `~/.ai_terminal_config.json`:

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

## 🛠️ Requirements

### System Requirements

- Python 3.8 or higher
- 4GB RAM minimum
- Internet connection for AI features

### Dependencies

- `requests>=2.28.0` - HTTP requests
- `psutil>=5.9.0` - System monitoring
- `jsonschema>=4.0.0` - JSON validation
- `colorama>=0.4.4` - Cross-platform colored output
- `rich>=12.0.0` - Rich text formatting
- `prompt-toolkit>=3.0.0` - Enhanced input handling

### Optional Dependencies

- **Ollama** - Local AI model server (recommended)
- **Grok API** - Cloud AI service (alternative)

## 🔌 AI Backend Setup

### Option 1: Ollama (Recommended)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3:8b

# Start Ollama (runs automatically on most systems)
ollama serve
```

### Option 2: Grok API

```bash
# Set your API key
export XAI_API_KEY="your-grok-api-key"
```

## 📚 Commands Reference

### Package Management
- `install <package>` - Install software
- `uninstall <package>` - Remove software
- `update` - Update all packages
- `search <package>` - Search for packages

### File Operations
- `create <type> <name>` - Create files from templates
- `delete <path>` - Safely delete files/folders
- `move <src> <dest>` - Move files/folders
- `copy <src> <dest>` - Copy files/folders

### Process Management
- `ps [filter]` - List processes
- `kill <pid/name>` - Terminate process
- `services` - List running services
- `monitor` - Real-time system monitoring

### System Information
- `status` - System resource usage
- `disk` - Disk usage information
- `network` - Network status
- `logs <file>` - Analyze log files

### Navigation
- `cd <path>` - Change directory
- `bookmark <name>` - Bookmark current directory
- `goto <bookmark>` - Go to bookmarked directory
- `history` - Show command history

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-terminal.git
cd ai-terminal

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black ai_terminal/
flake8 ai_terminal/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai) for local AI model serving
- [Grok](https://x.ai) for cloud AI services
- [Rich](https://github.com/Textualize/rich) for beautiful terminal output
- [psutil](https://github.com/giampaolo/psutil) for system monitoring

## 📞 Support

- 📧 Email: support@aiterminal.dev
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/ai-terminal/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/ai-terminal/discussions)
- 📖 Documentation: [Read the Docs](https://ai-terminal.readthedocs.io/)

---

**Made with ❤️ by the AI Terminal Team**
