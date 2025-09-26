#!/usr/bin/env python3
"""
Enhanced AI Terminal - A comprehensive, robust terminal assistant
Handles all human terminal activities: installation, deletion, creation, updates, etc.
"""

import os
import sys
import subprocess
import platform
import json
import logging
import requests
import re
import shlex
import atexit
import time
import psutil
import threading
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass
from enum import Enum
from difflib import get_close_matches

try:
    from .llm_client import LLMClient, PlanStep
except ImportError:  # When executed as a script
    from llm_client import LLMClient, PlanStep

try:
    from jsonschema import validate, ValidationError
except ImportError:
    validate = None
    ValidationError = None

try:
    import readline
except ImportError:
    readline = None

# Configuration
LOG_FILE = os.path.expanduser("~/.ai_terminal_logs/enhanced_terminal.log")
HISTORY_FILE = os.path.expanduser("~/.ai_terminal_history")
CONFIG_FILE = os.path.expanduser("~/.ai_terminal_config.json")

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Logging setup
logger = logging.getLogger("enhanced_ai_terminal")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(file_handler)
    logger.propagate = False


class OperationType(Enum):
    INSTALL = "install"
    DELETE = "delete"
    CREATE = "create"
    UPDATE = "update"
    NAVIGATE = "navigate"
    PROCESS = "process"
    NETWORK = "network"
    MONITOR = "monitor"
    DEV_TOOLS = "dev_tools"
    FILE_OPS = "file_ops"


@dataclass
class CommandResult:
    success: bool
    output: str
    error: str
    duration: float
    command: str


class PackageManager:
    """Enhanced package management system"""
    
    def __init__(self, system_os: str):
        self.system_os = system_os
        self.managers = {
            'darwin': {
                'brew': {'install': 'brew install {}', 'uninstall': 'brew uninstall {}', 'update': 'brew update && brew upgrade'},
                'pip': {'install': 'pip install {}', 'uninstall': 'pip uninstall {}', 'update': 'pip install --upgrade {}'},
                'npm': {'install': 'npm install -g {}', 'uninstall': 'npm uninstall -g {}', 'update': 'npm update -g {}'}
            },
            'linux': {
                'apt': {'install': 'sudo apt install -y {}', 'uninstall': 'sudo apt remove {}', 'update': 'sudo apt update && sudo apt upgrade'},
                'yum': {'install': 'sudo yum install -y {}', 'uninstall': 'sudo yum remove {}', 'update': 'sudo yum update'},
                'pip': {'install': 'pip install {}', 'uninstall': 'pip uninstall {}', 'update': 'pip install --upgrade {}'},
                'npm': {'install': 'npm install -g {}', 'uninstall': 'npm uninstall -g {}', 'update': 'npm update -g {}'}
            },
            'windows': {
                'choco': {'install': 'choco install {} -y', 'uninstall': 'choco uninstall {}', 'update': 'choco upgrade all'},
                'winget': {'install': 'winget install {}', 'uninstall': 'winget uninstall {}', 'update': 'winget upgrade --all'},
                'pip': {'install': 'pip install {}', 'uninstall': 'pip uninstall {}', 'update': 'pip install --upgrade {}'}
            }
        }
        self.available_managers = self._detect_available_managers()
    
    def _detect_available_managers(self) -> List[str]:
        """Detect which package managers are available"""
        available = []
        for manager in self.managers.get(self.system_os, {}):
            try:
                result = subprocess.run([manager, '--version'], 
                                      capture_output=True, timeout=5)
                if result.returncode == 0:
                    available.append(manager)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        return available
    
    def suggest_install_command(self, package: str) -> List[str]:
        """Suggest installation commands for a package"""
        commands = []
        for manager in self.available_managers:
            if manager in self.managers.get(self.system_os, {}):
                cmd_template = self.managers[self.system_os][manager]['install']
                commands.append(cmd_template.format(package))
        return commands


class FileOperations:
    """Enhanced file operations with safety and templates"""
    
    def __init__(self):
        self.templates = {
            'python': '''#!/usr/bin/env python3
"""
{description}
"""

def main():
    pass

if __name__ == "__main__":
    main()
''',
            'javascript': '''/**
 * {description}
 */

function main() {{
    // Your code here
}}

main();
''',
            'html': '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
</head>
<body>
    <h1>{title}</h1>
</body>
</html>
''',
            'readme': '''# {title}

## Description
{description}

## Installation
```bash
# Installation instructions
```

## Usage
```bash
# Usage examples
```

## Contributing
Pull requests are welcome.

## License
MIT
'''
        }
    
    def create_from_template(self, file_type: str, name: str, **kwargs) -> str:
        """Create file from template"""
        if file_type.lower() not in self.templates:
            return f"# {name}\n\n"
        
        template = self.templates[file_type.lower()]
        return template.format(
            title=kwargs.get('title', name),
            description=kwargs.get('description', f'Auto-generated {file_type} file'),
            **kwargs
        )
    
    def safe_delete(self, path: str, use_trash: bool = True) -> bool:
        """Safely delete files/directories"""
        path_obj = Path(path)
        if not path_obj.exists():
            return False
        
        if use_trash:
            # Try to use system trash
            try:
                if platform.system() == "Darwin":
                    subprocess.run(["osascript", "-e", f'tell app "Finder" to delete POSIX file "{path}"'])
                elif platform.system() == "Linux":
                    subprocess.run(["gio", "trash", path])
                elif platform.system() == "Windows":
                    subprocess.run(["powershell", "-Command", f"Add-Type -AssemblyName Microsoft.VisualBasic; [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile('{path}', 'OnlyErrorDialogs', 'SendToRecycleBin')"])
                return True
            except:
                pass
        
        # Fallback to regular deletion
        try:
            if path_obj.is_file():
                path_obj.unlink()
            else:
                import shutil
                shutil.rmtree(path)
            return True
        except Exception as e:
            logger.error(f"Failed to delete {path}: {e}")
            return False


class ProcessManager:
    """Process and service management"""
    
    def list_processes(self, filter_term: Optional[str] = None) -> List[Dict]:
        """List running processes"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                proc_info = proc.info
                if filter_term and filter_term.lower() not in proc_info['name'].lower():
                    continue
                processes.append(proc_info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return processes
    
    def kill_process(self, identifier: Union[int, str], force: bool = False) -> bool:
        """Kill process by PID or name"""
        try:
            if isinstance(identifier, int):
                proc = psutil.Process(identifier)
            else:
                # Find by name
                for proc in psutil.process_iter(['pid', 'name']):
                    if proc.info['name'].lower() == identifier.lower():
                        proc = psutil.Process(proc.info['pid'])
                        break
                else:
                    return False
            
            if force:
                proc.kill()
            else:
                proc.terminate()
            return True
        except Exception as e:
            logger.error(f"Failed to kill process {identifier}: {e}")
            return False


class SystemMonitor:
    """System monitoring and resource tracking"""
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current system resource usage"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory': psutil.virtual_memory()._asdict(),
            'disk': psutil.disk_usage('/')._asdict(),
            'network': psutil.net_io_counters()._asdict()
        }
    
    def get_running_services(self) -> List[str]:
        """Get list of running services"""
        services = []
        try:
            if platform.system() == "Darwin":
                result = subprocess.run(['launchctl', 'list'], capture_output=True, text=True)
                services = result.stdout.split('\n')[1:]  # Skip header
            elif platform.system() == "Linux":
                result = subprocess.run(['systemctl', 'list-units', '--type=service', '--state=running'], 
                                      capture_output=True, text=True)
                services = result.stdout.split('\n')[1:]  # Skip header
            elif platform.system() == "Windows":
                result = subprocess.run(['sc', 'query', 'state=', 'running'], capture_output=True, text=True)
                services = result.stdout.split('\n')
        except Exception as e:
            logger.error(f"Failed to get services: {e}")
        return services


class EnhancedAITerminal:
    """Enhanced AI Terminal with comprehensive functionality"""
    
    def __init__(self, *, enable_fallback: bool = True):
        self.system_os = platform.system().lower()
        self.shell_prompt = "🤖 AI-Terminal> "
        self.current_dir = os.getcwd()
        self.allow_fallback = enable_fallback
        
        # Initialize components
        self.package_manager = PackageManager(self.system_os)
        self.file_ops = FileOperations()
        self.process_manager = ProcessManager()
        self.system_monitor = SystemMonitor()
        self.llm_client = self._init_llm_client()
        
        # Configuration
        self.config = self._load_config()
        
        # LLM settings
        self.ollama_url = "http://localhost:11434"
        self.grok_api_key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.available_models = self._check_ollama()
        
        # State management
        self._command_history: List[CommandResult] = []
        self._session_context: Dict[str, Any] = {}
        self._bookmarks: Dict[str, str] = self.config.get('bookmarks', {})
        
        # Setup
        self._setup_history()
        self._setup_logging()
        
        logger.info("🚀 Enhanced AI Terminal initialized")
        logger.info(f"💻 OS: {platform.system()} | Directory: {self.current_dir}")
        logger.info(f"📦 Available package managers: {self.package_manager.available_managers}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        default_config = {
            'auto_confirm_safe': False,
            'use_trash': True,
            'max_history': 1000,
            'bookmarks': {},
            'aliases': {},
            'preferred_package_manager': None
        }
        
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    default_config.update(config)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
        
        return default_config
    
    def _save_config(self):
        """Save configuration to file"""
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def _check_ollama(self) -> List[str]:
        """Check available Ollama models"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
        except Exception:
            pass
        return []
    
    def _setup_history(self):
        """Setup command history"""
        if not readline:
            return
        
        try:
            if os.path.exists(HISTORY_FILE):
                readline.read_history_file(HISTORY_FILE)
            readline.set_history_length(self.config.get('max_history', 1000))
        except Exception as e:
            logger.warning(f"Failed to setup history: {e}")
        
        atexit.register(self._save_history)
    
    def _save_history(self):
        """Save command history"""
        if not readline:
            return
        
        try:
            readline.write_history_file(HISTORY_FILE)
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
    
    def _setup_logging(self):
        """Setup enhanced logging"""
        pass  # Already configured globally
    
    def execute_command(self, command: str, working_dir: Optional[str] = None) -> CommandResult:
        """Execute a shell command with enhanced error handling"""
        start_time = time.time()
        work_dir = working_dir or self.current_dir
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            duration = time.time() - start_time
            cmd_result = CommandResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr,
                duration=duration,
                command=command
            )
            
            self._command_history.append(cmd_result)
            return cmd_result
            
        except subprocess.TimeoutExpired:
            return CommandResult(
                success=False,
                output="",
                error="Command timed out after 5 minutes",
                duration=time.time() - start_time,
                command=command
            )
        except Exception as e:
            return CommandResult(
                success=False,
                output="",
                error=str(e),
                duration=time.time() - start_time,
                command=command
            )
    
    def run(self):
        """Main terminal loop"""
        print("🚀 Enhanced AI Terminal")
        print("💡 Type 'help' for commands, 'exit' to quit")
        print("🔧 Advanced operations: install, delete, create, monitor, etc.")
        
        while True:
            try:
                user_input = input(f"\n{self.shell_prompt}").strip()
                
                if not user_input:
                    continue
                
                if readline and user_input:
                    readline.add_history(user_input)
                
                if user_input.lower() in ['exit', 'quit']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                # Process the command
                self._process_user_input(user_input)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"💥 Error: {e}")
                logger.error(f"Unexpected error: {e}")
    
    def _show_help(self):
        """Show help information"""
        help_text = """
🤖 Enhanced AI Terminal Commands:

📦 Package Management:
  install <package>     - Install software package
  uninstall <package>   - Remove software package
  update               - Update all packages

📁 File Operations:
  create <type> <name>  - Create file from template
  delete <path>        - Safely delete files/folders
  mkdir <path>         - Create directories
  
🔧 Process Management:
  ps [filter]          - List running processes
  kill <pid/name>      - Terminate process
  services             - List running services

📊 System Monitoring:
  status               - System resource usage
  monitor              - Real-time monitoring
  logs <file>          - Analyze log files

🧭 Navigation:
  cd <path>            - Change directory
  bookmark <name>      - Bookmark current directory
  goto <bookmark>      - Go to bookmarked directory

⚙️  Configuration:
  config               - Show configuration
  alias <name> <cmd>   - Create command alias

💡 Natural Language:
  Just describe what you want to do in plain English!
  Examples:
  - "install python and create a new project"
  - "show me running processes using too much memory"
  - "delete all .tmp files in this directory"
        """
        print(help_text)
    
    def _process_user_input(self, user_input: str):
        """Process user input with enhanced understanding"""
        try:
            # Get AI interpretation with retries
            command_info = self._understand_complex_command(user_input)
            
            if not command_info:
                print("❌ Unable to understand the request")
                return
            
            print(f"🎯 {command_info.get('description', 'Executing command')}")
            
            # Execute commands
            success = self._execute_commands(command_info, user_input)
            if success:
                print("✅ Done")
            else:
                print("❌ Failed")
                
        except Exception as e:
            print(f"💥 Error: {e}")
            logger.error(f"Error processing input: {e}")


    def _understand_complex_command(self, user_input: str) -> Optional[Dict[str, Any]]:
        """Use AI to understand complex commands with retries"""
        max_attempts = 3
        attempt = 0
        last_error = None
        
        while attempt < max_attempts:
            attempt += 1
            try:
                return self._call_local_llm(user_input)
            except ValueError as exc:
                last_error = exc
                logger.warning(f"LLM response invalid (attempt {attempt}/{max_attempts}): {exc}")
                print(f"⚠️  LLM response invalid (attempt {attempt}/{max_attempts}). Retrying...")
            except Exception as exc:
                last_error = exc
                logger.warning(f"LLM call failed (attempt {attempt}/{max_attempts}): {exc}")
                print(f"⚠️  LLM call failed (attempt {attempt}/{max_attempts}). Retrying...")
        
        if self.allow_fallback:
            logger.warning(f"LLM unavailable after retries ({last_error}). Using fallback.")
            print(f"⚠️  Using fallback commands instead.")
            return self._advanced_fallback(user_input)
        
        return None
    
    def _call_local_llm(self, user_input: str) -> Dict[str, Any]:
        """Call LLM backend with enhanced understanding"""
        # Check for special cases first
        lower_input = user_input.lower()
        if any(phrase in lower_input for phrase in ['list users', 'show users']) or (
            'user' in lower_input and any(word in lower_input for word in ['list', 'show'])):
            return self._handle_user_directory_listing()
        
        if self.openai_api_key:
            return self._call_openai(user_input)
        elif self.grok_api_key:
            return self._call_grok(user_input)
        elif self.available_models:
            return self._call_ollama(user_input)
        else:
            raise Exception("No LLM backend available")
    
    def _call_ollama(self, user_input: str) -> Dict[str, Any]:
        """Call Ollama API"""
        model = self.available_models[0] if self.available_models else "llama2"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": user_input}
            ],
            "stream": False,
            "options": {"temperature": 0.1}
        }
        
        logger.info(f"🧠 Ollama analyzing: '{user_input}'")
        print(f"🧠 Analyzing: '{user_input}'")
        
        response = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=45)
        if not response.ok:
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            raise Exception(f"Ollama API error: {response.status_code}")
        
        data = response.json()
        content = data["message"]["content"]
        logger.debug(f"Ollama raw content: {content}")
        
        return self._parse_llm_response(content)
    
    def _call_openai(self, user_input: str) -> Dict[str, Any]:
        """Call OpenAI API"""
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-3.5-turbo",
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": user_input}
            ]
        }
        
        logger.info(f"🧠 OpenAI analyzing: '{user_input}'")
        print(f"🧠 Analyzing: '{user_input}'")
        
        response = requests.post("https://api.openai.com/v1/chat/completions", 
                               headers=headers, json=payload, timeout=45)
        if not response.ok:
            raise Exception(f"OpenAI API error: {response.status_code}")
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        
        return self._parse_llm_response(content)
    
    def _call_grok(self, user_input: str) -> Dict[str, Any]:
        """Call Grok API"""
        headers = {
            "Authorization": f"Bearer {self.grok_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "grok-1",
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": user_input}
            ]
        }
        
        logger.info(f"🧠 Grok analyzing: '{user_input}'")
        print(f"🧠 Analyzing: '{user_input}'")
        
        response = requests.post("https://api.x.ai/v1/chat/completions", 
                               headers=headers, json=payload, timeout=45)
        if not response.ok:
            raise Exception(f"Grok API error: {response.status_code}")
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        
        return self._parse_llm_response(content)
    
    def _get_system_prompt(self) -> str:
        """Get enhanced system prompt"""
        current_files = "\n".join([f"  {f}" for f in os.listdir(self.current_dir)[:10]])
        
        return f"""You are an advanced AI terminal assistant. Convert natural language to executable commands.

CURRENT SYSTEM: {self.system_os.upper()}
CURRENT DIRECTORY: {self.current_dir}
FILES IN CURRENT DIRECTORY:
{current_files}

AVAILABLE PACKAGE MANAGERS: {', '.join(self.package_manager.available_managers)}

CAPABILITIES:
1. Package management: install, uninstall, update software
2. File operations: create, delete, move, copy files/directories
3. Process management: list, kill processes, manage services
4. System monitoring: resource usage, logs, network status
5. Development tools: git operations, project scaffolding
6. Navigation: directory traversal, bookmarks

ALWAYS return valid JSON with this schema:
{{
    "commands": ["full command with all arguments", "another complete command"],
    "description": "what will be executed",
    "type": "install|delete|create|update|navigate|process|monitor|dev_tools|file_ops",
    "working_directory": "optional/path"
}}

IMPORTANT: Each command in the "commands" array must be a COMPLETE command string with all arguments.
For example: ["docker run --help", "ls -la"] NOT ["docker", "run", "--help", "ls", "-la"]

Convert the user's request to appropriate terminal commands."""
    
    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM response"""
        logger.debug(f"Raw LLM response: {content}")
        
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                parsed_response = json.loads(json_match.group())
            else:
                # Try parsing the whole content
                parsed_response = json.loads(content)
        except json.JSONDecodeError:
            # Handle double-encoded JSON (quoted JSON string)
            try:
                if content.startswith('"') and content.endswith('"'):
                    # Remove outer quotes and parse
                    unquoted = json.loads(content)
                    parsed_response = json.loads(unquoted)
                else:
                    # Try parsing as quoted JSON without outer quotes
                    parsed_response = json.loads(json.loads(f'"{content}"'))
            except (json.JSONDecodeError, TypeError):
                logger.error(f"Failed to parse LLM response: {content}")
                raise ValueError("LLM response did not contain valid JSON")
        
        logger.debug(f"Parsed LLM response: {parsed_response}")
        
        # Fix incorrectly split commands
        if 'commands' in parsed_response:
            commands = parsed_response['commands']
            fixed_commands = self._fix_split_commands(commands)
            if fixed_commands != commands:
                logger.info(f"Fixed split commands: {commands} -> {fixed_commands}")
                parsed_response['commands'] = fixed_commands
        
        return parsed_response
    
    def _fix_split_commands(self, commands: List[str]) -> List[str]:
        """Fix commands that have been incorrectly split into separate words"""
        if not commands:
            return commands
        
        # Check if we have a pattern that suggests incorrect splitting
        # Common patterns: single words that should be part of a larger command
        fixed_commands = []
        i = 0
        
        while i < len(commands):
            current_cmd = commands[i].strip()
            
            # Check if this looks like a base command that should have arguments
            if self._is_base_command(current_cmd) and i + 1 < len(commands):
                # Try to reconstruct the full command
                full_command_parts = [current_cmd]
                j = i + 1
                
                # Collect subsequent parts that look like arguments
                while j < len(commands) and self._looks_like_argument(commands[j]):
                    full_command_parts.append(commands[j].strip())
                    j += 1
                
                # If we collected multiple parts, join them
                if len(full_command_parts) > 1:
                    fixed_commands.append(' '.join(full_command_parts))
                    i = j
                else:
                    fixed_commands.append(current_cmd)
                    i += 1
            else:
                fixed_commands.append(current_cmd)
                i += 1
        
        return fixed_commands
    
    def _is_base_command(self, cmd: str) -> bool:
        """Check if a string looks like a base command that typically takes arguments"""
        base_commands = {
            'docker', 'git', 'npm', 'pip', 'brew', 'apt', 'yum', 'node', 'python', 'python3',
            'ls', 'cd', 'mkdir', 'rm', 'cp', 'mv', 'cat', 'grep', 'find', 'ps', 'top',
            'curl', 'wget', 'ssh', 'scp', 'rsync', 'tar', 'zip', 'unzip'
        }
        return cmd.lower() in base_commands
    
    def _looks_like_argument(self, arg: str) -> bool:
        """Check if a string looks like a command argument"""
        arg = arg.strip()
        if not arg:
            return False
        
        # Arguments typically start with - or --, or are subcommands/parameters
        if arg.startswith('-'):
            return True
        
        # Common subcommands
        subcommands = {
            'run', 'build', 'push', 'pull', 'install', 'uninstall', 'update', 'upgrade',
            'start', 'stop', 'restart', 'status', 'list', 'show', 'help', 'version',
            'add', 'commit', 'clone', 'checkout', 'merge', 'branch', 'log', 'diff'
        }
        
        if arg.lower() in subcommands:
            return True
        
        # If it's a single word without spaces and doesn't look like a standalone command
        if ' ' not in arg and not self._is_base_command(arg):
            return True
        
        return False

    def _is_already_installed(self, command: str, package: str) -> bool:
        """Determine if a package appears to already be installed."""
        command_lower = command.lower()

        # Quick command availability check
        if shutil.which(package):
            return True

        # Package-manager specific checks
        if command_lower.startswith("brew install"):
            return self._run_check_command(["brew", "list", "--versions", package]) or \
                self._run_check_command(["brew", "list", "--cask", package])

        if command_lower.startswith("sudo apt") or command_lower.startswith("apt"):
            return self._run_check_command(["dpkg", "-s", package])

        if command_lower.startswith("sudo yum") or command_lower.startswith("yum"):
            return self._run_check_command(["rpm", "-q", package])

        if command_lower.startswith("pip install"):
            return self._run_check_command(["pip", "show", package]) or \
                self._run_check_command(["pip3", "show", package])

        if command_lower.startswith("npm install"):
            return self._run_check_command(["npm", "list", "-g", package])

        if command_lower.startswith("winget install"):
            return self._run_check_command(["winget", "list", package])

        if command_lower.startswith("choco install"):
            return self._run_check_command(["choco", "list", "--local-only", package])

        return False

    def _run_check_command(self, args: List[str]) -> bool:
        """Run a helper command to detect existing installations."""
        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except FileNotFoundError:
            return False
        except subprocess.TimeoutExpired:
            return False

        if result.returncode != 0:
            return False

        output = (result.stdout or "").strip()
        error = (result.stderr or "").strip()
        return bool(output) or "installed" in error.lower()

    def _init_llm_client(self) -> Optional[LLMClient]:
        try:
            return LLMClient()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to initialize LLM client: {exc}")
            return None

    def _prompt_confirmation(self, message: str) -> bool:
        """Prompt the user for confirmation, honoring auto-confirm config."""
        if self.config.get('auto_confirm_safe', False):
            return True
        return input(f"{message} [y/N]: ").strip().lower() in {"y", "yes"}

    def _escalate_to_llm_failure(
        self,
        *,
        failed_command: str,
        error_output: str,
        executed_steps: List[PlanStep],
        user_input: str,
        working_dir: str,
    ) -> bool:
        if not self.llm_client:
            print("❌ LLM assistance unavailable. Manual intervention required.")
            return False

        print("🤖 Escalating failure to LLM for guidance...")

        failed_step = PlanStep(
            step=len(executed_steps) + 1,
            command=failed_command,
            description="Failed command",
        )

        try:
            fix_steps = self.llm_client.suggest_fix(
                failed_step=failed_step,
                error_output=error_output,
                executed_steps=executed_steps,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"❌ LLM recovery attempt failed: {exc}")
            return False

        if not fix_steps:
            print("❌ LLM did not provide any recovery steps. Manual intervention required.")
            return False

        print("📋 LLM suggested recovery plan:")
        for step in fix_steps:
            desc = step.description or step.command
            print(f"  {step.step}. {desc}")

        if not self._prompt_confirmation("Execute LLM recovery plan?"):
            print("❌ Recovery plan rejected by user. Manual intervention required.")
            return False

        for idx, step in enumerate(fix_steps, start=1):
            print(f"🔁 [LLM {idx}/{len(fix_steps)}] {step.command}")
            result = self.execute_command(step.command, working_dir)

            if result.output:
                print(result.output)
            if result.error:
                print(f"⚠️  {result.error}")

            if not result.success:
                print("❌ LLM recovery step failed. Manual intervention required.")
                return False

            executed_steps.append(
                PlanStep(
                    step=len(executed_steps) + 1,
                    command=step.command,
                    description=step.description or "LLM recovery step",
                )
            )

        print("🔁 Retrying original command after LLM fixes...")
        retry_result = self.execute_command(failed_command, working_dir)

        if retry_result.output:
            print(retry_result.output)
        if retry_result.error:
            print(f"⚠️  {retry_result.error}")

        if not retry_result.success:
            print("❌ Command still failing after LLM recovery. Manual intervention required.")
            return False

        executed_steps.append(
            PlanStep(
                step=len(executed_steps) + 1,
                command=failed_command,
                description="Command retried after LLM fix",
            )
        )

        print("✅ Command succeeded after LLM-guided recovery.")
        return True
    
    def _advanced_fallback(self, user_input: str) -> Dict[str, Any]:
        """Advanced fallback for when LLM is unavailable"""
        input_lower = user_input.lower()
        
        # Installation requests
        if 'install' in input_lower:
            package = re.search(r'install\s+(\w+)', input_lower)
            if package:
                pkg_name = package.group(1)
                commands = self.package_manager.suggest_install_command(pkg_name)
                return {
                    "commands": commands[:1],  # Use first available
                    "description": f"Install {pkg_name}",
                    "type": "install"
                }
        
        # File creation
        if 'create' in input_lower:
            return {
                "commands": [f"touch {user_input.split()[-1]}"],
                "description": "Create file",
                "type": "create"
            }
        
        # Directory listing
        if any(word in input_lower for word in ['list', 'ls', 'show']):
            return {
                "commands": ["ls -la"],
                "description": "List directory contents",
                "type": "file_ops"
            }
        
        # Default
        return {
            "commands": [user_input],
            "description": "Execute command directly",
            "type": "file_ops"
        }
    
    def _handle_user_directory_listing(self) -> Dict[str, Any]:
        """Handle user directory listing"""
        users_root = Path("/Users") if self.system_os != "windows" else Path("C:/Users")
        
        # Create helper script
        helper_script = Path.home() / ".ai_terminal_user_listing.py"
        helper_script.write_text(f'''
import sys
from pathlib import Path

users_root = Path(r"{users_root}")
dirs = sorted([p for p in users_root.iterdir() if p.is_dir()])

if not dirs:
    print(f"No folders found under {users_root}")
    sys.exit(0)

print("Available user directories:")
for idx, path in enumerate(dirs, 1):
    print(f"{idx}. {path.name}")

choice = input("Select a user directory #: ").strip()
if not choice.isdigit() or not (1 <= int(choice) <= len(dirs)):
    print("Invalid selection.")
    sys.exit(1)

selected = dirs[int(choice) - 1]
print(f"\\nContents of {selected}:\\n")
for entry in sorted(selected.iterdir()):
    suffix = "/" if entry.is_dir() else ""
    print(f"{entry.name}{suffix}")
''')
        
        return {
            "commands": [f"python3 ~/.ai_terminal_user_listing.py"],
            "description": f"List folders under {users_root} and display selected contents",
            "type": "file_ops",
            "working_directory": str(users_root)
        }
    
    def _execute_commands(self, command_info: Dict[str, Any], user_input: str) -> bool:
        """Execute commands with enhanced error handling"""
        commands = command_info.get("commands", [])
        if not commands:
            return False
        
        working_dir = command_info.get("working_directory", self.current_dir)
        
        print(f"📂 Working Directory: {working_dir}")
        print("⚡ Commands:")
        for cmd in commands:
            print(f"  {cmd}")
        
        # Ask for confirmation
        if not self.config.get('auto_confirm_safe', False):
            confirm = input("✅ Execute these commands? [y/N]: ").strip().lower()
            if confirm not in {"y", "yes"}:
                print("❌ Execution cancelled.")
                return False
        
        executed_steps: List[PlanStep] = []

        # Execute commands
        for i, cmd in enumerate(commands):
            print(f"🔧 [{i+1}/{len(commands)}] {cmd}")
            
            # Check if it's an installation command
            install_target = self._extract_install_target(cmd)
            start_time = time.time() if install_target else None
            
            if install_target:
                if self._is_already_installed(cmd, install_target):
                    print(f"✅ {install_target} is already installed. Skipping command.")
                    executed_steps.append(
                        PlanStep(
                            step=len(executed_steps) + 1,
                            command=cmd,
                            description=f"Skipped installation; {install_target} already present",
                        )
                    )
                    continue
                print(f"⬇️ Installing {install_target} (this may take a moment)...")
            
            result = self.execute_command(cmd, working_dir)
            
            if install_target and start_time:
                duration = time.time() - start_time
                status = "✅" if result.success else "⚠️"
                print(f"{status} Installation step for {install_target} finished in {duration:.1f}s")

            if result.output:
                print(result.output)
            if result.error:
                print(f"⚠️  {result.error}")

            if result.success:
                executed_steps.append(
                    PlanStep(
                        step=len(executed_steps) + 1,
                        command=cmd,
                        description="Executed successfully",
                    )
                )
                continue

            print(f"❌ Command failed: {cmd}")

            # Try heuristic recovery first
            if self._attempt_intelligent_recovery(cmd, result.error, user_input):
                executed_steps.append(
                    PlanStep(
                        step=len(executed_steps) + 1,
                        command=cmd,
                        description="Recovered via heuristic fix",
                    )
                )
                continue

            # Escalate to LLM for assistance
            if self._escalate_to_llm_failure(
                failed_command=cmd,
                error_output=result.error,
                executed_steps=executed_steps,
                user_input=user_input,
                working_dir=working_dir,
            ):
                continue

            print("❌ Manual intervention required. Aborting remaining steps.")
            return False

        return True

    def _extract_install_target(self, command: str) -> Optional[str]:
        """Extract package name from install command"""
        patterns = [
            r"brew\s+install(?:\s+--cask)?\s+([\w\-\.]+)",
            r"sudo\s+yum\s+install\s+(?:-y\s+)?([\w\-\.]+)",
            r"choco\s+install\s+([\w\-\.]+)",
            r"pip\s+install\s+([\w\-\.]+)",
            r"npm\s+install\s+(?:-g\s+)?([\w\-\.]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, command, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def _attempt_intelligent_recovery(self, failed_command: str, error_output: str, original_request: str) -> bool:
        """Attempt intelligent recovery by analyzing the error and finding alternatives"""
        print("🔄 Analyzing error and attempting recovery...")
        
        # Analyze the error and suggest alternatives
        alternatives = self._analyze_error_and_suggest_alternatives(failed_command, error_output, original_request)
        
        if not alternatives:
            print("❌ No recovery options found")
            return False
        
        print("💡 Found alternative solutions:")
        for i, alt in enumerate(alternatives, 1):
            print(f"  {i}. {alt['description']}")
        
        # Try alternatives automatically
        for alt in alternatives:
            print(f"🔧 Trying: {alt['command']}")
            result = self.execute_command(alt['command'])
            
            if result.success:
                if result.output:
                    print(result.output)
                print("✅ Recovery successful!")
                return True
            else:
                print(f"⚠️  Alternative failed: {result.error}")
        
        print("❌ All recovery attempts failed")
        return False
    
    def _analyze_error_and_suggest_alternatives(self, failed_command: str, error_output: str, original_request: str) -> List[Dict[str, str]]:
        """Analyze error and suggest alternative commands"""
        alternatives = []
        
        # Handle top command memory sorting issues
        if "top" in failed_command and "invalid argument" in error_output and "mem" in failed_command.lower():
            alternatives.extend([
                {
                    "command": "top -o mem -l 1 -n 10",
                    "description": "Use 'mem' instead of '%MEM' for macOS top"
                },
                {
                    "command": "ps aux | sort -nrk 4 | head -10",
                    "description": "Use ps with memory sorting"
                },
                {
                    "command": "ps -A -o %mem,%cpu,comm | sort -nr | head -10",
                    "description": "Use ps with memory percentage output"
                }
            ])
        
        # Handle general top command issues
        elif "top" in failed_command and "invalid argument" in error_output:
            alternatives.extend([
                {
                    "command": "top -l 1 -n 10",
                    "description": "Use basic top command"
                },
                {
                    "command": "ps aux | head -15",
                    "description": "Use ps command instead"
                }
            ])
        
        # Handle memory-related requests with fallback commands
        elif any(word in original_request.lower() for word in ['memory', 'mem', 'ram']):
            if self.system_os == 'darwin':
                alternatives.extend([
                    {
                        "command": "ps aux | sort -nrk 4 | head -10",
                        "description": "Show top memory-consuming processes"
                    },
                    {
                        "command": "vm_stat",
                        "description": "Show virtual memory statistics"
                    }
                ])
            elif self.system_os == 'linux':
                alternatives.extend([
                    {
                        "command": "ps aux --sort=-%mem | head -10",
                        "description": "Show top memory-consuming processes"
                    },
                    {
                        "command": "free -h",
                        "description": "Show memory usage"
                    }
                ])
        
        # Handle process listing requests
        elif any(word in original_request.lower() for word in ['process', 'running', 'task']):
            alternatives.extend([
                {
                    "command": "ps aux | head -15",
                    "description": "List running processes"
                },
                {
                    "command": "pgrep -l .",
                    "description": "List all processes with names"
                }
            ])
        
        # Handle installation failures
        elif any(word in failed_command for word in ['brew', 'apt', 'yum', 'pip', 'npm']):
            if 'brew' in failed_command:
                package = failed_command.split()[-1]
                alternatives.extend([
                    {
                        "command": f"brew search {package}",
                        "description": f"Search for {package} in brew"
                    },
                    {
                        "command": f"brew install --cask {package}",
                        "description": f"Try installing {package} as a cask"
                    }
                ])
        
        return alternatives


if __name__ == "__main__":
    terminal = EnhancedAITerminal()
    terminal.run()
