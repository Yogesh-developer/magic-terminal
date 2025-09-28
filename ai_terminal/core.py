#!/usr/bin/env python3
"""
Magic Terminal - A comprehensive, robust terminal assistant
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
import time
import itertools
import psutil
import threading
import shutil
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass
from enum import Enum
from difflib import get_close_matches
from urllib.parse import urlparse

try:
    from .llm_client import LLMClient, PlanStep
    from .config_manager import ConfigManager
    from .history_manager import HistoryManager
    from .http_utils import request_with_retries
    from .safety import CommandAuditor, CommandWarning
except ImportError:  # When executed as a script
    from llm_client import LLMClient, PlanStep
    from config_manager import ConfigManager
    from history_manager import HistoryManager
    from http_utils import request_with_retries
    from safety import CommandAuditor, CommandWarning

try:
    from jsonschema import validate, ValidationError
except ImportError:
    validate = None
    ValidationError = None

# Configuration
LOG_FILE = os.path.expanduser("~/.magic_terminal_logs/enhanced_terminal.log")
HISTORY_FILE = os.path.expanduser("~/.magic_terminal_history")
CONFIG_FILE = os.path.expanduser("~/.magic_terminal_config.json")

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Logging setup
logger = logging.getLogger("magic_terminal")
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
                'winget': {'install': 'winget install {}', 'uninstall': 'winget uninstall {}', 'update': 'winget upgrade --all'},
                'pip': {'install': 'pip install {}', 'uninstall': 'pip uninstall {}', 'update': 'pip install --upgrade {}'}
            }
        }
        self.available_managers = self._detect_available_managers()

    def _detect_available_managers(self) -> List[str]:
        """Detect which package managers are available"""
        available = []
        for manager in self.managers.get(self.system_os, {}):
            if shutil.which(manager):
                available.append(manager)
                continue

            try:
                result = subprocess.run(
                    [manager, '--version'],
                    capture_output=True,
                    timeout=5,
                    check=True,
                )
                if result.returncode == 0:
                    available.append(manager)
            except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError, subprocess.CalledProcessError):
                continue
        return available
    
    def suggest_install_command(self, package: str) -> List[str]:
        """Suggest installation commands for a package"""
        commands: List[str] = []
        for manager in self.available_managers:
            manager_map = self.managers.get(self.system_os, {})
            if manager in manager_map:
                cmd_template = manager_map[manager]['install']
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


class ProgressIndicator:
    """Single-line progress reporter for long-running operations."""

    def __init__(self, *, interval: float = 2.0) -> None:
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._message: str = ""
        self._start_time: float = 0.0
        self._spinner = itertools.cycle("|/-\\")
        self._last_length: int = 0
        self._max_length: int = 0
        self._lock = threading.Lock()
        self._tracker: Optional["DownloadTracker"] = None

    @staticmethod
    def _format_bytes(num_bytes: float) -> str:
        units = ["B", "KB", "MB", "GB", "TB"]
        value = float(num_bytes)
        for unit in units:
            if value < 1024 or unit == units[-1]:
                return f"{value:.1f} {unit}"
            value /= 1024
        return f"{value:.1f} TB"

    def _format_status(self, spinner: str, elapsed: float) -> str:
        if self._tracker:
            downloaded, total, speed, remaining = self._tracker.get_status()
            parts = [f"{self._message} [{spinner}]"]
            if downloaded is not None:
                parts.append(f"{self._format_bytes(downloaded)} downloaded")
            if total:
                progress_pct = (downloaded / total * 100) if downloaded is not None and total > 0 else 0.0
                parts.append(f"of {self._format_bytes(total)} ({progress_pct:.1f}%)")
            if speed:
                parts.append(f"at {self._format_bytes(speed)}/s")
            if remaining:
                parts.append(f"~{remaining:.0f}s remaining")
            parts.append(f"elapsed {elapsed:.1f}s")
            return " ".join(parts)

        return f"{self._message} [{spinner}] elapsed {elapsed:.1f}s"

    def _render_status(self, elapsed: float, spinner: Optional[str] = None) -> None:
        if spinner is None:
            spinner = next(self._spinner)
        status = self._format_status(spinner, elapsed)
        self._max_length = max(self._max_length, len(status))
        with self._lock:
            padding = self._max_length - len(status)
            sys.stdout.write("\r" + status + (" " * padding))
            sys.stdout.flush()
            self._last_length = len(status)

    def _clear_line(self) -> None:
        with self._lock:
            sys.stdout.write("\r" + " " * self._max_length + "\r")
            sys.stdout.flush()
            self._last_length = 0
            self._max_length = 0

    def start(self, message: str, tracker: Optional["DownloadTracker"] = None) -> None:
        if self._thread is not None:
            self.stop()

        self._message = message
        self._start_time = time.time()
        self._spinner = itertools.cycle("|/-\\")
        self._stop_event.clear()
        self._tracker = tracker
        self._max_length = 0

        def _run() -> None:
            while not self._stop_event.wait(self.interval):
                elapsed = time.time() - self._start_time
                self._render_status(elapsed)

        self._render_status(0.0, spinner="*")
        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self, final_message: Optional[str] = None) -> None:
        if self._thread is None:
            if final_message:
                print(final_message)
            return

        self._stop_event.set()
        self._thread.join()
        self._thread = None

        elapsed = time.time() - self._start_time
        self._clear_line()
        if final_message:
            print(f"{final_message} (elapsed {elapsed:.1f}s)")
        self._tracker = None


class DownloadTracker:
    """Abstract base for download progress tracking."""

    def get_status(self) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[float]]:
        raise NotImplementedError


class BrewDownloadTracker(DownloadTracker):
    """Track Homebrew download progress via cache file inspection."""

    def __init__(self, cache_path: Optional[Path], file_name: Optional[str], total_bytes: Optional[int]) -> None:
        self.cache_path = cache_path
        self.file_name = file_name
        self.total_bytes = total_bytes
        self.cache_dirs: List[Path] = []
        if cache_path and cache_path.parent:
            self.cache_dirs.append(cache_path.parent)

        # Mac default
        self.cache_dirs.append(Path.home() / "Library/Caches/Homebrew/downloads")
        # Linux default
        self.cache_dirs.append(Path.home() / ".cache/Homebrew/downloads")
        # Windows default
        local_appdata = os.getenv("LOCALAPPDATA")
        if local_appdata:
            self.cache_dirs.append(Path(local_appdata) / "Homebrew/Cache/Downloads")

        # Ensure uniqueness
        seen: Set[Path] = set()
        unique_dirs: List[Path] = []
        for cache_dir in self.cache_dirs:
            if cache_dir and cache_dir not in seen:
                seen.add(cache_dir)
                unique_dirs.append(cache_dir)
        self.cache_dirs = unique_dirs
        self._last_bytes = 0
        self._last_time = time.time()
        self._speed = 0.0
        self._lock = threading.Lock()
        self._current_path: Optional[Path] = None

    def _resolve_path(self) -> Optional[Path]:
        if self.cache_path and self.cache_path.exists():
            return self.cache_path

        if not self.file_name:
            return None

        for cache_dir in self.cache_dirs:
            if not cache_dir.exists():
                continue
            pattern = f"*{self.file_name}*"
            candidates = [p for p in cache_dir.glob(pattern) if p.is_file()]
            if not candidates:
                stem = Path(self.file_name).stem
                candidates = [p for p in cache_dir.glob(f"*{stem}*") if p.is_file()]
            if candidates:
                return max(candidates, key=lambda p: p.stat().st_mtime)
        return None

    def get_status(self) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[float]]:
        with self._lock:
            now = time.time()
            path = self._resolve_path()

            if not path or not path.exists():
                return None, self.total_bytes, None, None

            if path != self._current_path:
                self._current_path = path
                self._last_bytes = 0
                self._last_time = now
                self._speed = 0.0

            size = path.stat().st_size

            delta_bytes = size - self._last_bytes
            delta_time = now - self._last_time
            if delta_time > 0:
                instant_speed = delta_bytes / delta_time
                if self._speed == 0.0:
                    self._speed = instant_speed
                else:
                    self._speed = 0.7 * self._speed + 0.3 * instant_speed

            self._last_bytes = size
            self._last_time = now

            remaining_time = None
            if self.total_bytes and self._speed > 0:
                remaining_bytes = max(self.total_bytes - size, 0)
                remaining_time = remaining_bytes / self._speed if remaining_bytes > 0 else 0.0

            speed = self._speed if self._speed > 0 else None
            return size, self.total_bytes, speed, remaining_time


class NetworkUsageTracker(DownloadTracker):
    """Cross-platform tracker based on network interface statistics."""

    def __init__(self) -> None:
        self._start = psutil.net_io_counters()
        self._last_time = time.time()
        self._last_bytes = 0
        self._speed = 0.0

    def get_status(self) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[float]]:
        now = time.time()
        counters = psutil.net_io_counters()
        downloaded = counters.bytes_recv - self._start.bytes_recv
        delta_bytes = downloaded - self._last_bytes
        delta_time = now - self._last_time

        if delta_time > 0:
            instant_speed = delta_bytes / delta_time if delta_bytes > 0 else 0.0
            if self._speed == 0.0:
                self._speed = instant_speed
            else:
                self._speed = 0.7 * self._speed + 0.3 * instant_speed

        self._last_bytes = downloaded
        self._last_time = now

        speed = self._speed if self._speed > 0 else None
        return max(downloaded, 0), None, speed, None


class EnhancedAITerminal:
    """Magic Terminal with comprehensive functionality"""
    
    def __init__(self, *, enable_fallback: bool = True):
        self.system_os = platform.system().lower()
        self.shell_prompt = "Magic-Terminal> "
        self.current_dir = os.getcwd()
        self.allow_fallback = enable_fallback
        
        # Configuration & state managers
        self.config_manager = ConfigManager(Path(CONFIG_FILE))
        self.config = self.config_manager.load()
        self.history_manager = HistoryManager(Path(HISTORY_FILE), max_length=self.config.get('max_history', 1000))
        self.history_manager.load()

        # Initialize components
        self.package_manager = PackageManager(self.system_os)
        self.file_ops = FileOperations()
        self.process_manager = ProcessManager()
        self.system_monitor = SystemMonitor()
        self.progress_indicator = ProgressIndicator()
        self.llm_client = self._init_llm_client()
        self.command_auditor = CommandAuditor()

        # LLM settings
        self.ollama_url = "http://localhost:11434"
        self.grok_api_key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self._ollama_cache: Optional[Tuple[float, List[str]]] = None
        self.available_models = self._check_ollama()
        
        # State management
        self._command_history: List[CommandResult] = []
        self._session_context: Dict[str, Any] = {}
        self._bookmarks: Dict[str, str] = self.config.get('bookmarks', {})
        
        # Setup
        self._setup_logging()

        logger.info("Magic Terminal initialized")
        logger.info(f"OS: {platform.system()} | Directory: {self.current_dir}")
        logger.info(f"Available package managers: {self.package_manager.available_managers}")

    def _log_command_event(self, event: str, **data: Any) -> None:
        """Log structured command events as JSON."""
        payload = {"event": event, **data}
        try:
            logger.info(json.dumps(payload))
        except TypeError:
            safe_payload = {key: str(value) for key, value in payload.items()}
            logger.info(json.dumps(safe_payload))

    
    def _save_config(self):
        """Save configuration to file"""
        try:
            self.config_manager.save(self.config)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def _check_ollama(self, *, cache_ttl: float = 60.0) -> List[str]:
        """Check available Ollama models with basic caching and retry logic."""

        now = time.time()
        if self._ollama_cache and now - self._ollama_cache[0] < cache_ttl:
            return list(self._ollama_cache[1])

        models: List[str] = []
        try:
            response = request_with_retries(
                "GET",
                f"{self.ollama_url}/api/tags",
                timeout=3,
                retries=2,
                backoff_factor=0.4,
                logger=logger,
            )
            response.raise_for_status()
            payload = response.json()
            models = [model['name'] for model in payload.get('models', [])]
        except requests.RequestException as exc:
            logger.debug("Ollama availability check failed: %s", exc)

        self._ollama_cache = (now, models)
        return list(models)
    
    def _setup_logging(self):
        """Setup enhanced logging"""
        pass  # Already configured globally
    
    def execute_command(self, command: str, working_dir: Optional[str] = None) -> CommandResult:
        """Execute a shell command with enhanced error handling"""
        start_time = time.time()
        work_dir = working_dir or self.current_dir

        self._log_command_event("command_start", command=command, cwd=work_dir)

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            duration = time.time() - start_time
            cmd_result = CommandResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr,
                duration=duration,
                command=command,
            )

            self._command_history.append(cmd_result)
            self._log_command_event(
                "command_finish",
                command=command,
                cwd=work_dir,
                success=cmd_result.success,
                exit_code=result.returncode,
                duration=cmd_result.duration,
            )
            return cmd_result

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            self._log_command_event(
                "command_timeout",
                command=command,
                cwd=work_dir,
                duration=duration,
            )
            return CommandResult(
                success=False,
                output="",
                error="Command timed out after 5 minutes",
                duration=duration,
                command=command,
            )
        except Exception as e:
            duration = time.time() - start_time
            self._log_command_event(
                "command_error",
                command=command,
                cwd=work_dir,
                error=str(e),
                duration=duration,
            )
            return CommandResult(
                success=False,
                output="",
                error=str(e),
                duration=duration,
                command=command,
            )
    
    def run(self):
        """Main terminal loop"""
        print("Magic Terminal")
        print("Type 'help' for commands, 'exit' to quit")
        print("Advanced operations: install, delete, create, monitor, etc.")
        
        while True:
            try:
                user_input = input(f"\n{self.shell_prompt}").strip()
                
                if not user_input:
                    continue
                
                self.history_manager.add_entry(user_input)
                
                if user_input.lower() in ['exit', 'quit']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                # Process the command
                self._process_user_input(user_input)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                logger.error(f"Unexpected error: {e}")
    
    def _show_help(self):
        """Show help information"""
        help_text = """
Magic Terminal Commands:

Package Management:
  install <package>      - Install software package
  uninstall <package>    - Remove software package
  update                 - Update all packages using your preferred manager

File Operations:
  create <type> <name>   - Create file from template library
  delete <path>          - Safely delete files/folders with trash support
  mkdir <path>           - Create directories
  
Process Management:
  ps [filter]            - List running processes
  kill <pid/name>        - Terminate process (with confirmation)
  services               - List running services

System Monitoring:
  status                 - Show system resource usage snapshot
  monitor                - Start real-time monitoring session
  logs <file>            - Analyze log files for common issues

Navigation & Bookmarks:
  cd <path>              - Change directory
  bookmark <name>        - Bookmark current directory
  goto <bookmark>        - Jump to bookmarked directory

Configuration & History:
  config                 - Show resolved configuration
  alias <name> <cmd>     - Create command alias
  history search <term>  - Recall prior commands (readline required)

Safety & Confirmation:
  Commands are audited for destructive patterns.
  Use `config` to enable `auto_confirm_safe`; high-risk commands still prompt.

Natural Language:
  Describe goals in plain English!
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
                print("Unable to understand the request")
                return
            
            print(f"{command_info.get('description', 'Executing command')}")
            
            # Execute commands
            success = self._execute_commands(command_info, user_input)
            if success:
                print("Done")
            else:
                print("Failed")
                
        except Exception as e:
            print(f"Error: {e}")
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
                print(f"Warning: LLM response invalid (attempt {attempt}/{max_attempts}). Retrying...")
            except Exception as exc:
                last_error = exc
                logger.warning(f"LLM call failed (attempt {attempt}/{max_attempts}): {exc}")
                print(f"Warning: LLM call failed (attempt {attempt}/{max_attempts}). Retrying...")
        
        if self.allow_fallback:
            logger.warning(f"LLM unavailable after retries ({last_error}). Using fallback.")
            print("Warning: Using fallback commands instead.")
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
        
        logger.info(f"Ollama analyzing: '{user_input}'")
        print(f"Analyzing: '{user_input}'")
        
        response = request_with_retries(
            "POST",
            f"{self.ollama_url}/api/chat",
            json=payload,
            timeout=45,
            retries=2,
            backoff_factor=0.5,
            logger=logger,
        )
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
        
        logger.info(f"OpenAI analyzing: '{user_input}'")
        print(f"Analyzing: '{user_input}'")
        
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
        
        logger.info(f"Grok analyzing: '{user_input}'")
        print(f"Analyzing: '{user_input}'")
        
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
            print("LLM assistance unavailable. Manual intervention required.")
            return False

        print("Escalating failure to LLM for guidance...")

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
            print(f"LLM recovery attempt failed: {exc}")
            return False

        if not fix_steps:
            print("LLM did not provide any recovery steps. Manual intervention required.")
            return False

        print("LLM suggested recovery plan:")
        for i, step in enumerate(fix_steps, start=1):
            desc = step.description or step.command
            print(f"  {i}. {desc}")

        # Allow user to select specific steps or execute all
        print()
        choice = input("Execute: [a]ll steps, [1-9] specific step, or [n]one? ").strip().lower()
        
        if choice in {"n", "no", "none"}:
            print("Recovery plan rejected by user. Manual intervention required.")
            return False
        
        steps_to_execute = []
        if choice in {"a", "all", "y", "yes"}:
            steps_to_execute = fix_steps
        elif choice.isdigit():
            step_num = int(choice)
            if 1 <= step_num <= len(fix_steps):
                steps_to_execute = [fix_steps[step_num - 1]]
            else:
                print(f"Invalid step number {step_num}. Must be between 1 and {len(fix_steps)}.")
                return False
        else:
            print("Invalid choice. Recovery plan aborted.")
            return False

        recovery_successful = False
        for idx, step in enumerate(steps_to_execute, start=1):
            print(f"LLM step {idx}/{len(steps_to_execute)}: {step.command}")
            result = self.execute_command(step.command, working_dir)

            if result.output:
                print(result.output)
                # If recovery step produced output, consider it successful
                recovery_successful = True
            if result.error:
                print(f"Warning: {result.error}")

            if not result.success:
                print("LLM recovery step failed. Manual intervention required.")
                return False

            executed_steps.append(
                PlanStep(
                    step=len(executed_steps) + 1,
                    command=step.command,
                    description=step.description or "LLM recovery step",
                )
            )

        # Only retry original command if it makes sense to do so
        # Don't retry if recovery steps already provided the answer
        if self._should_retry_original_command(failed_command, steps_to_execute, recovery_successful):
            print("Retrying original command after LLM fixes...")
            retry_result = self.execute_command(failed_command, working_dir)

            if retry_result.output:
                print(retry_result.output)
            if retry_result.error:
                print(f"Warning: {retry_result.error}")

            if not retry_result.success:
                print("Command still failing after LLM recovery. Manual intervention required.")
                return False
        else:
            print("Recovery steps completed successfully!")
            return True

        executed_steps.append(
            PlanStep(
                step=len(executed_steps) + 1,
                command=failed_command,
                description="Command retried after LLM fix",
            )
        )

        print("Command succeeded after LLM-guided recovery.")
        return True

    def _should_retry_original_command(self, failed_command: str, recovery_steps: List[PlanStep], recovery_successful: bool) -> bool:
        """Determine if we should retry the original command after recovery steps."""
        
        # If recovery didn't produce any output, we should retry
        if not recovery_successful:
            return True
            
        failed_cmd_lower = failed_command.lower()
        
        # Don't retry if the recovery steps were alternative commands that already provided the answer
        for step in recovery_steps:
            step_cmd_lower = step.command.lower()
            
            # If recovery step was an alternative version command (python3 vs python), don't retry original
            if ("--version" in failed_cmd_lower and "--version" in step_cmd_lower) or \
               ("version" in failed_cmd_lower and "version" in step_cmd_lower):
                return False
                
            # If recovery step was checking availability (which, type), don't retry original
            if any(cmd in step_cmd_lower for cmd in ["which", "type", "whereis"]):
                return False
                
            # If recovery step was an alternative info command (system_profiler, sysctl), don't retry original
            if any(cmd in step_cmd_lower for cmd in ["system_profiler", "sysctl", "log show"]):
                return False
        
        # For other cases, it might make sense to retry (e.g., after installing missing packages)
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
        helper_script = Path.home() / ".magic_terminal_user_listing.py"
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
            "commands": [f"python3 ~/.magic_terminal_user_listing.py"],
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
        
        print(f"Working Directory: {working_dir}")
        print("Commands:")
        for cmd in commands:
            print(f"  {cmd}")
        
        # Ask for confirmation
        if not self.config.get('auto_confirm_safe', False):
            confirm = input("Execute these commands? [y/N]: ").strip().lower()
            if confirm not in {"y", "yes"}:
                print("Execution cancelled.")
                return False
        
        executed_steps: List[PlanStep] = []

        # Execute commands
        for i, cmd in enumerate(commands):
            print(f"Running step {i+1}/{len(commands)}: {cmd}")
            
            # Check if it's an installation command
            install_target = self._extract_install_target(cmd)
            start_time = time.time() if install_target else None
            tracker: Optional[DownloadTracker] = None
            
            if install_target:
                if self._is_already_installed(cmd, install_target):
                    print(f"{install_target} is already installed. Skipping command.")
                    executed_steps.append(
                        PlanStep(
                            step=len(executed_steps) + 1,
                            command=cmd,
                            description=f"Skipped installation; {install_target} already present",
                        )
                    )
                    continue
                tracker = self._create_download_tracker(cmd, install_target)
                self.progress_indicator.start(f"Installing {install_target}...", tracker=tracker)

            result = self.execute_command(cmd, working_dir)

            if install_target and start_time:
                duration = time.time() - start_time
                status = "completed" if result.success else "failed"
                self.progress_indicator.stop(f"Installation step for {install_target} {status}")
                print(f"Installation step for {install_target} finished in {duration:.1f}s")

            if result.output:
                print(result.output)
            if result.error:
                print(f"Warning: {result.error}")

            if result.success:
                executed_steps.append(
                    PlanStep(
                        step=len(executed_steps) + 1,
                        command=cmd,
                        description="Executed successfully",
                    )
                )
                continue

            print(f"Command failed: {cmd}")

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

            print("Manual intervention required. Aborting remaining steps.")
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

    def _create_download_tracker(self, command: str, target: str) -> Optional[DownloadTracker]:
        """Create download tracker for known package managers."""
        tracker: Optional[DownloadTracker] = None

        if command.strip().startswith("brew install"):
            try:
                info = subprocess.run(
                    ["brew", "info", "--json=v2", target],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                url = None
                is_cask = False
                if info.returncode == 0 and info.stdout:
                    data = json.loads(info.stdout)
                    if data.get("casks"):
                        cask = data["casks"][0]
                        url = cask.get("url")
                        is_cask = True
                    elif data.get("formulae"):
                        formula = data["formulae"][0]
                        url = formula.get("urls", {}).get("stable", {}).get("url")

                cache_cmd = ["brew", "--cache"]
                if is_cask:
                    cache_cmd.append("--cask")
                cache_cmd.append(target)
                cache_run = subprocess.run(
                    cache_cmd,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                cache_path = None
                if cache_run.returncode == 0 and cache_run.stdout:
                    potential_path = Path(cache_run.stdout.strip())
                    cache_path = potential_path if potential_path.exists() else None

                file_name = None
                if url:
                    parsed = urlparse(url)
                    file_name = Path(parsed.path).name or None
                elif cache_path:
                    file_name = cache_path.name

                total_bytes = None
                if url:
                    try:
                        response = requests.head(url, allow_redirects=True, timeout=5)
                        content_length = response.headers.get("Content-Length")
                        if content_length and content_length.isdigit():
                            total_bytes = int(content_length)
                    except Exception:
                        pass
                tracker = BrewDownloadTracker(cache_path, file_name, total_bytes)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Unable to build brew tracker for {target}: {exc}")

        if not tracker:
            tracker = NetworkUsageTracker()

        return tracker

    def _attempt_intelligent_recovery(self, failed_command: str, error_output: str, original_request: str) -> bool:
        """Attempt intelligent recovery by analyzing the error and finding alternatives"""
        print("Analyzing error and attempting recovery...")
        # Analyze the error and suggest alternatives
        alternatives = self._analyze_error_and_suggest_alternatives(failed_command, error_output, original_request)
        if not alternatives:
            print("No recovery options found")
            return False

        print("Found alternative solutions:")
        for i, alt in enumerate(alternatives, 1):
            print(f"  {i}. {alt['description']}")

        # Try alternatives automatically
        for alt in alternatives:
            print(f"Trying: {alt['command']}")
            result = self.execute_command(alt['command'])

            if result.success:
                if result.output:
                    print(result.output)
                print("Recovery successful!")
                return True
            else:
                print(f"Warning: Alternative failed: {result.error}")

        print("All recovery attempts failed")
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
