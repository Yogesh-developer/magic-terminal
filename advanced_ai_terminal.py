#!/usr/bin/env python3
import os
import sys
import subprocess
import platform
import json
import textwrap
import logging
import requests
import re
import glob
import shlex
import atexit
import time
from difflib import get_close_matches
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

try:
    from jsonschema import validate, ValidationError
except ImportError:  # pragma: no cover - optional dependency
    validate = None
    ValidationError = None

try:
    import readline  # noqa: F401  # Unix-like systems
except ImportError:  # pragma: no cover
    readline = None


LOG_FILE = os.path.expanduser("~/.ai_terminal_logs/ai_terminal.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

HISTORY_FILE = os.path.expanduser("~/.ai_terminal_history")

logger = logging.getLogger("advanced_ai_terminal")
if not logger.handlers:
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))

    logger.addHandler(file_handler)
    logger.propagate = False


LLM_COMMAND_SCHEMA = {
    "type": "object",
    "properties": {
        "commands": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "description": {"type": "string"},
        "confidence": {"type": "number"},
        "type": {"type": "string"},
        "working_directory": {"type": "string"},
    },
    "required": ["commands"],
}

DANGEROUS_PATTERNS = [
    r"\brm\s+-rf\b",
    r"\brm\s+-r\b",
    r"\bdd\b",
    r">\s*/dev/sd",
    r">\s*/dev/nvme",
    r"\b:>\b",
    r"\bmkfs\b",
    r"\bfdisk\b",
    r"\bpowershell\s+-Command\s+Remove-Item\b",
]


class AdvancedAITerminal:
    GROK_ENDPOINT = "https://api.x.ai/v1/chat/completions"

    def __init__(self, *, enable_fallback: bool = False):
        self.system_os = platform.system().lower()
        self.shell_prompt = "AI-Terminal> "
        self.ollama_url = "http://localhost:11434"
        self.grok_api_key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
        self.current_dir = os.getcwd()
        env_fallback = os.getenv("AI_TERMINAL_ALLOW_FALLBACK")
        if env_fallback is not None:
            self.allow_fallback = env_fallback == "1"
        else:
            self.allow_fallback = enable_fallback
        self.available_models = self.check_ollama()
        self._file_cache: Dict[tuple, str] = {}
        self._recovery_depth = 0
        self._recovery_limit = int(os.getenv("AI_TERMINAL_RECOVERY_LIMIT", "2"))
        self._recovery_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._global_recovery_attempts: Dict[Tuple[str, str], bool] = {}
        self._last_execution_cancelled = False

        logger.info("🚀 Advanced AI Terminal with File Execution")
        logger.info(f"💻 OS: {platform.system()} | Directory: {self.current_dir}")

        self._setup_history()

        self._announce_backend()

    def _announce_backend(self) -> None:
        if self.grok_api_key:
            logger.info("🧠 LLM Backend: Grok (remote)")
        elif self.available_models:
            logger.info("🧠 LLM Backend: Ollama (%s)", ", ".join(self.available_models))
        else:
            message = (
                "⚠️  No LLM backend detected. "
                "Set XAI_API_KEY/GROK_API_KEY for Grok or run Ollama locally."
            )
            logger.warning(message)
            if not self.allow_fallback:
                logger.error("❗ Set AI_TERMINAL_ALLOW_FALLBACK=1 to enable rule-based fallback.")

    def check_ollama(self) -> list:
        """Check if Ollama is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
        except Exception:
            pass
        return []

    def _setup_history(self) -> None:
        if not readline:
            return

        try:
            if os.path.exists(HISTORY_FILE):
                readline.read_history_file(HISTORY_FILE)
        except (FileNotFoundError, PermissionError):
            pass

        try:
            readline.set_history_length(1000)
        except Exception:  # noqa: BLE001
            pass

        atexit.register(self._save_history)

    def _save_history(self) -> None:
        if not readline:
            return

        try:
            readline.write_history_file(HISTORY_FILE)
        except (FileNotFoundError, PermissionError):
            pass

    def understand_complex_command(self, user_input: str) -> Dict[str, Any]:
        """Use AI to understand complex commands including file execution"""
        max_attempts = 3
        attempt = 0
        last_error: Optional[Exception] = None

        while attempt < max_attempts:
            attempt += 1
            try:
                return self.call_local_llm(user_input)
            except ValueError as exc:
                last_error = exc
                logger.warning("⚠️  LLM response invalid (attempt %s/%s): %s", attempt, max_attempts, exc)
                print(f"⚠️  LLM response invalid (attempt {attempt}/{max_attempts}). Retrying...")
            except Exception as exc:
                last_error = exc
                logger.warning("⚠️  LLM call failed (attempt %s/%s): %s", attempt, max_attempts, exc)
                print(f"⚠️  LLM call failed (attempt {attempt}/{max_attempts}). Retrying...")

        if self.allow_fallback:
            logger.warning("⚠️  LLM unavailable after retries (%s). Falling back to heuristics.", last_error)
            print(f"⚠️  LLM unavailable after retries ({last_error}). Using fallback commands instead.")
            return self.advanced_fallback(user_input)

        raise last_error or ValueError("LLM response did not contain valid JSON commands")

    def call_local_llm(self, user_input: str) -> Dict[str, Any]:
        """Call preferred LLM backend with enhanced understanding of file operations"""
        lower_input = user_input.lower()
        if any(phrase in lower_input for phrase in ['list users', 'list user directories', 'show users']) or (
            'user' in lower_input and any(word in lower_input for word in ['list', 'show'])):
            logger.info("ℹ️  Bypassing LLM for user directory listing request.")
            return self._handle_user_directory_listing(user_input)

        if self.grok_api_key:
            llm_response = self._call_grok(user_input)
        else:
            llm_response = self._call_ollama(user_input)

        return self._parse_enhanced_response(llm_response, user_input)

    def _call_grok(self, user_input: str) -> str:
        logger.info("🧠 Grok analyzing: '%s'", user_input)
        print(f"🧠 Grok analyzing: '{user_input}'")
        headers = {
            "Authorization": f"Bearer {self.grok_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "grok-1",
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": self._get_enhanced_system_prompt()},
                {"role": "user", "content": user_input}
            ]
        }
        response = requests.post(self.GROK_ENDPOINT, headers=headers, json=payload, timeout=45)
        if not response.ok:
            logger.warning("⚠️  Grok API error:\n  Status: %s\n  Body: %s", response.status_code, response.text)
            response.raise_for_status()
        data = response.json()
        logger.debug("🧾 Grok raw response:\n%s", json.dumps(data, indent=2))
        try:
            content = data["choices"][0]["message"]["content"]
            self._display_llm_console(content)
            return content
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected Grok response: {data}") from exc

    def _call_ollama(self, user_input: str) -> str:
        if not self.available_models:
            raise RuntimeError("No Ollama models detected")

        if any("codellama" in model for model in self.available_models):
            model = next(model for model in self.available_models if "codellama" in model)
        else:
            model = self.available_models[0]

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self._get_enhanced_system_prompt()},
                {"role": "user", "content": user_input}
            ],
            "stream": False,
            "options": {"temperature": 0.1}
        }

        logger.info("🧠 Ollama analyzing: '%s'", user_input)
        print(f"🧠 Ollama analyzing: '{user_input}'")
        response = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=45)
        if not response.ok:
            logger.warning("⚠️  Ollama API error:\n  Status: %s\n  Body: %s", response.status_code, response.text)
            response.raise_for_status()
        data = response.json()
        logger.debug("🧾 Ollama raw response:\n%s", json.dumps(data, indent=2))
        try:
            content = data["message"]["content"]
            self._display_llm_console(content)
            return content
        except KeyError as exc:
            raise RuntimeError(f"Unexpected Ollama response: {data}") from exc

    def _get_enhanced_system_prompt(self) -> str:
        """Enhanced system prompt for file execution and path handling"""
        current_files = "\n".join([f"  - {f}" for f in os.listdir(self.current_dir)[:10]])

        return f"""You are an advanced AI terminal assistant. Convert natural language to executable commands.

CURRENT SYSTEM: {self.system_os.upper()}
CURRENT DIRECTORY: {self.current_dir}
FILES IN CURRENT DIRECTORY:
{current_files}

SPECIAL CAPABILITIES:
1. Run Python scripts: "python script.py" or "python3 script.py"
2. Execute shell scripts: "./script.sh" or "bash script.sh"
3. Change directories: "cd /path/to/directory"
4. Execute files with relative/absolute paths
5. Handle file operations with specific paths
6. When the user references globally installed software or system tooling, prefer the native CLI directly (for example, respond with `<tool> --version` for version checks, or other appropriate subcommands).

COMMAND EXAMPLES:
- "run the python file called example.py" → "python3 example.py"
- "execute the script in the src folder" → "cd src && python3 script.py"
- "run my program from the desktop" → "cd ~/Desktop && python3 program.py"
- "execute the test script with arguments" → "python3 test.py arg1 arg2"

ALWAYS return valid JSON. Each entry in "commands" MUST be a single shell-ready string. If multiple commands are required, list them in execution order as separate strings.
{{
    "commands": [
        "first command string",
        "second command string if necessary"
    ],
    "description": "what will be executed",
    "confidence": 0.0-1.0,
    "type": "python_script|shell_script|file_operation|system",
    "working_directory": "optional/specific/path"  // if different from current
}}

Do NOT split command arguments into separate list items.

Respond to the user's request:"""

    def _parse_enhanced_response(self, llm_response: str, user_input: str) -> Dict[str, Any]:
        """Parse LLM response with enhanced error handling"""
        try:
            command_info = json.loads(llm_response)
        except (json.JSONDecodeError, TypeError):
            command_info = self._extract_json_block(llm_response)

        if not command_info:
            raise ValueError("LLM response did not contain valid JSON commands")

        command_info.setdefault("llm_raw", llm_response)
        if not self._validate_llm_command_info(command_info):
            raise ValueError("LLM response failed validation")
        return self._enhance_with_file_detection(command_info, user_input)

    def _extract_json_block(self, llm_response: str) -> Optional[Dict[str, Any]]:
        json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
        if not json_match:
            return None
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            return None

    def _display_llm_console(self, content: str) -> None:
        divider = "-" * 40
        cleaned = content.strip() if content else "(no content)"
        print(f"\n🖥️ LLM Console\n{divider}\n{cleaned}\n{divider}\n")

    def _enhance_with_file_detection(self, command_info: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """Enhance AI response with actual file detection"""
        commands = command_info.get("commands", [])
        enhanced_commands = []

        for cmd in commands:
            enhanced_cmd = self._resolve_file_paths(cmd, user_input)
            enhanced_commands.append(enhanced_cmd)

        command_info["commands"] = enhanced_commands
        return command_info

    def _resolve_file_paths(self, command: str, user_input: str) -> str:
        """Resolve file paths in commands"""
        # Look for file patterns in the command
        file_patterns = [
            r'python3?\s+([a-zA-Z0-9_\-\./]+\.[a-zA-Z0-9]+)',
            r'\./([a-zA-Z0-9_\-\./]+\.sh)',
            r'bash\s+([a-zA-Z0-9_\-\./]+\.sh)',
            r'cd\s+([a-zA-Z0-9_\-\./]+)',
            r'(?:cat|less|more|tail|head)\s+([a-zA-Z0-9_\-\./]+\.[a-zA-Z0-9]+)'
        ]

        for pattern in file_patterns:
            match = re.search(pattern, command)
            if match:
                filename = match.group(1)
                # Check if file exists, try to find it
                found_path = self._find_file(filename)
                if found_path and os.path.exists(found_path):
                    replacement = self._quote_path(found_path)
                    command = command.replace(filename, replacement)

        return command

    def _find_file(self, filename: str) -> str:
        """Find a file in current directory or subdirectories with fuzzy matching"""
        filename = filename.strip()
        if not filename:
            return filename

        cache_key = (self.current_dir, filename)
        if cache_key in self._file_cache:
            return self._file_cache[cache_key]

        candidate_path = Path(filename).expanduser()
        if candidate_path.is_absolute():
            if candidate_path.exists():
                resolved = str(candidate_path)
                self._file_cache[cache_key] = resolved
                return resolved
            self._file_cache[cache_key] = str(candidate_path)
            return str(candidate_path)

        direct_path = Path(self.current_dir) / candidate_path
        if direct_path.exists():
            resolved = str(direct_path.resolve())
            self._file_cache[cache_key] = resolved
            return resolved

        search_root = Path(self.current_dir)
        name_only = candidate_path.name
        direct_matches = [p for p in search_root.rglob(name_only) if p.is_file()]
        if direct_matches:
            direct_matches.sort(key=lambda p: (len(str(p)), len(p.parents)))
            resolved = str(direct_matches[0].resolve())
            self._file_cache[cache_key] = resolved
            return resolved

        all_files = [str(p.resolve()) for p in search_root.rglob('*') if p.is_file()]
        matches = get_close_matches(filename, all_files, n=1, cutoff=0.6)
        if matches:
            resolved = matches[0]
            self._file_cache[cache_key] = resolved
            return resolved

        for path in all_files:
            if name_only.lower() in Path(path).name.lower():
                self._file_cache[cache_key] = path
                return path

        self._file_cache[cache_key] = filename
        return filename

    def advanced_fallback(self, user_input: str) -> Dict[str, Any]:
        """Advanced fallback with better file execution understanding"""
        input_lower = user_input.lower()

        # === FILE CREATION ===
        if 'create' in input_lower and ('file' in input_lower or '.json' in input_lower or '.txt' in input_lower):
            creation_result = self._handle_file_creation(user_input)
            if creation_result:
                return creation_result

        # === FILE EXECUTION PATTERNS ===
        if any(word in input_lower for word in ['run', 'execute', 'launch', 'start']):
            return self._handle_execution_commands(input_lower, user_input)

        # === USER DIRECTORY LISTING ===
        elif any(phrase in input_lower for phrase in ['list', 'show']) and 'user' in input_lower:
            listing_result = self._handle_user_directory_listing(user_input)
            if listing_result:
                return listing_result

        # === FILE OPERATIONS ===
        elif any(phrase in input_lower for phrase in ['list files', 'show files', 'ls', 'dir']):
            cmd = "dir" if self.system_os == "windows" else "ls -la"
            return {
                "commands": [cmd],
                "description": "List directory contents",
                "type": "file_operation"
            }

        # === SYSTEM CHECKS ===
        elif 'python' in input_lower and 'check' in input_lower:
            return {
                "commands": ["python3 --version"],
                "description": "Check Python installation",
                "type": "system"
            }

        # === DEFAULT ===
        else:
            return {
                "commands": [f"echo 'Unable to interpret request: {shlex.quote(user_input)}'"],
                "description": "Unable to interpret request",
                "type": "info"
            }

    def _handle_user_directory_listing(self, original_input: str) -> Dict[str, Any]:
        """Handle requests to list user directories and inspect their contents"""

        users_root = Path("C:/Users") if self.system_os == "windows" else Path("/Users")
        if not users_root.exists():
            return {
                "commands": [f"echo 'User root {users_root} not found'"],
                "description": "User root directory not available",
                "type": "info"
            }

        helper_script = Path.home() / ".ai_terminal_user_listing.py"
        helper_script.parent.mkdir(parents=True, exist_ok=True)
        helper_script.write_text(
            "import sys\n"
            "from pathlib import Path\n"
            f"users_root = Path(r'{users_root}')\n"
            "dirs = sorted([p for p in users_root.iterdir() if p.is_dir()])\n"
            "if not dirs:\n"
            f"    print('No folders found under {users_root}')\n"
            "    sys.exit(0)\n"
            "print('Available user directories:')\n"
            "for idx, path in enumerate(dirs, 1):\n"
            "    print(f'{idx}. {path.name}')\n"
            "choice = input('Select a user directory #: ').strip()\n"
            "if not choice.isdigit() or not (1 <= int(choice) <= len(dirs)):\n"
            "    print('Invalid selection.')\n"
            "    sys.exit(1)\n"
            "selected = dirs[int(choice) - 1]\n"
            "print(f\"\\nContents of {selected}:\\n\")\n"
            "for entry in sorted(selected.iterdir()):\n"
            "    suffix = '/' if entry.is_dir() else ''\n"
            "    print(f'{entry.name}{suffix}')\n"
        )

        python_cmd = "python" if self.system_os == "windows" else "python3"
        command = f"{python_cmd} {shlex.quote(str(helper_script))}"

        return {
            "commands": [command],
            "description": f"List folders under {users_root} and display selected contents",
            "type": "file_operation",
            "working_directory": str(users_root)
        }

    def _handle_file_creation(self, original_input: str) -> Dict[str, Any]:
        """Handle natural language file creation requests"""
        normalized = original_input
        for ext in ("json", "txt", "md", "py", "sh", "log", "csv"):
            normalized = normalized.replace(f",{ext}", f".{ext}")

        file_matches = re.findall(r'([\w~/\\\-\.]+\.[A-Za-z0-9]+)', normalized)
        if not file_matches:
            return {}

        raw_target = file_matches[0].strip().strip('\"\'')
        target_path = Path(raw_target).expanduser()

        location_fragment = ''
        location_match = re.search(r'(?:in|inside|into)\s+(?:the\s+)?([\w\-/\\\. ]+)', normalized, re.IGNORECASE)
        if location_match:
            location_fragment = location_match.group(1)

        if not target_path.is_absolute():
            if location_fragment:
                target_directory = self._resolve_user_directory(location_fragment)
                target_path = target_directory / Path(raw_target).name
            else:
                target_directory = Path(self.current_dir)
                target_path = target_directory / raw_target
        else:
            target_directory = target_path.parent

        python_cmd = "python" if self.system_os == "windows" else "python3"
        code = (
            "from pathlib import Path; "
            "path = Path(r'" + str(target_path) + "'); "
            "path.parent.mkdir(parents=True, exist_ok=True); "
            "path.touch(exist_ok=True)"
        )
        command = f"{python_cmd} -c {shlex.quote(code)}"

        return {
            "commands": [command],
            "description": f"Create file at {target_path}",
            "type": "file_operation",
            "working_directory": str(target_path.parent)
        }

    def _resolve_user_directory(self, fragment: str) -> Path:
        """Resolve a natural-language directory fragment to an absolute path"""
        if not fragment:
            return Path(self.current_dir)

        fragment_lower = fragment.lower()
        home = Path.home()
        known_locations = {
            'desktop': home / 'Desktop',
            'documents': home / 'Documents',
            'downloads': home / 'Downloads',
            'pictures': home / 'Pictures',
            'music': home / 'Music',
            'movies': home / 'Movies'
        }

        for key, path in known_locations.items():
            if key in fragment_lower:
                return path

        cleaned_fragment = re.sub(r'\b(folder|directory)\b', '', fragment, flags=re.IGNORECASE)
        cleaned_fragment = re.sub(r'\b(which|that)\b.*', '', cleaned_fragment, flags=re.IGNORECASE).strip()
        cleaned_fragment = cleaned_fragment.replace('\\', '/').strip()

        if re.match(r'^(users?|/users?)/', cleaned_fragment.lower()):
            cleaned_fragment = '/' + cleaned_fragment.lstrip('/ ')

        if cleaned_fragment:
            fuzzy_match = self._fuzzy_resolve_path(cleaned_fragment)
            if fuzzy_match:
                return fuzzy_match

        expanded = Path(cleaned_fragment).expanduser()
        if not expanded.is_absolute():
            expanded = Path(self.current_dir) / expanded

        return expanded.resolve()

    def _fuzzy_resolve_path(self, fragment: str) -> Optional[Path]:
        """Attempt to resolve a directory fragment using fuzzy matching."""

        if not fragment:
            return None

        # Break fragment into path components and attempt fuzzy matching per level
        tokens = [token for token in re.split(r'[\\/]+', fragment) if token]
        if not tokens:
            return None

        candidate_roots: List[Path] = [Path(self.current_dir), Path.home()]
        users_root = Path('C:/Users') if self.system_os == 'windows' else Path('/Users')
        if users_root.exists():
            candidate_roots.append(users_root)

        for root in candidate_roots:
            try:
                resolved = root.resolve()
            except FileNotFoundError:
                continue

            current = resolved
            matched_all = True
            for token in tokens:
                try:
                    entries = [p for p in current.iterdir() if p.is_dir()]
                except (PermissionError, FileNotFoundError):
                    matched_all = False
                    break

                names = [p.name for p in entries]
                match = get_close_matches(token, names, n=1, cutoff=0.6)
                if not match:
                    matched_all = False
                    break

                current = next(p for p in entries if p.name == match[0])

            if matched_all:
                return current

        return None

    def _handle_execution_commands(self, input_lower: str, original_input: str) -> Dict[str, Any]:
        """Handle file execution commands"""
        # Extract filename and type
        patterns = [
            r'(run|execute|launch)\s+(?:the\s+)?(python\s+)?(?:file|script)\s+(?:called|named)?\s*[\'\"]?([a-zA-Z0-9_\-\.]+\.py)[\'\"]?',
            r'(run|execute)\s+([a-zA-Z0-9_\-\.]+\.py)',
            r'(run|execute)\s+([a-zA-Z0-9_\-\.]+\.sh)',
            r'python3?\s+([a-zA-Z0-9_\-\.]+\.py)'
        ]

        for pattern in patterns:
            match = re.search(pattern, input_lower)
            if match:
                filename = match.group(2) if match.lastindex >= 2 else match.group(1)
                return self._create_execution_command(filename, original_input)

        # Fallback: try to extract any .py or .sh file mentioned
        file_match = re.search(r'([a-zA-Z0-9_\-\.]+\.(py|sh))', input_lower)
        if file_match:
            filename = file_match.group(1)
            return self._create_execution_command(filename, original_input)

        # If no specific file mentioned, list available executable files
        return self._suggest_executable_files(original_input)

    def _create_execution_command(self, filename: str, original_input: str) -> Dict[str, Any]:
        """Create execution command for a file"""
        actual_path = os.path.abspath(self._find_file(filename))

        if actual_path and os.path.exists(actual_path):
            command_path = self._quote_path(actual_path)
            working_directory = os.path.dirname(actual_path) or self.current_dir

            if filename.lower().endswith('.py'):
                python_cmd = "python" if self.system_os == "windows" else "python3"
                cmd = f"{python_cmd} {command_path}"
                desc = f"Execute Python script: {actual_path}"
            elif filename.lower().endswith('.sh'):
                shell_cmd = "bash" if self.system_os != "windows" else "bash"
                cmd = f"{shell_cmd} {command_path}"
                desc = f"Execute shell script: {actual_path}"
            else:
                cmd = command_path if self.system_os == "windows" else f"./{command_path}"
                desc = f"Execute file: {actual_path}"

            return {
                "commands": [cmd],
                "description": desc,
                "type": "script_execution",
                "working_directory": working_directory
            }

        return {
            "commands": [f"echo 'File {filename} not found'", "ls -la *.py *.sh 2>/dev/null || dir *.py *.sh"],
            "description": f"File not found, showing available files",
            "type": "file_search"
        }

    def _suggest_executable_files(self, original_input: str) -> Dict[str, Any]:
        """Suggest executable files when none specified"""
        # Find Python and shell files
        py_files = glob.glob('*.py')
        sh_files = glob.glob('*.sh')

        if py_files or sh_files:
            suggestion = "Available files: " + ", ".join(py_files + sh_files)
            return {
                "commands": [f"echo '{suggestion}'", "ls -la *.py *.sh 2>/dev/null || dir *.py *.sh"],
                "description": "Show available executable files",
                "type": "suggestion"
            }
        else:
            return {
                "commands": [f"echo 'No .py or .sh files found in current directory'", "ls -la"],
                "description": "No executable files found",
                "type": "info"
            }

    def _validate_llm_command_info(self, command_info: Dict[str, Any]) -> bool:
        if not validate:
            return True
        try:
            validate(instance=command_info, schema=LLM_COMMAND_SCHEMA)
            return True
        except ValidationError as exc:
            logger.warning("⚠️  LLM response failed schema validation: %s", exc.message)
            return False

    def _is_potentially_destructive(self, command: str) -> Optional[str]:
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return pattern
        return None

    def _extract_requested_files(self, user_input: Optional[str]) -> Dict[str, Tuple[str, Set[str]]]:
        if not user_input:
            return {}

        matches = re.findall(r'([~\w\-/\\\.]+\.[A-Za-z0-9]+)', user_input)
        requested: Dict[str, Tuple[str, Set[str]]] = {}

        for token in matches:
            cleaned = token.strip().strip('\"\'')
            if not cleaned:
                continue
            name = Path(cleaned).name
            parts = name.split('.')
            if len(parts) < 2:
                continue
            base = parts[0].lower()
            exts = {part.lower() for part in parts[1:] if part}
            if not exts:
                continue
            if base in requested:
                existing_token, existing_exts = requested[base]
                requested[base] = (existing_token, existing_exts.union(exts))
            else:
                requested[base] = (cleaned, exts)

        return requested

    def _extract_paths_from_command(self, command: str) -> List[str]:
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()

        paths: List[str] = []
        for idx, token in enumerate(tokens):
            if not token or token in {"&&", "||", "|"}:
                continue
            if token.startswith('-'):
                continue
            if idx > 0 and tokens[idx - 1] in {"-m", "--module", "-c", "--command"}:
                continue
            if re.match(r'^[a-zA-Z]+://', token):
                continue
            cleaned = token.strip('"\'')
            if not cleaned:
                continue
            if '\n' in cleaned:
                continue
            if '/' in cleaned or '\\' in cleaned or re.search(r'\.[A-Za-z0-9]{1,6}$', cleaned):
                paths.append(cleaned)
        return paths

    def _detect_extension_mismatches(
        self,
        commands: List[str],
        user_input: Optional[str],
    ) -> List[Tuple[Tuple[str, Set[str]], str]]:
        requested = self._extract_requested_files(user_input)
        if not requested:
            return []

        mismatches: List[Tuple[Tuple[str, Set[str]], str]] = []

        for command in commands:
            for path_str in self._extract_paths_from_command(command):
                name = Path(path_str.strip("\"'" )).name
                parts = name.split('.')
                if len(parts) < 2:
                    continue
                base = parts[0].lower()
                exts = {part.lower() for part in parts[1:] if part}
                if base in requested:
                    expected_token, expected_exts = requested[base]
                    if expected_exts and not exts.intersection(expected_exts):
                        mismatch = ((expected_token, expected_exts), path_str)
                        if mismatch not in mismatches:
                            mismatches.append(mismatch)

        return mismatches

    def _detect_missing_files(self, commands: List[str], working_dir: str) -> List[str]:
        creating_keywords = ("mkdir", "touch", ">", "tee", "path.touch", "path.parent.mkdir")
        missing: Set[str] = set()

        for command in commands:
            if any(keyword in command for keyword in creating_keywords):
                continue

            for token in self._extract_paths_from_command(command):
                cleaned = token.strip('\"\'')
                if not cleaned or cleaned.startswith('$'):
                    continue
                if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', cleaned):
                    continue

                candidate = Path(cleaned).expanduser()
                if not candidate.is_absolute():
                    candidate = Path(working_dir) / candidate

                if not candidate.exists():
                    missing.add(str(candidate))

        return sorted(missing)

    def execute_commands(
        self,
        command_info: Dict[str, Any],
        recovery_origin: bool = False,
        ask_confirmation: bool = True,
        user_input: Optional[str] = None,
    ) -> bool:
        """Execute commands with working directory support, recovery, and safety preview"""
        if not self._validate_llm_command_info(command_info):
            logger.warning("⚠️  Command info rejected by schema validator.")
            return False

        self._last_execution_cancelled = False
        commands = [cmd for cmd in command_info.get("commands", []) if cmd]
        if not commands:
            logger.warning("⚠️  No commands to execute.")
            return False

        working_dir = command_info.get("working_directory") or self.current_dir
        working_dir = os.path.abspath(os.path.expanduser(working_dir))
        if not os.path.exists(working_dir):
            logger.warning("⚠️  Working directory '%s' not found. Using current directory.", working_dir)
            working_dir = self.current_dir

        logger.info("📂 Working Directory: %s", working_dir)
        print(f"📂 Working Directory: {working_dir}")

        logger.info("⚡ Commands to execute:")
        print("⚡ Commands:")
        for cmd in commands:
            logger.info("  ▶ %s", cmd)
            print(f"  {cmd}")

        mismatches = self._detect_extension_mismatches(commands, user_input)
        if mismatches and ask_confirmation:
            print("❓ The command references files with different extensions than requested:")
            for mismatch in mismatches:
                requested, used = mismatch
                expected_exts = ", ".join(sorted({f".{ext}" for ext in requested[1]}))
                print(f"   - Expected {requested[0]} with {expected_exts} but command uses {used}")
            confirm_mismatch = input("Continue with the suggested files? [y/N]: ").strip().lower()
            if confirm_mismatch not in {"y", "yes"}:
                print("❌ Cancelled due to file mismatch.")
                self._last_execution_cancelled = True
                return False

        missing_files = self._detect_missing_files(commands, working_dir)
        if missing_files:
            print("⚠️  The following paths were not found:")
            for path in missing_files:
                print(f"   - {path}")
            if ask_confirmation:
                confirm_missing = input("❓ Continue anyway? [y/N]: ").strip().lower()
                if confirm_missing not in {"y", "yes"}:
                    print("❌ Cancelled due to missing files.")
                    self._last_execution_cancelled = True
                    return False

        if ask_confirmation:
            prompt_text = (
                "✅ Execute these commands? [y/N]: "
                if not recovery_origin
                else "♻️ Execute recovery commands? [y/N]: "
            )
            confirm = input(prompt_text).strip().lower()
            if confirm not in {"y", "yes"}:
                logger.info("❌ Execution cancelled.")
                print("❌ Execution cancelled.")
                self._last_execution_cancelled = True
                return False

        if recovery_origin:
            self._recovery_depth += 1
            if self._recovery_depth > self._recovery_limit:
                logger.error("❌ Recovery depth exceeded. Aborting.")
                self._recovery_depth -= 1
                return False

        original_dir = self.current_dir

        try:
            if working_dir != self.current_dir:
                os.chdir(working_dir)
                self.current_dir = os.getcwd()
                logger.info("📁 Changed to: %s", self.current_dir)
                print(f"📁 Changed to: {self.current_dir}")

            for i, cmd in enumerate(commands):
                logger.info("🔧 [%s/%s] Executing: %s", i + 1, len(commands), cmd)
                print(f"🔧 {cmd}")

                install_target = self._extract_install_target(cmd)
                start_time = None
                if install_target:
                    print(f"⬇️ Installing {install_target} (this may take a moment)...")
                    start_time = time.time()

                danger = self._is_potentially_destructive(cmd)
                if danger and not recovery_origin:
                    confirmation = input(
                        f"⚠️ Command looks destructive ({danger}). Type 'FORCE' to run: "
                    ).strip()
                    if confirmation != "FORCE":
                        logger.warning("❌ Skipping destructive command.")
                        return False

                success, stdout, stderr = self._run_shell_command(cmd)

                if install_target and start_time is not None:
                    duration = time.time() - start_time
                    status = "✅" if success else "⚠️"
                    print(f"{status} Installation step for {install_target} finished in {duration:.1f}s")

                if stdout:
                    logger.info("📄 %s", stdout)
                    print(stdout)
                if stderr:
                    logger.warning("⚠️  %s", stderr)
                    print(f"⚠️  {stderr}")

                if not success:
                    recovered = self._attempt_recovery(
                        failed_command=cmd,
                        error_output=stderr,
                        prior_commands=commands[:i],
                        pending_commands=commands[i + 1 :],
                    )
                    if recovered:
                        # Successful recovery already reran the failed command
                        continue
                    print("❌ Command failed.")
                    return False

            return True

        except Exception as e:
            logger.error("💥 Error: %s", e)
            return False
        finally:
            if original_dir != self.current_dir:
                os.chdir(original_dir)
                self.current_dir = original_dir
            if recovery_origin and self._recovery_depth > 0:
                self._recovery_depth -= 1

    def _run_shell_command(self, command: str, timeout: int = 0) -> Tuple[bool, str, str]:
        """Execute a shell command with timeout handling."""

        timeout = timeout or int(os.getenv("AI_TERMINAL_COMMAND_TIMEOUT", "120"))

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, "", f"Command '{command}' timed out after {timeout} seconds"
        except Exception as exc:  # noqa: BLE001
            return False, "", f"Execution error: {exc}"

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        return result.returncode == 0, stdout, stderr

    def _attempt_recovery(
        self,
        *,
        failed_command: str,
        error_output: str,
        prior_commands: List[str],
        pending_commands: List[str],
        max_attempts: int = 2,
    ) -> bool:
        """Ask the LLM for recovery steps and retry the failed command."""

        key = (failed_command, (error_output or "")[:200])
        if self._global_recovery_attempts.get(key):
            logger.warning("⚠️ Recovery already attempted for this failure. Skipping repetitive attempts.")
            print("⚠️ Recovery already attempted for this failure. Skipping.")
            return False

        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            logger.info("🔁 Attempting automated recovery (attempt %s/%s)...", attempt, max_attempts)
            print(f"🔁 Recovery attempt {attempt}/{max_attempts}")
            cache_key = (failed_command, (error_output or "")[:200], attempt)
            recovery_info = self._recovery_cache.get(cache_key)
            if not recovery_info:
                recovery_info = self._request_recovery_plan(
                    failed_command=failed_command,
                    error_output=error_output,
                    prior_commands=prior_commands,
                    pending_commands=pending_commands,
                )
                if recovery_info:
                    self._recovery_cache[cache_key] = recovery_info

            if not recovery_info:
                install_plan = self._suggest_installation(failed_command, error_output)
                if install_plan:
                    logger.info("💡 Suggesting installation commands: %s", install_plan)
                    recovery_info = install_plan
                else:
                    logger.warning("⚠️  No recovery plan received from the LLM.")
                    print("⚠️  No recovery plan from LLM.")
                    return False

            recovery_commands = [cmd for cmd in recovery_info.get("commands", []) if cmd]
            if not recovery_commands:
                logger.warning("⚠️  Recovery plan did not include commands.")
                print("⚠️  Recovery plan missing commands.")
                return False

            if not self._validate_llm_command_info(recovery_info):
                logger.warning("⚠️  Recovery response failed validation.")
                print("⚠️  Recovery response failed validation.")
                return False

            success = self.execute_commands(
                recovery_info,
                recovery_origin=True,
                ask_confirmation=True,
                user_input=recovery_info.get("llm_raw"),
            )
            if not success:
                if self._last_execution_cancelled:
                    print("⚠️  Recovery cancelled by user.")
                    self._global_recovery_attempts[key] = True
                    return False
                logger.warning("⚠️  Recovery commands failed. Requesting another plan...")
                print("⚠️  Recovery commands failed. Trying again...")
                continue

            logger.info("✅ Recovery commands executed. Retrying failed command...")
            print("✅ Recovery commands executed. Retrying original command...")
            success, stdout, stderr = self._run_shell_command(failed_command)
            if stdout:
                logger.info("📄 %s", stdout)
                print(stdout)
            if stderr:
                logger.warning("⚠️  %s", stderr)
                print(f"⚠️  {stderr}")
            if success:
                logger.info("✅ Original command succeeded after recovery.")
                print("✅ Original command succeeded after recovery.")
                self._global_recovery_attempts[key] = True
                return True

            error_output = stderr
            logger.warning("⚠️  Original command still failing.")
            print("⚠️  Original command still failing.")

        logger.error("❌ Automated recovery exhausted.")
        print("❌ Automated recovery exhausted.")
        self._global_recovery_attempts[key] = True
        return False

    def _suggest_installation(self, failed_command: str, error_output: str) -> Optional[Dict[str, Any]]:
        """Suggest installation commands when a binary is missing."""

        binary = self._extract_missing_binary(error_output)
        if not binary:
            return None

        install_cmds: List[str] = []
        safe_binary = shlex.quote(binary)

        if self.system_os == "darwin":
            install_cmds.append(f"brew install --cask {safe_binary}")
            install_cmds.append(f"brew install {safe_binary}")
        elif self.system_os == "linux":
            install_cmds.append(f"sudo apt-get install -y {binary}")
            install_cmds.append(f"sudo yum install -y {binary}")
        elif self.system_os == "windows":
            install_cmds.append(f"choco install {binary} -y")

        if not install_cmds:
            return None

        return {
            "commands": install_cmds,
            "description": f"Attempt to install missing binary: {binary}",
            "type": "system",
            "working_directory": self.current_dir,
        }

    def _extract_missing_binary(self, error_output: Optional[str]) -> Optional[str]:
        if not error_output:
            return None

        patterns = [
            r"command not found: (\w+)",
            r"'(.+?)' is not recognized",
            r"no such file or directory: (\w+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, error_output, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _extract_install_target(self, command: str) -> Optional[str]:
        patterns = [
            r"brew\s+install(?:\s+--cask)?\s+([\w\-\.]+)",
            r"sudo\s+apt-get\s+install\s+-y\s+([\w\-\.]+)",
            r"sudo\s+yum\s+install\s+-y\s+([\w\-\.]+)",
            r"choco\s+install\s+([\w\-\.]+)",
            r"pip\s+install\s+([\w\-\.]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, command, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _request_recovery_plan(
        self,
        *,
        failed_command: str,
        error_output: str,
        prior_commands: List[str],
        pending_commands: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Request corrective commands from the LLM."""

        prompt = textwrap.dedent(
            f"""
            The previous command failed while executing a user workflow.

            Failed command:
            {failed_command}

            Error output:
            {error_output or '(no stderr captured)'}

            Commands run successfully before the failure:
            {prior_commands or 'None'}

            Remaining planned commands to execute after recovery:
            {pending_commands or 'None'}

            Operating system: {self.system_os}
            Current working directory: {self.current_dir}

            Provide recovery steps to resolve the failure. Respond ONLY with JSON using the standard schema where each command is a single shell-ready string.
            """
        ).strip()

        try:
            cache_key = (failed_command, (error_output or "")[:200], tuple(prior_commands), tuple(pending_commands))
            if cache_key in self._recovery_cache:
                return self._recovery_cache[cache_key]

            response = self.call_local_llm(prompt)
            if response:
                self._recovery_cache[cache_key] = response
            return response
        except Exception as exc:  # noqa: BLE001
            logger.warning("⚠️  Unable to obtain recovery plan: %s", exc)
            return None

    def _quote_path(self, path: str) -> str:
        """Quote a filesystem path for safe shell execution"""
        if self.system_os == "windows":
            return f'"{path}"'
        return shlex.quote(path)

    def show_help(self):
        """Show enhanced help"""
        help_text = """
🎯 ADVANCED AI TERMINAL - FILE EXECUTION & PATH SUPPORT

NOW UNDERSTANDS:
• "run the python script called example.py"
• "execute my program from the desktop folder"
• "launch the test file in the src directory"
• "start the application from downloads"

FILE TYPES SUPPORTED:
• Python files (.py) → python3 filename.py
• Shell scripts (.sh) → bash filename.sh
• Any executable → ./filename

PATH SUPPORT:
• Relative paths: "run src/main.py"
• Absolute paths: "run /Users/name/project/script.py"
• Home directory: "run ~/Documents/script.py"

EXAMPLES:
AI-Terminal> run the python file called test.py
AI-Terminal> execute my script from the desktop
AI-Terminal> launch the program in the current folder
AI-Terminal> run example.py with arguments hello world

NAVIGATION:
AI-Terminal> change to downloads directory
AI-Terminal> go to the parent folder
AI-Terminal> list files in the src folder
"""
        print(help_text)

    def run(self):
        """Main execution loop"""
        print(f"\n💡 Try: 'run example.py' or 'execute my script from desktop'")
        print("🔧 Advanced file execution supported!")

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
                    self.show_help()
                    continue

                if user_input.lower() == 'pwd':
                    print(f"📁 {self.current_dir}")
                    continue

                # Get AI interpretation
                command_info = self.understand_complex_command(user_input)

                print(f"🎯 {command_info.get('description', 'Executing command')}")

                success = self.execute_commands(command_info, user_input=user_input)
                if success:
                    print("✅ Done")
                else:
                    print("❌ Failed")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"💥 Error: {e}")


def create_test_scripts():
    """Create test scripts for demonstration"""
    test_py = """#!/usr/bin/env python3
print("🎯 Hello from Python script!")
print(f"📁 Current directory: {__import__('os').getcwd()}")
print("✅ Script executed successfully!")
"""

    test_sh = """#!/bin/bash
echo "🐚 Hello from Shell script!"
echo "📁 Current directory: $(pwd)"
echo "✅ Script executed successfully!"
"""

    with open("test_script.py", "w") as f:
        f.write(test_py)

    with open("test_script.sh", "w") as f:
        f.write(test_sh)

    # Make shell script executable
    os.chmod("test_script.sh", 0o755)

    print("✅ Created test_script.py and test_script.sh for demonstration")


if __name__ == "__main__":
    print("🚀 Advanced AI Terminal with File Execution")

    # Create test scripts if they don't exist
    if not os.path.exists("test_script.py"):
        create_test_scripts()

    terminal = AdvancedAITerminal()
    terminal.run()
