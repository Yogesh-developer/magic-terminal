"""LLM client module for AI terminal planning and recovery."""

from __future__ import annotations

import json
import logging
import os
import platform
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


logger = logging.getLogger(__name__)


@dataclass
class PlanStep:
    """Structured command step returned by the LLM."""

    step: int
    command: str
    description: str


class LLMClient:
    """Handles communication with LLM backends and heuristic fallbacks."""

    OPENAI_URL = "https://api.openai.com/v1/chat/completions"
    GROK_URL = "https://api.x.ai/v1/chat/completions"

    def __init__(
        self,
        *,
        openai_model: str = "gpt-4o-mini",
        grok_model: str = "grok-1",
        temperature: float = 0.1,
    ) -> None:
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.grok_key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
        self.openai_model = openai_model
        self.grok_model = grok_model
        self.temperature = temperature

        self._system_os = platform.system()
        self._shell = self._detect_shell()

    def generate_plan(self, request: str) -> List[PlanStep]:
        """Return a list of step-by-step commands for the given request."""

        prompt = self._build_plan_prompt(request)
        response_text = self._call_llm(prompt)
        if response_text:
            steps = self._parse_steps(response_text)
            if steps:
                return steps
            self._log_failed_response("plan", response_text)

        return self._fallback_plan(request)

    def suggest_fix(
        self,
        *,
        failed_step: PlanStep,
        error_output: str,
        executed_steps: List[PlanStep],
    ) -> List[PlanStep]:
        """Return corrective steps when execution fails."""

        # Always try fallback first to ensure we have something
        fallback_suggestions = self._fallback_fix(failed_step, error_output)
        
        # Try LLM for potentially better suggestions
        prompt = self._build_fix_prompt(failed_step, error_output, executed_steps)
        response_text = self._call_llm(prompt)
        if response_text:
            fixes = self._parse_steps(response_text)
            if fixes:
                # Return LLM suggestions if they're good
                return fixes
            self._log_failed_response("fix", response_text)
        
        # If LLM failed or no LLM available, return fallback suggestions
        # If fallback is empty, provide generic troubleshooting steps
        if not fallback_suggestions:
            fallback_suggestions = self._generic_troubleshooting_steps(failed_step, error_output)
            
        return fallback_suggestions

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def _build_plan_prompt(self, request: str) -> str:
        os_name = self._system_os
        shell = self._shell

        return textwrap.dedent(
            f"""
            You are an expert DevOps assistant. Given a user request, return a JSON
            array of step-by-step shell commands to accomplish the task on the
            following environment:

            OS: {os_name}
            Preferred shell: {shell}

            Requirements:
              - Respond ONLY with valid JSON.
              - JSON must be a list of objects with keys: step (int), command (string), description (string).
              - Each command must be a SINGLE STRING containing the complete command with all arguments.
              - DO NOT split commands into arrays. Use complete command strings like "node --version", not ["node", "--version"].
              - Commands must be safe, idempotent where possible, and include any prerequisites
                such as package index refresh if required.
              - Use platform-appropriate package managers (apt, yum, pacman, brew, winget, etc.).
              - Do not include explanations outside of the JSON payload.

            Example response:
            [
              {{"step": 1, "command": "sudo apt update", "description": "Refresh package index"}},
              {{"step": 2, "command": "sudo apt install -y git", "description": "Install Git"}},
              {{"step": 3, "command": "node --version", "description": "Check Node.js version"}}
            ]

            User request: {request}
            """
        ).strip()
    
    def _build_fix_prompt(
        self,
        failed_step: PlanStep,
        error_output: str,
        executed_steps: List[PlanStep],
    ) -> str:
        executed_json = [step.__dict__ for step in executed_steps]
        os_name = self._system_os
        shell = self._shell

        return textwrap.dedent(
            f"""
            You are troubleshooting a failed shell command on OS {os_name} (shell {shell}).
            The user attempted the following steps (already executed):
            {json.dumps(executed_json, indent=2)}

            The failing step:
            {json.dumps(failed_step.__dict__, indent=2)}

            Error output:
            {error_output}

            Provide a JSON array of corrective steps (same structure as the original plan)
            to resolve the failure and continue the workflow. If a prerequisite is missing,
            include commands to install or configure it. Respond ONLY with JSON.
            """
        ).strip()

    # ------------------------------------------------------------------
    # LLM invocation helpers
    # ------------------------------------------------------------------
    def _call_llm(self, prompt: str) -> Optional[str]:
        try:
            if self.openai_key:
                return self._call_openai(prompt)
            if self.grok_key:
                return self._call_grok(prompt)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  LLM call failed: {exc}")
        return None

    def _log_failed_response(self, kind: str, response_text: str) -> None:
        snippet = response_text.strip()
        if len(snippet) > 600:
            snippet = snippet[:600] + "…"
        logger.warning("LLM %s response could not be parsed; raw response: %s", kind, snippet)
        print(f"⚠️  Could not parse LLM {kind} response. See logs for details.")

    def _call_openai(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.openai_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.openai_model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": "You respond only with JSON."},
                {"role": "user", "content": prompt},
            ],
        }
        response = requests.post(self.OPENAI_URL, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _call_grok(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.grok_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.grok_model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": "You respond only with JSON."},
                {"role": "user", "content": prompt},
            ],
        }
        response = requests.post(self.GROK_URL, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    def _parse_steps(self, response_text: str) -> List[PlanStep]:
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError:
            start = response_text.find("[")
            end = response_text.rfind("]")
            if start != -1 and end != -1:
                try:
                    payload = json.loads(response_text[start : end + 1])
                except json.JSONDecodeError:
                    return []
            else:
                return []

        steps: List[PlanStep] = []
        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                command = item.get("command")
                if not command:
                    continue
                steps.append(
                    PlanStep(
                        step=int(item.get("step", len(steps) + 1)),
                        command=str(command).strip(),
                        description=str(item.get("description", "")).strip(),
                    )
                )
        return steps

    def _fallback_plan(self, request: str) -> List[PlanStep]:
        request_lower = request.lower()
        package = None
        for keyword in ["python", "git", "node", "nodejs", "docker", "vscode", "visual studio code"]:
            if keyword in request_lower:
                package = keyword
                break

        if not package:
            return []

        command = self._heuristic_install_command(package)
        if not command:
            return []

        return [PlanStep(step=1, command=command, description=f"Install {package}")]

    def _fallback_fix(self, failed_step: PlanStep, error_output: str) -> List[PlanStep]:
        suggestions: List[PlanStep] = []
        error_lower = error_output.lower()
        command_lower = failed_step.command.lower()

        # Permission-related fixes
        if "permission" in error_lower and not failed_step.command.startswith("sudo"):
            suggestions.append(
                PlanStep(
                    step=failed_step.step,
                    command=f"sudo {failed_step.command}",
                    description="Retry with elevated privileges",
                )
            )

        # macOS-specific command fixes
        if self._system_os.lower() == "darwin":
            # Fix for top command with invalid %MEM argument
            if "top" in command_lower and "invalid argument" in error_lower and "%mem" in command_lower:
                suggestions.append(
                    PlanStep(
                        step=failed_step.step,
                        command="top -o mem -l 1 -n 10",
                        description="Use 'mem' instead of '%MEM' for macOS top",
                    )
                )
                suggestions.append(
                    PlanStep(
                        step=failed_step.step + 1,
                        command="ps aux | sort -k4 -nr | head -10",
                        description="Alternative: Use ps with memory sorting",
                    )
                )

            # Fix for systemctl (Linux command) on macOS
            elif "systemctl" in command_lower and "command not found" in error_lower:
                if "docker" in command_lower:
                    suggestions.append(
                        PlanStep(
                            step=failed_step.step,
                            command="ps aux | grep -i docker",
                            description="Check if Docker processes are running",
                        )
                    )
                    suggestions.append(
                        PlanStep(
                            step=failed_step.step + 1,
                            command="open -a Docker",
                            description="Start Docker Desktop application",
                        )
                    )
                else:
                    suggestions.append(
                        PlanStep(
                            step=failed_step.step,
                            command="launchctl list | grep -i service",
                            description="Use launchctl to check services on macOS",
                        )
                    )

            # Fix for journalctl (Linux command) on macOS
            elif "journalctl" in command_lower and "command not found" in error_lower:
                suggestions.append(
                    PlanStep(
                        step=failed_step.step,
                        command="log show --last 50 --style compact",
                        description="Use macOS log command instead of journalctl",
                    )
                )
                suggestions.append(
                    PlanStep(
                        step=failed_step.step + 1,
                        command="tail -50 /var/log/system.log",
                        description="Alternative: Read system log directly",
                    )
                )

            # Fix for sysctl unknown oid
            elif "sysctl" in command_lower and "unknown oid" in error_lower:
                if "hw.cpu" in command_lower:
                    suggestions.append(
                        PlanStep(
                            step=failed_step.step,
                            command="sysctl -n machdep.cpu.brand_string",
                            description="Get CPU brand information",
                        )
                    )
                    suggestions.append(
                        PlanStep(
                            step=failed_step.step + 1,
                            command="sysctl -n hw.ncpu",
                            description="Get number of CPU cores",
                        )
                    )
                    suggestions.append(
                        PlanStep(
                            step=failed_step.step + 2,
                            command="system_profiler SPHardwareDataType | grep 'Processor'",
                            description="Alternative: Use system_profiler for CPU details",
                        )
                    )

            # Fix for python command not found
            elif "python" in command_lower and "command not found" in error_lower:
                suggestions.append(
                    PlanStep(
                        step=failed_step.step,
                        command="python3 --version",
                        description="Try python3 instead of python",
                    )
                )
                suggestions.append(
                    PlanStep(
                        step=failed_step.step + 1,
                        command="which python3",
                        description="Check if python3 is available",
                    )
                )

        # Docker daemon not running (cross-platform)
        if ("docker daemon" in error_lower and "not running" in error_lower) or \
           ("cannot connect to the docker daemon" in error_lower):
            if self._system_os.lower() == "darwin":
                suggestions.append(
                    PlanStep(
                        step=failed_step.step,
                        command="open -a Docker",
                        description="Start Docker Desktop on macOS",
                    )
                )
                suggestions.append(
                    PlanStep(
                        step=failed_step.step + 1,
                        command="sleep 10 && docker version",
                        description="Wait for Docker to start and verify",
                    )
                )
            else:
                suggestions.append(
                    PlanStep(
                        step=failed_step.step,
                        command="sudo systemctl start docker",
                        description="Start Docker daemon",
                    )
                )
                suggestions.append(
                    PlanStep(
                        step=failed_step.step + 1,
                        command="sudo systemctl status docker",
                        description="Check Docker daemon status",
                    )
                )

        # Linux-specific fixes
        if "not found" in error_lower and "apt" in failed_step.command:
            suggestions.append(
                PlanStep(
                    step=max(failed_step.step - 1, 1),
                    command="sudo apt update",
                    description="Refresh package list before retry",
                )
            )

        # Generic "command not found" fallback for any missed cases
        if not suggestions and ("command not found" in error_lower or "not found" in error_lower):
            command_name = command_lower.split()[0] if command_lower.split() else "command"
            
            if self._system_os.lower() == "darwin":
                suggestions.append(
                    PlanStep(
                        step=failed_step.step,
                        command=f"brew search {command_name}",
                        description=f"Search for {command_name} in Homebrew",
                    )
                )
                suggestions.append(
                    PlanStep(
                        step=failed_step.step + 1,
                        command=f"brew install {command_name}",
                        description=f"Try installing {command_name} with Homebrew",
                    )
                )
            else:
                suggestions.append(
                    PlanStep(
                        step=failed_step.step,
                        command=f"apt search {command_name}",
                        description=f"Search for {command_name} in package manager",
                    )
                )

        return suggestions

    def _generic_troubleshooting_steps(self, failed_step: PlanStep, error_output: str) -> List[PlanStep]:
        """Provide generic troubleshooting steps when specific fixes aren't available."""
        suggestions: List[PlanStep] = []
        error_lower = error_output.lower()
        command_lower = failed_step.command.lower()

        # Command not found - suggest alternatives
        if "command not found" in error_lower or "not found" in error_lower:
            command_name = command_lower.split()[0] if command_lower.split() else "command"
            suggestions.append(
                PlanStep(
                    step=failed_step.step,
                    command=f"which {command_name}",
                    description=f"Check if {command_name} is installed and in PATH",
                )
            )
            suggestions.append(
                PlanStep(
                    step=failed_step.step + 1,
                    command=f"type {command_name}",
                    description=f"Alternative check for {command_name} availability",
                )
            )

        # Permission denied
        elif "permission denied" in error_lower:
            suggestions.append(
                PlanStep(
                    step=failed_step.step,
                    command=f"ls -la {command_lower.split()[-1] if command_lower.split() else '.'}",
                    description="Check file/directory permissions",
                )
            )

        # Network/connection issues
        elif any(keyword in error_lower for keyword in ["connection", "network", "timeout", "unreachable"]):
            suggestions.append(
                PlanStep(
                    step=failed_step.step,
                    command="ping -c 1 8.8.8.8",
                    description="Test internet connectivity",
                )
            )

        # File/directory not found
        elif "no such file" in error_lower or "directory" in error_lower:
            suggestions.append(
                PlanStep(
                    step=failed_step.step,
                    command="pwd",
                    description="Check current directory",
                )
            )
            suggestions.append(
                PlanStep(
                    step=failed_step.step + 1,
                    command="ls -la",
                    description="List files in current directory",
                )
            )

        # If no specific suggestions, provide general help
        if not suggestions:
            suggestions.append(
                PlanStep(
                    step=failed_step.step,
                    command=f"man {command_lower.split()[0] if command_lower.split() else 'help'}",
                    description="Check manual/help for the command",
                )
            )
            suggestions.append(
                PlanStep(
                    step=failed_step.step + 1,
                    command=f"{command_lower.split()[0] if command_lower.split() else 'command'} --help",
                    description="Try getting help for the command",
                )
            )

        return suggestions

    def _heuristic_install_command(self, package: str) -> Optional[str]:
        os_name = self._system_os.lower()
        package_lower = package.lower()

        if "darwin" in os_name or "mac" in os_name:
            if package_lower in {"vscode", "visual studio code"}:
                return "brew install --cask visual-studio-code"
            if package_lower in {"node", "nodejs"}:
                return "brew install node"
            return f"brew install {package_lower}"

        if "windows" in os_name:
            mapping = {
                "vscode": "Microsoft.VisualStudioCode",
                "visual studio code": "Microsoft.VisualStudioCode",
                "python": "Python.Python.3.12",
                "git": "Git.Git",
            }
            package_id = mapping.get(package_lower, package_lower)
            return f"winget install -e --id {package_id}"

        return f"sudo apt update && sudo apt install -y {package_lower}"

    def _detect_shell(self) -> str:
        if self._system_os.lower().startswith("win"):
            return "powershell"
        return os.getenv("SHELL", "/bin/bash")

