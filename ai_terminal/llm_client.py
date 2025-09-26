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

        prompt = self._build_fix_prompt(failed_step, error_output, executed_steps)
        response_text = self._call_llm(prompt)
        if response_text:
            fixes = self._parse_steps(response_text)
            if fixes:
                return fixes
            self._log_failed_response("fix", response_text)
        return self._fallback_fix(failed_step, error_output)

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

        if "permission" in error_lower and not failed_step.command.startswith("sudo"):
            suggestions.append(
                PlanStep(
                    step=failed_step.step,
                    command=f"sudo {failed_step.command}",
                    description="Retry with elevated privileges",
                )
            )

        if "not found" in error_lower and "apt" in failed_step.command:
            suggestions.append(
                PlanStep(
                    step=max(failed_step.step - 1, 1),
                    command="sudo apt update",
                    description="Refresh package list before retry",
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

