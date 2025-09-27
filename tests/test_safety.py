from ai_terminal.safety import CommandAuditor


def test_audit_detects_high_risk_command() -> None:
    auditor = CommandAuditor()
    warnings = auditor.audit("rm -rf /")
    assert warnings
    assert any(warning.severity == "high" for warning in warnings)


def test_audit_detects_medium_risk_command() -> None:
    auditor = CommandAuditor()
    warnings = auditor.audit("rm -rf ~/Downloads")
    assert warnings
    assert any(warning.severity == "medium" for warning in warnings)


def test_highest_severity_prefers_high() -> None:
    auditor = CommandAuditor()
    warnings = auditor.audit("rm -rf / && rm -rf ~/Downloads")
    severity = auditor.highest_severity(warnings)
    assert severity == "high"


def test_audit_returns_empty_for_safe_command() -> None:
    auditor = CommandAuditor()
    warnings = auditor.audit("ls -la")
    assert warnings == []
