import json
from agent_lab.runtime import LabRuntime


def test_mismatched_checkpoint_metrics_are_never_displayed(tmp_path):
    (tmp_path / "report.json").write_text(json.dumps({"checkpoint_sha256":"current"}))
    for name in ("evaluation.json", "sandbox-audit.json"):
        (tmp_path / name).write_text(json.dumps({"checkpoint_sha256":"different", "success_rate":1}))
    status=LabRuntime(tmp_path).status()
    assert status["evaluation"] is None and status["sandbox_audit"] is None
    (tmp_path / "sandbox-audit.json").write_text(json.dumps({"checkpoint_sha256":"current"}))
    assert LabRuntime(tmp_path).status()["sandbox_audit"]["checkpoint_sha256"]=="current"
