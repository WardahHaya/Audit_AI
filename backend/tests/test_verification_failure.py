import asyncio
import importlib
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.services.schemas import OrchestratorDecision


def test_compliance_review_overrides_model_document_route(monkeypatch):
    module = importlib.import_module("app.graph.nodes.orchestrator")
    service = SimpleNamespace(invoke_structured=AsyncMock(return_value=OrchestratorDecision(
        route="document", reason="Review uploaded documents")))
    monkeypatch.setattr(module, "get_runtime", lambda _: SimpleNamespace(ollama_service=service))
    monkeypatch.setattr(module, "emit_start", AsyncMock())
    monkeypatch.setattr(module, "emit_complete", AsyncMock())
    result = asyncio.run(module.orchestrator({
        "original_query": "Review this evidence for potential compliance gaps. Explain each finding, cite relevant requirements, and suggest corrective actions. Clearly flag anything you cannot verify.",
        "attachment_contexts": [{"file_name": "fire.pdf"}],
    }, {}))
    assert result["orchestrator_decision"]["route"] == "grounded"


@pytest.mark.parametrize("route", ["document", "grounded"])
def test_invalid_verification_withholds_draft_and_provider_error(monkeypatch, route):
    module = importlib.import_module("app.graph.nodes.main_llm_call")
    service = SimpleNamespace(
        invoke_text=AsyncMock(return_value="Unsupported finding: missing exit signs."),
        invoke_structured=AsyncMock(side_effect=ValueError("tool_use_failed: raw draft")),
    )
    emitter = SimpleNamespace(token=AsyncMock())
    monkeypatch.setattr(module, "get_runtime", lambda _: SimpleNamespace(ollama_service=service, emitter=emitter))
    monkeypatch.setattr(module, "emit_start", AsyncMock())
    monkeypatch.setattr(module, "emit_complete", AsyncMock())
    result = asyncio.run(module.main_llm_call({
        "original_query": "Review this file", "orchestrator_decision": {"route": route},
        "attachment_contexts": [{"file_name": "fire.pdf", "extracted_text": "Inspect signs."}],
    }, {}))
    assert result["response_status"] == "verification_failed"
    assert "withheld" in result["final_response"]
    streamed = "".join(call.args[0] for call in emitter.token.call_args_list)
    assert "missing exit signs" not in streamed
    assert "tool_use_failed" not in streamed
