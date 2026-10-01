from app.graph.node_utils import compact_history, document_workspace_attachments


def test_large_documents_share_budget_and_keep_provenance():
    originals = [
        {"upload_id": str(i), "file_name": f"policy-{i}.txt", "extracted_text": "x" * 30000}
        for i in range(4)
    ]
    result = document_workspace_attachments(originals)
    assert sum(len(row["extracted_content"]) for row in result) <= 8000
    assert all(row["content_truncated"] and row["extracted_content"] for row in result)
    assert [row["upload_id"] for row in result] == [str(i) for i in range(4)]
    assert all(len(row["extracted_text"]) == 30000 for row in originals)


def test_short_document_is_preserved_and_spare_budget_is_reused():
    result = document_workspace_attachments([
        {"extracted_text": "A short policy."},
        {"extracted_text": "y" * 30000},
    ])
    assert result[0]["extracted_content"] == "A short policy."
    assert not result[0]["content_truncated"]
    assert len(result[1]["extracted_content"]) == 6000
    assert document_workspace_attachments([]) == []


def test_history_keeps_latest_turn_within_budget():
    history = [{"role": "assistant", "content": "x" * 30000},
               {"role": "user", "content": "Please review the exit policy."}]
    result = compact_history(history)
    assert len(result) <= 2000
    assert result.endswith("user: Please review the exit policy.")
    assert "Earlier history omitted" in result
