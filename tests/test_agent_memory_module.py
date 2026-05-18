from agent_memory import get_card, list_cards, search_cards


def test_agent_memory_manifest_and_cards_load():
    cards = list_cards()
    assert len(cards) >= 8
    assert any(card.card_id == "audit_findings" for card in cards)
    assert any(card.card_id == "maintenance_protocol" for card in cards)
    assert all(card.file_path.exists() for card in cards)


def test_agent_memory_search_and_get_card():
    card = get_card("training_pipeline")
    assert "Training Pipeline" in card.title
    assert "checkpoints/" in card.read_text()
    assert "tensorboard/" in card.read_text()

    results = search_cards(query="checkpoint", tags=["type:training-pipeline"])
    assert any(result.card_id == "training_pipeline" for result in results)

    maintenance = get_card("maintenance_protocol")
    maintenance_text = maintenance.read_text().lower()
    assert "repository behavior" in maintenance_text
    assert "agent_memory" in maintenance_text
