from __future__ import annotations


def test_key_imports() -> None:
    import llm_harness
    import llm_harness.cli
    import llm_harness.eval.dataset
    import llm_harness.eval.normalize
    import llm_harness.eval.rules
    import llm_harness.eval.runner
    import llm_harness.eval.scoring
    import llm_harness.logging.events
    import llm_harness.providers.base
    import llm_harness.providers.dummy
    import llm_harness.providers.ollama
    import llm_harness.reporting.summarize

    assert llm_harness is not None
