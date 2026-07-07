from gguf_limit_bench.qe_format import assess_qe_response, summarize_qe_assessments


def test_qe_format_accepts_one_to_three_lex_terms_and_one_hyde():
    assessment = assess_qe_response(
        "LEX: qwen template, reasoning_content, enable_thinking\n"
        "HYDE: A troubleshooting note about Qwen chat templates and reasoning extraction."
    )

    assert assessment.format_ok is True
    assert assessment.lex_terms == ("qwen template", "reasoning_content", "enable_thinking")
    assert assessment.hyde.startswith("A troubleshooting note")
    assert assessment.answered_question is False
    assert assessment.issues == ()
    assert assessment.score == 1.0


def test_qe_format_rejects_missing_hyde_and_direct_answer():
    assessment = assess_qe_response("LEX: qwen template\nANSWER: use the old template")

    assert assessment.format_ok is False
    assert assessment.answered_question is True
    assert "missing_hyde" in assessment.issues
    assert "direct_answer" in assessment.issues
    assert assessment.score < 1.0


def test_qe_format_rejects_more_than_three_lex_terms_and_multiple_hyde_sections():
    assessment = assess_qe_response(
        "LEX: qwen, template, reasoning, llama.cpp\n"
        "HYDE: First synthetic document.\n"
        "HYDE: Second synthetic document."
    )

    assert assessment.format_ok is False
    assert assessment.lex_terms == ("qwen", "template", "reasoning", "llama.cpp")
    assert "too_many_lex_terms" in assessment.issues
    assert "multiple_hyde_sections" in assessment.issues


def test_qe_format_salvages_multiple_hyde_sections_as_warning_when_payload_is_usable():
    assessment = assess_qe_response(
        "LEX: llama.cpp flags, props endpoint, slots endpoint\n"
        "HYDE: A broad note about llama.cpp flags.\n"
        "HYDE: A specific runtime-doctor note about checking props and slots before trusting flags."
    )

    assert assessment.format_ok is True
    assert assessment.hyde == (
        "A specific runtime-doctor note about checking props and slots before trusting flags."
    )
    assert "multiple_hyde_sections" in assessment.issues
    assert assessment.score < 1.0


def test_qe_format_summary_counts_failure_modes():
    assessments = [
        assess_qe_response("LEX: qwen\nHYDE: A relevant note."),
        assess_qe_response("LEX: qwen\nANSWER: direct answer"),
        assess_qe_response("HYDE: Missing lexical query."),
    ]

    summary = summarize_qe_assessments(assessments)

    assert summary["attempts"] == 3
    assert summary["valid"] == 1
    assert summary["format_rate"] == 1 / 3
    assert summary["direct_answer_count"] == 1
    assert summary["issue_counts"]["missing_lex"] == 1
    assert summary["issue_counts"]["missing_hyde"] == 1


def test_qe_format_canonicalizes_common_label_drift():
    assessment = assess_qe_response(
        "The user is asking about a runtime issue.\n\n"
        "Lex search terms: llama.cpp flags, slots endpoint, props endpoint\n"
        "Hyde document: A runtime-doctor note explaining which llama.cpp "
        "flags, slots, props, and template fields to inspect before trusting a profile."
    )

    assert assessment.format_ok is True
    assert assessment.lex_terms == ("llama.cpp flags", "slots endpoint", "props endpoint")
    assert assessment.hyde.startswith("A runtime-doctor note")
    assert "noncanonical_labels" in assessment.issues
    assert assessment.score < 1.0


def test_qe_format_uses_last_lex_and_hyde_when_model_rambles_before_payload():
    assessment = assess_qe_response(
        "Lex search terms: noisy first idea\n"
        "Hyde document: noisy first document\n"
        "LEX: web ui, tui, shared runners\n"
        "HYDE: An architecture note saying the website and TUI call the same runner."
    )

    assert assessment.format_ok is True
    assert assessment.lex_terms == ("web ui", "tui", "shared runners")
    assert (
        assessment.hyde == "An architecture note saying the website and TUI call the same runner."
    )
    assert "noncanonical_labels" in assessment.issues
