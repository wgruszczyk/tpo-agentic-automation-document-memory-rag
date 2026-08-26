from __future__ import annotations

from product_memory.eval_generation import (
    GeneratedCase,
    Vocabulary,
    _best_sentence,
    _salient_terms,
    build_vocabulary,
    render_yaml,
)

TOTAL = 1000


def _vocabulary(frequencies: dict[str, int]) -> Vocabulary:
    return build_vocabulary(frequencies, total_chunks=TOTAL)


def test_a_word_seen_once_in_the_corpus_is_treated_as_noise() -> None:
    vocabulary = _vocabulary({"whatappconnection": 1, "settlement": 4})

    assert "whatappconnection" not in vocabulary.usable
    assert "settlement" in vocabulary.usable


def test_a_word_in_most_chunks_is_too_common_to_ask_about() -> None:
    vocabulary = _vocabulary({"invoice": 4, "document": 900})

    assert "invoice" in vocabulary.usable
    assert "document" not in vocabulary.usable


def test_short_words_and_question_words_never_enter_the_vocabulary() -> None:
    vocabulary = _vocabulary({"vat": 5, "which": 5, "reconciliation": 5})

    assert "vat" not in vocabulary.usable
    assert "which" not in vocabulary.usable
    assert "reconciliation" in vocabulary.usable


def test_salient_terms_keep_the_order_they_were_written_in() -> None:
    vocabulary = _vocabulary({"settlement": 2, "reconciliation": 3, "invoice": 4})

    terms = _salient_terms("The invoice drives settlement and reconciliation.", vocabulary, limit=3)

    assert terms == ["invoice", "settlement", "reconciliation"]


def test_only_the_most_distinctive_terms_survive_the_limit() -> None:
    vocabulary = _vocabulary({"common": 400, "rare": 2, "middling": 40})

    terms = _salient_terms("common middling rare", vocabulary, limit=2)

    assert terms == ["middling", "rare"]


def test_a_sentence_is_rejected_when_it_carries_too_little_vocabulary() -> None:
    vocabulary = _vocabulary({"settlement": 3})

    assert _best_sentence("Yes. No.", vocabulary, min_terms=2) is None
    assert _best_sentence("settlement settlement.", vocabulary, min_terms=2) is None


def test_the_richest_sentence_is_the_one_chosen() -> None:
    vocabulary = _vocabulary({"invoice": 300, "settlement": 2, "reconciliation": 2})

    chosen = _best_sentence(
        "The invoice was sent. Settlement reconciliation followed.", vocabulary, min_terms=2
    )

    assert chosen == "Settlement reconciliation followed."


def test_rendered_yaml_grades_the_source_document_and_warns_against_committing() -> None:
    rendered = render_yaml(
        [
            GeneratedCase(
                question='payment "terms" annex',
                source_paths=["contracts/annex.pdf"],
                document_id="doc-1",
                chunk_id="chunk-1",
                salience=4.2,
            )
        ],
        seed="unit-test",
    )

    assert "Do not commit" in rendered
    assert "seed: unit-test" in rendered
    assert '- question: "payment \\"terms\\" annex"' in rendered
    assert '    - path: "contracts/annex.pdf"' in rendered
    assert "      grade: 3" in rendered


def test_rendered_yaml_parses_back_into_graded_cases(tmp_path) -> None:
    from product_memory.evaluation import load_cases

    path = tmp_path / "generated.yaml"
    path.write_text(
        render_yaml(
            [
                GeneratedCase(
                    question="settlement reconciliation",
                    source_paths=["finance/ledger.xlsx"],
                    document_id="doc-1",
                    chunk_id="chunk-1",
                    salience=3.0,
                )
            ],
            seed="unit-test",
        ),
        encoding="utf-8",
    )

    cases = load_cases(path)

    assert cases[0].question == "settlement reconciliation"
    assert cases[0].expect[0].fragment == "finance/ledger.xlsx"
    assert cases[0].expect[0].grade == 3


def test_a_question_several_documents_answer_lists_them_all(tmp_path) -> None:
    from product_memory.evaluation import load_cases

    path = tmp_path / "merged.yaml"
    path.write_text(
        render_yaml(
            [
                GeneratedCase(
                    question="settlement reconciliation",
                    # Successive drafts share their most distinctive sentence, and any of them
                    # answers it; one question per copy would mark all but one a miss.
                    source_paths=["finance/ledger-v1.xlsx", "finance/ledger-v2.xlsx"],
                    document_id="doc-1",
                    chunk_id="chunk-1",
                    salience=3.0,
                )
            ],
            seed="unit-test",
        ),
        encoding="utf-8",
    )

    cases = load_cases(path)

    assert len(cases) == 1
    assert cases[0].fragments == ["finance/ledger-v1.xlsx", "finance/ledger-v2.xlsx"]
    assert [item.grade for item in cases[0].expect] == [3, 3]


def test_sampling_visits_every_folder_in_proportion_to_its_size() -> None:
    from contextlib import contextmanager

    from product_memory.eval_generation import _sample_documents

    captured: dict[str, str] = {}

    class Cursor:
        def fetchall(self) -> list:
            return []

    class Connection:
        def execute(self, sql: str, params: dict) -> Cursor:
            captured["sql"] = sql
            return Cursor()

    class Db:
        @contextmanager
        def connection(self):
            yield Connection()

    _sample_documents(Db(), count=10, seed="s", project=None)  # type: ignore[arg-type]

    # Ranking within a folder and ordering on that rank as a fraction of the folder's size means
    # any prefix holds each folder in proportion; a flat random order lets the largest one win.
    assert "PARTITION BY folder" in captured["sql"]
    assert "::float / count(*) OVER (PARTITION BY folder)" in captured["sql"]
    assert "split_part(d.source_path, '/', 1) AS folder" in captured["sql"]
