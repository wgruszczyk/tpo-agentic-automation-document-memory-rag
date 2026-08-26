from pathlib import Path

from product_memory.ingestion.service import IngestionService
from product_memory.settings import Settings


def _service(tmp_path: Path) -> IngestionService:
    settings = Settings(knowledge_dir=tmp_path, _env_file=None)
    return IngestionService(settings, db=None, provider=None, parser=None, chunker=None)  # type: ignore[arg-type]


def _relative_paths(service: IngestionService, root: Path) -> list[str]:
    return [path.relative_to(root).as_posix() for path in service._discover_paths(root)]  # noqa: SLF001


def _write(path: Path, content: str = "content") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_example_knowledge_is_used_when_only_placeholder_exists(tmp_path: Path) -> None:
    _write(tmp_path / ".gitkeep")
    _write(tmp_path / "example-knowledge.md")

    assert _relative_paths(_service(tmp_path), tmp_path) == ["example-knowledge.md"]


def test_example_knowledge_is_skipped_when_real_documents_exist(tmp_path: Path) -> None:
    _write(tmp_path / ".README.md")
    _write(tmp_path / "example-knowledge.md")
    _write(tmp_path / "teams-transcript.vtt")

    assert _relative_paths(_service(tmp_path), tmp_path) == ["teams-transcript.vtt"]


def test_root_readme_does_not_count_as_a_real_document(tmp_path: Path) -> None:
    _write(tmp_path / "README.md")
    _write(tmp_path / "example-knowledge.md")

    assert _relative_paths(_service(tmp_path), tmp_path) == ["example-knowledge.md"]


def test_office_lock_files_are_never_indexed(tmp_path: Path) -> None:
    # Office writes these beside a document while it is open. They carry the extension of a
    # presentation but hold no package, so reading one can only ever fail.
    _write(tmp_path / "deck.pptx")
    _write(tmp_path / "~$deck.pptx")
    _write(tmp_path / "minutes" / "~$notes.docx")

    assert _relative_paths(_service(tmp_path), tmp_path) == ["deck.pptx"]


def test_discovers_files_inside_symlinked_directories(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    external_dir = tmp_path / "external-teams"
    _write(knowledge_dir / "example-knowledge.md")
    _write(external_dir / "meeting.vtt", "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\ncontent")
    (knowledge_dir / "teams").symlink_to(external_dir, target_is_directory=True)

    assert _relative_paths(_service(knowledge_dir), knowledge_dir) == ["teams/meeting.vtt"]
