from __future__ import annotations

import indexer


def test_empty_project_does_not_load_embedding_model(tmp_path, monkeypatch):
    called = False

    def _unexpected_embedding_load():
        nonlocal called
        called = True
        raise AssertionError("embedding model must not load for an empty project")

    monkeypatch.setattr(indexer, "get_embeddings", _unexpected_embedding_load)

    indexer.index_codebase(str(tmp_path), db_path=str(tmp_path / "db"), quiet=True)

    assert called is False


def test_quiet_nonempty_index_has_a_noop_progress_callback(tmp_path, monkeypatch):
    (tmp_path / "app.py").write_text("print('hello')", encoding="utf-8")

    class _Store:
        def __init__(self, **_kwargs):
            pass

        def delete(self, **_kwargs):
            pass

        def add_documents(self, _documents):
            pass

    class _Splitter:
        def __init__(self, **_kwargs):
            pass

        def split_text(self, _content):
            return ["content"]

        def create_documents(self, _splits, metadatas):
            return []

    monkeypatch.setattr(indexer, "get_embeddings", lambda: object())
    monkeypatch.setattr(indexer, "Chroma", _Store)
    monkeypatch.setattr(indexer, "RecursiveCharacterTextSplitter", _Splitter)

    indexer.index_codebase(str(tmp_path), db_path=str(tmp_path / "db"), quiet=True)
