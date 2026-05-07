"""Tests for ``contract_question_agent.cuad_downloader``.

All network access is faked. The unit tests inject a custom ``url_opener``;
the CLI tests monkeypatch ``urllib.request.urlopen``. No test reaches the
real network.
"""

from __future__ import annotations

import urllib.error
from pathlib import Path

import pytest

from contract_question_agent.cuad_downloader import (
    DEFAULT_OUTPUTS,
    DEFAULT_SOURCE,
    SOURCE_URLS,
    CuadDownloadError,
    default_output_for,
    download_cuad,
    main,
    resolve_source_url,
)


# --------------------------------------------------------------------------- #
# Fake response / opener helpers
# --------------------------------------------------------------------------- #


class _FakeResponse:
    """Minimal stand-in for the file-like object returned by urlopen()."""

    def __init__(self, payload: bytes):
        self._payload = payload
        self._pos = 0
        self.read_calls = 0

    def read(self, n: int = -1) -> bytes:
        self.read_calls += 1
        if self._pos >= len(self._payload):
            return b""
        if n is None or n < 0:
            data = self._payload[self._pos:]
            self._pos = len(self._payload)
            return data
        data = self._payload[self._pos : self._pos + n]
        self._pos += len(data)
        return data

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc_info) -> bool:
        return False


def _make_opener(payload: bytes, captured: list[str] | None = None):
    last_response: dict[str, _FakeResponse] = {}

    def opener(url: str) -> _FakeResponse:
        if captured is not None:
            captured.append(url)
        resp = _FakeResponse(payload)
        last_response["resp"] = resp
        return resp

    opener.last_response = last_response  # type: ignore[attr-defined]
    return opener


# --------------------------------------------------------------------------- #
# resolve_source_url
# --------------------------------------------------------------------------- #


class TestResolveSourceUrl:
    def test_huggingface_default(self):
        url = resolve_source_url("huggingface")
        assert url == (
            "https://huggingface.co/datasets/theatticusproject/cuad/"
            "resolve/main/CUAD_v1/CUAD_v1.json"
        )

    def test_zenodo_optional(self):
        url = resolve_source_url("zenodo")
        assert url.startswith("https://")
        assert url.endswith(".zip")

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="Unknown CUAD source"):
            resolve_source_url("dropbox")


def test_default_source_is_huggingface():
    assert DEFAULT_SOURCE == "huggingface"
    assert "huggingface" in SOURCE_URLS


class TestDefaultOutputs:
    def test_huggingface_default_is_json(self):
        assert DEFAULT_OUTPUTS["huggingface"] == Path("data/cuad/raw/CUAD_v1.json")
        assert default_output_for("huggingface") == Path("data/cuad/raw/CUAD_v1.json")

    def test_zenodo_default_is_zip(self):
        assert DEFAULT_OUTPUTS["zenodo"] == Path("data/cuad/raw/CUAD_v1.zip")
        assert default_output_for("zenodo") == Path("data/cuad/raw/CUAD_v1.zip")

    def test_default_extension_matches_source_payload(self):
        # The whole point of per-source defaults: extension must match payload.
        assert default_output_for("huggingface").suffix == ".json"
        assert default_output_for("zenodo").suffix == ".zip"

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="Unknown CUAD source"):
            default_output_for("dropbox")


# --------------------------------------------------------------------------- #
# download_cuad — happy path & semantics
# --------------------------------------------------------------------------- #


class TestDownloadCuad:
    def test_writes_payload_to_output(self, tmp_path):
        payload = b'{"data": [], "version": "1"}'
        out = tmp_path / "raw" / "CUAD_v1.json"
        captured: list[str] = []

        result = download_cuad(
            output=out,
            url_opener=_make_opener(payload, captured),
        )

        assert result == out
        assert out.read_bytes() == payload
        assert captured == [SOURCE_URLS["huggingface"]]

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "deeply" / "nested" / "CUAD_v1.json"
        download_cuad(output=out, url_opener=_make_opener(b"x"))
        assert out.exists()

    def test_skip_when_file_exists(self, tmp_path):
        out = tmp_path / "CUAD_v1.json"
        out.write_bytes(b"existing content")
        captured: list[str] = []

        result = download_cuad(
            output=out, url_opener=_make_opener(b"NEW", captured)
        )

        assert result == out
        assert out.read_bytes() == b"existing content"
        assert captured == []  # opener not called

    def test_force_overwrites_existing(self, tmp_path):
        out = tmp_path / "CUAD_v1.json"
        out.write_bytes(b"old")

        download_cuad(
            output=out, force=True, url_opener=_make_opener(b"new")
        )

        assert out.read_bytes() == b"new"

    def test_streams_in_chunks(self, tmp_path):
        payload = b"A" * 1000
        out = tmp_path / "out.json"
        opener = _make_opener(payload)

        download_cuad(output=out, chunk_size=128, url_opener=opener)

        resp = opener.last_response["resp"]
        # 1000 bytes / 128 chunks = 8 reads with data + 1 final empty read.
        assert resp.read_calls >= 8
        assert out.read_bytes() == payload

    def test_zenodo_source_used(self, tmp_path):
        captured: list[str] = []
        download_cuad(
            source="zenodo",
            output=tmp_path / "CUAD_v1.zip",
            url_opener=_make_opener(b"PK\x03\x04zip-bytes", captured),
        )
        assert captured == [SOURCE_URLS["zenodo"]]

    def test_unknown_source_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError):
            download_cuad(
                source="ftp", output=tmp_path / "x.json",
                url_opener=_make_opener(b""),
            )

    def test_no_output_uses_per_source_default(self, tmp_path, monkeypatch):
        # download_cuad(output=None) should resolve to default_output_for(source).
        monkeypatch.chdir(tmp_path)

        hf_path = download_cuad(url_opener=_make_opener(b"{}"))
        assert hf_path == Path("data/cuad/raw/CUAD_v1.json")
        assert (tmp_path / hf_path).read_bytes() == b"{}"

        zen_path = download_cuad(
            source="zenodo", url_opener=_make_opener(b"PK\x03\x04")
        )
        assert zen_path == Path("data/cuad/raw/CUAD_v1.zip")
        assert (tmp_path / zen_path).read_bytes() == b"PK\x03\x04"

    def test_explicit_output_overrides_per_source_default(self, tmp_path):
        # Explicit output must win even if it has a "wrong" extension for source.
        out = tmp_path / "custom-name.bin"
        result = download_cuad(
            source="zenodo", output=out, url_opener=_make_opener(b"PK")
        )
        assert result == out
        assert out.read_bytes() == b"PK"

    def test_logs_source_and_destination(self, tmp_path, caplog):
        out = tmp_path / "CUAD_v1.json"
        with caplog.at_level("INFO", logger="contract_question_agent.cuad_downloader"):
            download_cuad(output=out, url_opener=_make_opener(b"{}"))
        joined = " ".join(rec.message for rec in caplog.records)
        assert "source=huggingface" in joined
        assert str(out) in joined


# --------------------------------------------------------------------------- #
# download_cuad — failure paths
# --------------------------------------------------------------------------- #


class TestDownloadFailures:
    def test_network_error_wrapped(self, tmp_path):
        def bad_opener(url):
            raise urllib.error.URLError("connection refused")

        out = tmp_path / "CUAD_v1.json"
        with pytest.raises(CuadDownloadError) as excinfo:
            download_cuad(output=out, url_opener=bad_opener)

        assert "connection refused" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, urllib.error.URLError)
        assert not out.exists()
        # Partial file must not be left behind.
        assert not out.with_name(out.name + ".part").exists()

    def test_unexpected_exception_wrapped(self, tmp_path):
        def explode(url):
            raise RuntimeError("boom")

        out = tmp_path / "CUAD_v1.json"
        with pytest.raises(CuadDownloadError):
            download_cuad(output=out, url_opener=explode)
        assert not out.exists()

    def test_failed_download_does_not_leave_part_file(self, tmp_path):
        class _Boom:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self, n=-1):
                raise urllib.error.URLError("midstream failure")

        def opener(url):
            return _Boom()

        out = tmp_path / "CUAD_v1.json"
        with pytest.raises(CuadDownloadError):
            download_cuad(output=out, url_opener=opener)

        assert not out.exists()
        assert not out.with_name(out.name + ".part").exists()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


class TestCli:
    def test_cli_success(self, tmp_path, monkeypatch, capsys):
        payload = b'{"data": []}'
        captured: list[str] = []

        def fake_urlopen(url):
            captured.append(url)
            return _FakeResponse(payload)

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        out = tmp_path / "raw" / "CUAD_v1.json"
        rc = main(["--output", str(out)])

        assert rc == 0
        assert out.read_bytes() == payload
        assert captured == [SOURCE_URLS["huggingface"]]
        printed = capsys.readouterr().out.strip()
        assert printed == str(out)

    def test_cli_force_flag(self, tmp_path, monkeypatch):
        out = tmp_path / "CUAD_v1.json"
        out.write_bytes(b"old")
        monkeypatch.setattr(
            "urllib.request.urlopen", lambda url: _FakeResponse(b"new")
        )

        rc = main(["--output", str(out), "--force"])

        assert rc == 0
        assert out.read_bytes() == b"new"

    def test_cli_skip_when_exists(self, tmp_path, monkeypatch):
        out = tmp_path / "CUAD_v1.json"
        out.write_bytes(b"keep me")

        called: list[str] = []

        def fake_urlopen(url):
            called.append(url)
            return _FakeResponse(b"new")

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        rc = main(["--output", str(out)])

        assert rc == 0
        assert out.read_bytes() == b"keep me"
        assert called == []

    def test_cli_network_failure_returns_1(self, tmp_path, monkeypatch):
        def bad_urlopen(url):
            raise urllib.error.URLError("boom")

        monkeypatch.setattr("urllib.request.urlopen", bad_urlopen)
        out = tmp_path / "CUAD_v1.json"

        rc = main(["--output", str(out)])

        assert rc == 1
        assert not out.exists()

    def test_cli_rejects_unknown_source(self, tmp_path):
        with pytest.raises(SystemExit):
            main(["--source", "dropbox", "--output", str(tmp_path / "x.json")])

    def test_cli_default_output_huggingface(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "urllib.request.urlopen", lambda url: _FakeResponse(b'{"data": []}')
        )

        rc = main([])  # no --output, no --source

        assert rc == 0
        assert (tmp_path / "data/cuad/raw/CUAD_v1.json").exists()
        assert not (tmp_path / "data/cuad/raw/CUAD_v1.zip").exists()

    def test_cli_default_output_zenodo(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "urllib.request.urlopen", lambda url: _FakeResponse(b"PK\x03\x04")
        )

        rc = main(["--source", "zenodo"])  # no --output

        assert rc == 0
        assert (tmp_path / "data/cuad/raw/CUAD_v1.zip").exists()
        assert not (tmp_path / "data/cuad/raw/CUAD_v1.json").exists()

    def test_cli_explicit_output_overrides_default(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "urllib.request.urlopen", lambda url: _FakeResponse(b"PK")
        )
        explicit = tmp_path / "elsewhere" / "custom.bin"

        rc = main(["--source", "zenodo", "--output", str(explicit)])

        assert rc == 0
        assert explicit.read_bytes() == b"PK"
        assert not (tmp_path / "data/cuad/raw/CUAD_v1.zip").exists()

    def test_cli_zenodo_source(self, tmp_path, monkeypatch):
        captured: list[str] = []

        def fake_urlopen(url):
            captured.append(url)
            return _FakeResponse(b"PK")

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        out = tmp_path / "CUAD_v1.zip"

        rc = main(["--source", "zenodo", "--output", str(out)])

        assert rc == 0
        assert captured == [SOURCE_URLS["zenodo"]]
