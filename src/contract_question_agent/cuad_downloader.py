"""Optional CUAD downloader.

Fetches ``CUAD_v1.json`` from a public source so the loader can process it
locally. This module is intentionally separate from :mod:`cuad_loader`:

* The **loader** parses CUAD and emits filtered JSONL.
* The **downloader** only obtains the raw payload.

It does not load LLM prompts, generate verification questions, or produce
legal advice. CUAD itself is © The Atticus Project and licensed under
CC BY 4.0; see ``docs/data.md`` for attribution requirements.

Tests inject a fake ``url_opener`` so they never touch the network.
"""

from __future__ import annotations

import argparse
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, ContextManager, Final, Protocol

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #

SOURCE_URLS: Final[dict[str, str]] = {
    # Primary: the Atticus Project's HuggingFace dataset mirror, which serves
    # CUAD_v1.json directly (no zip wrapper) over HTTPS.
    "huggingface": (
        "https://huggingface.co/datasets/theatticusproject/cuad/"
        "resolve/main/CUAD_v1/CUAD_v1.json"
    ),
    # Optional: Zenodo hosts the original zipped release. Useful as a fallback;
    # callers must unzip it themselves before passing to the loader.
    "zenodo": "https://zenodo.org/record/4595826/files/CUAD_v1.zip",
}

DEFAULT_SOURCE: Final[str] = "huggingface"
DEFAULT_OUTPUT: Final[Path] = Path("data/cuad/raw/CUAD_v1.json")
DEFAULT_CHUNK_SIZE: Final[int] = 1 << 16  # 64 KiB


class _ReadableResponse(Protocol):
    def read(self, n: int = ..., /) -> bytes: ...


UrlOpener = Callable[[str], ContextManager[_ReadableResponse]]


class CuadDownloadError(RuntimeError):
    """Raised when CUAD cannot be downloaded (network error, write error, etc.)."""


# --------------------------------------------------------------------------- #
# Core download
# --------------------------------------------------------------------------- #


def resolve_source_url(source: str) -> str:
    """Return the canonical URL for ``source`` or raise ``ValueError``."""
    try:
        return SOURCE_URLS[source]
    except KeyError as exc:
        choices = ", ".join(sorted(SOURCE_URLS))
        raise ValueError(
            f"Unknown CUAD source {source!r}; available: {choices}"
        ) from exc


def download_cuad(
    *,
    source: str = DEFAULT_SOURCE,
    output: Path | str = DEFAULT_OUTPUT,
    force: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    url_opener: UrlOpener | None = None,
) -> Path:
    """Download CUAD from ``source`` to ``output``.

    Skips the download when ``output`` already exists, unless ``force=True``.
    Streams the response in ``chunk_size`` blocks and writes through an
    intermediate ``.part`` file so a failed download does not leave a usable
    partial artifact behind.

    Args:
        source: Key into :data:`SOURCE_URLS` (``"huggingface"`` or ``"zenodo"``).
        output: Destination path.
        force: When True, overwrite an existing ``output``.
        chunk_size: Bytes per ``read`` call while streaming.
        url_opener: Override for ``urllib.request.urlopen`` (used by tests).

    Returns:
        The destination path on success (whether downloaded or pre-existing).

    Raises:
        ValueError: Unknown ``source``.
        CuadDownloadError: Network or filesystem failure.
    """
    url = resolve_source_url(source)
    output_path = Path(output)

    if output_path.exists() and not force:
        logger.info(
            "CUAD already present at %s; skipping download (use --force to overwrite).",
            output_path,
        )
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading CUAD: source=%s url=%s -> %s", source, url, output_path)

    opener: UrlOpener = url_opener or urllib.request.urlopen
    tmp_path = output_path.with_name(output_path.name + ".part")

    try:
        with opener(url) as resp:
            with tmp_path.open("wb") as out_fp:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out_fp.write(chunk)
        tmp_path.replace(output_path)
    except urllib.error.URLError as exc:
        _cleanup(tmp_path)
        raise CuadDownloadError(
            f"Network failure while downloading CUAD from {url}: {exc}"
        ) from exc
    except OSError as exc:
        _cleanup(tmp_path)
        raise CuadDownloadError(
            f"Failed to write CUAD payload to {output_path}: {exc}"
        ) from exc
    except Exception as exc:
        _cleanup(tmp_path)
        raise CuadDownloadError(
            f"Unexpected error while downloading CUAD from {url}: {exc}"
        ) from exc

    logger.info("Downloaded CUAD to %s", output_path)
    return output_path


def _cleanup(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:  # pragma: no cover - best-effort cleanup
        logger.debug("Could not remove partial download at %s", path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cuad-downloader",
        description=(
            "Download CUAD_v1.json for local processing. Data layer only — "
            "this tool does not provide legal advice."
        ),
    )
    parser.add_argument(
        "--source",
        choices=sorted(SOURCE_URLS),
        default=DEFAULT_SOURCE,
        help=f"Source to download from (default: {DEFAULT_SOURCE}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable INFO-level logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        path = download_cuad(
            source=args.source,
            output=args.output,
            force=args.force,
        )
    except CuadDownloadError as exc:
        logger.error("%s", exc)
        return 1
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
