"""Tests for RAGAnythingParser's handling of MinerU (magic-pdf 0.6.1) output.

These tests are hermetic and do not need magic_pdf: the parser is created with
``RAGAnythingParser.__new__`` and UNIPipe / DiskReaderWriter are replaced by fakes
that mirror magic-pdf 0.6.1 (``pipe.pdf_mid_data["pdf_info"]`` page dicts, crops
written by ``cut_image`` as ``<sha256>.jpg`` relative to the image writer root).
"""

import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from app.models.parsing import ParsedDocument
from app.parsers.rag_anything_parser import RAGAnythingParser
from app.processing.chunker import SemanticChunker

FIGURE_BYTES = b"\xff\xd8\xff\xe0 figure crop \xff\xd9"
TABLE_BYTES = b"\xff\xd8\xff\xe0 table crop \xff\xd9"

FIGURE_BBOX = [310.0, 402.0, 540.0, 520.0]
TABLE_BBOX = [60.0, 110.0, 540.0, 300.0]


def crop_name(pdf_md5: str, kind: str, page_idx: int, bbox: list[float]) -> str:
    """Name of a crop as produced by magic_pdf.libs.pdf_image_tools.cut_image."""
    key = f"{pdf_md5}/{kind}/{page_idx}_" + "_".join(str(int(v)) for v in bbox)
    return f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.jpg"


FIGURE_NAME = crop_name("0" * 32, "images", 0, FIGURE_BBOX)
TABLE_NAME = crop_name("0" * 32, "tables", 1, TABLE_BBOX)


def span(content: str, span_type: str = "text", bbox: list[float] | None = None) -> dict:
    return {"bbox": bbox or [0.0, 0.0, 10.0, 10.0], "content": content, "type": span_type}


def line(*spans: dict) -> dict:
    return {"bbox": [0.0, 0.0, 10.0, 10.0], "spans": list(spans)}


def block(block_type: str, lines: list[dict], bbox: list[float], **extra: Any) -> dict:
    return {"type": block_type, "bbox": bbox, "lines": lines, **extra}


def page(page_idx: int, para_blocks: list[dict], **extra: Any) -> dict:
    """A pdf_info page dict with the keys magic-pdf 0.6.1 produces."""
    images = [b for b in para_blocks if b["type"] == "image"]
    tables = [b for b in para_blocks if b["type"] == "table"]
    equations = [b for b in para_blocks if b["type"] == "interline_equation"]
    info = {
        "preproc_blocks": para_blocks,
        "layout_bboxes": [],
        "page_idx": page_idx,
        "page_size": [612.0, 792.0],
        "_layout_tree": [],
        "images": images,
        "tables": tables,
        "interline_equations": equations,
        "discarded_blocks": [],
        "need_drop": False,
        "drop_reason": [],
        "para_blocks": para_blocks,
    }
    info.update(extra)
    return info


def build_pdf_info(figure_name: str, table_name: str) -> list[dict]:
    """Three pages: rich first page, table page (flagged need_drop), empty last page."""
    figure_block = {
        "type": "image",
        "bbox": [300.0, 400.0, 550.0, 560.0],
        "blocks": [
            {
                "type": "image_body",
                "bbox": FIGURE_BBOX,
                "lines": [
                    {
                        "bbox": FIGURE_BBOX,
                        "spans": [
                            {
                                "bbox": FIGURE_BBOX,
                                "score": 0.97,
                                "type": "image",
                                "image_path": figure_name,
                            }
                        ],
                    }
                ],
            },
            block(
                "image_caption",
                [line(span("Figure 2. Residual learning: a building block."))],
                [305.0, 525.0, 545.0, 556.0],
            ),
        ],
    }
    table_block = {
        "type": "table",
        "bbox": [55.0, 90.0, 545.0, 320.0],
        "blocks": [
            {
                "type": "table_body",
                "bbox": TABLE_BBOX,
                "lines": [
                    {
                        "bbox": TABLE_BBOX,
                        "spans": [
                            {
                                "bbox": TABLE_BBOX,
                                "score": 0.95,
                                "type": "table",
                                "image_path": table_name,
                            }
                        ],
                    }
                ],
            },
            block(
                "table_caption",
                [line(span("Table 1. Architectures for ImageNet."))],
                [60.0, 92.0, 540.0, 106.0],
            ),
            block(
                "table_footnote",
                [line(span("Building blocks are shown in brackets."))],
                [60.0, 302.0, 540.0, 318.0],
            ),
        ],
    }

    page0 = page(
        0,
        [
            block(
                "title",
                [line(span("Deep Residual Learning for Image Recognition"))],
                [120.0, 70.0, 490.0, 90.0],
            ),
            block("title", [line(span("1. Introduction"))], [50.0, 120.0, 150.0, 132.0]),
            block(
                "text",
                [
                    line(
                        span("We denote the underlying mapping as"),
                        span("\\mathcal{H}(x)", "inline_equation"),
                        span("and let the stacked"),
                    ),
                    line(
                        span("layers fit another mapping of"),
                        span("\\mathcal{F}(x) := \\mathcal{H}(x) - x", "inline_equation"),
                    ),
                ],
                [50.0, 140.0, 290.0, 180.0],
            ),
            block(
                "interline_equation",
                [
                    line(
                        span(
                            "\\mathbf{y} = \\mathcal{F}(\\mathbf{x}) + \\mathbf{x}",
                            "interline_equation",
                        )
                    )
                ],
                [80.0, 190.0, 260.0, 205.0],
            ),
            figure_block,
        ],
    )
    page1 = page(
        1,
        [
            # First paragraph was merged into the previous page by para_split
            block("text", [], [50.0, 60.0, 290.0, 80.0], lines_deleted=True),
            table_block,
            block(
                "text",
                [line(span("Our 152-layer residual net has lower complexity than VGG nets."))],
                [50.0, 330.0, 290.0, 350.0],
            ),
        ],
        need_drop=True,
        drop_reason=["complicated_layout"],
    )
    page2 = page(2, [])
    return [page0, page1, page2]


def make_parser(output_dir: Path | None = None) -> RAGAnythingParser:
    """Create a parser without importing magic_pdf (constructor bypassed)."""
    parser = RAGAnythingParser.__new__(RAGAnythingParser)
    parser.logger = logging.getLogger("test_rag_anything_parser")
    parser.output_dir = output_dir
    parser.UNIPipe = None
    parser.DiskReaderWriter = FakeDiskReaderWriter
    return parser


class FakeDiskReaderWriter:
    """Mirrors magic_pdf.rw.DiskReaderWriter.write for relative paths."""

    def __init__(self, parent_path: str, encoding: str = "utf-8") -> None:
        self.path = parent_path
        self.encoding = encoding

    def write(self, content: Any, path: str, mode: str = "text") -> None:
        abspath = os.path.join(self.path, path)
        os.makedirs(os.path.dirname(abspath), exist_ok=True)
        if mode == "binary":
            with open(abspath, "wb") as f:
                f.write(content)
        else:
            with open(abspath, "w", encoding=self.encoding) as f:
                f.write(content)


class FakeUNIPipe:
    """Stands in for magic_pdf.pipe.UNIPipe.UNIPipe (0.6.1)."""

    created: list["FakeUNIPipe"] = []

    def __init__(
        self, pdf_bytes: bytes, jso_useful_key: dict, image_writer: Any, is_debug: bool = False
    ) -> None:
        self.pdf_bytes = pdf_bytes
        self.pdf_type = jso_useful_key["_pdf_type"]
        self.model_list = jso_useful_key["model_list"]
        self.image_writer = image_writer
        self.is_debug = is_debug
        self.pdf_mid_data: dict | None = None
        self.calls: list[str] = []
        FakeUNIPipe.created.append(self)

    def pipe_classify(self) -> None:
        self.calls.append("classify")
        self.pdf_type = "txt"

    def pipe_analyze(self) -> None:
        self.calls.append("analyze")
        self.model_list = [{"layout_dets": [], "page_info": {"page_no": 0}}]

    def pipe_parse(self) -> None:
        # Like pdf_parse_union: crops go through the image writer as "<sha256>.jpg"
        self.calls.append("parse")
        pdf_md5 = hashlib.md5(self.pdf_bytes).hexdigest().upper()
        figure_name = crop_name(pdf_md5, "images", 0, FIGURE_BBOX)
        table_name = crop_name(pdf_md5, "tables", 1, TABLE_BBOX)
        self.image_writer.write(FIGURE_BYTES, figure_name, "binary")
        self.image_writer.write(TABLE_BYTES, table_name, "binary")
        self.pdf_mid_data = {
            "pdf_info": build_pdf_info(figure_name, table_name),
            "_parse_type": "txt",
            "_version_name": "0.6.1",
        }

    def pipe_mk_uni_format(self, *_args: Any, **_kwargs: Any) -> list:
        raise AssertionError("pipe_mk_uni_format() returns a flat, page-less list")

    def pipe_mk_markdown(self, *_args: Any, **_kwargs: Any) -> str:
        raise AssertionError("pipe_mk_markdown() is not expected to be used")


@pytest.fixture
def image_root(tmp_path: Path) -> Path:
    """Image writer root holding the crops referenced by the fixture."""
    root = tmp_path / "images" / "paper"
    root.mkdir(parents=True)
    (root / FIGURE_NAME).write_bytes(FIGURE_BYTES)
    (root / TABLE_NAME).write_bytes(TABLE_BYTES)
    return root


@pytest.fixture
def converted(image_root: Path) -> ParsedDocument:
    pdf_mid_data = {
        "pdf_info": build_pdf_info(FIGURE_NAME, TABLE_NAME),
        "_parse_type": "txt",
        "_version_name": "0.6.1",
    }
    return make_parser()._convert_to_parsed_document(
        pdf_mid_data, Path("/uploads/paper.pdf"), image_root
    )


def test_convert_extracts_text_blocks_in_reading_order(converted: ParsedDocument):
    texts = [(b.page, b.layout_type, b.content) for b in converted.text_blocks]

    assert texts == [
        (0, "heading", "Deep Residual Learning for Image Recognition"),
        (0, "heading", "1. Introduction"),
        (
            0,
            "paragraph",
            "We denote the underlying mapping as $\\mathcal{H}(x)$ and let the stacked "
            "layers fit another mapping of $\\mathcal{F}(x) := \\mathcal{H}(x) - x$",
        ),
        (0, "equation", "$$\\mathbf{y} = \\mathcal{F}(\\mathbf{x}) + \\mathbf{x}$$"),
        (0, "caption", "Figure 2. Residual learning: a building block."),
        (1, "caption", "Table 1. Architectures for ImageNet."),
        (1, "footnote", "Building blocks are shown in brackets."),
        (1, "paragraph", "Our 152-layer residual net has lower complexity than VGG nets."),
    ]
    assert converted.text_blocks[0].bbox == (120.0, 70.0, 490.0, 90.0)


def test_convert_counts_all_pages_and_keeps_zero_based_page_idx(converted: ParsedDocument):
    # The last page has no para_blocks but still counts; page_idx stays 0-based
    assert converted.num_pages == 3
    pages = [b.page for b in converted.text_blocks] + [i.page for i in converted.images]
    assert all(0 <= p < converted.num_pages for p in pages)
    assert {b.page for b in converted.text_blocks} == {0, 1}


def test_convert_reads_images_relative_to_image_writer_root(converted: ParsedDocument):
    figure, table_crop = converted.images

    assert figure.image_data == FIGURE_BYTES
    assert figure.page == 0
    assert figure.bbox == tuple(FIGURE_BBOX)
    assert figure.format == "jpeg"
    assert figure.visual_features == {
        "block_type": "image",
        "caption": "Figure 2. Residual learning: a building block.",
    }

    # Tables are only cropped images in magic-pdf 0.6.1; the crop is kept as an image
    assert table_crop.image_data == TABLE_BYTES
    assert table_crop.page == 1
    assert table_crop.visual_features == {
        "block_type": "table",
        "caption": "Table 1. Architectures for ImageNet.",
    }


def test_convert_builds_table_block_without_inventing_cells(converted: ParsedDocument):
    assert len(converted.tables) == 1
    table = converted.tables[0]
    assert table.page == 1
    assert table.bbox == (55.0, 90.0, 545.0, 320.0)
    assert table.rows == [[]]
    assert table.headers is None


def test_convert_metadata(converted: ParsedDocument):
    assert converted.metadata == {
        "parser": "mineru_unipipe",
        "version": "0.6.1",
        "parse_type": "txt",
        "source": "/uploads/paper.pdf",
    }


def test_convert_does_not_look_for_images_in_nested_images_dir(tmp_path: Path):
    # Crop stored where the old code looked (<root>/images/<name>) must not be found
    root = tmp_path / "root"
    (root / "images").mkdir(parents=True)
    (root / "images" / FIGURE_NAME).write_bytes(FIGURE_BYTES)

    doc = make_parser()._convert_to_parsed_document(
        {"pdf_info": build_pdf_info(FIGURE_NAME, TABLE_NAME)}, Path("paper.pdf"), root
    )

    assert doc.images == []
    # Captions are still extracted as text when the crop is missing
    assert "Figure 2. Residual learning: a building block." in [b.content for b in doc.text_blocks]


def test_convert_table_with_html_body(tmp_path: Path):
    table_block = {
        "type": "table",
        "bbox": [0.0, 0.0, 100.0, 50.0],
        "blocks": [
            {
                "type": "table_body",
                "bbox": [0.0, 0.0, 100.0, 50.0],
                "lines": [
                    {
                        "bbox": [0.0, 0.0, 100.0, 50.0],
                        "spans": [
                            {
                                "bbox": [0.0, 0.0, 100.0, 50.0],
                                "type": "table",
                                "html": "<table><tr><th>layer</th><th>output</th></tr>"
                                "<tr><td>conv1</td><td>112x112</td></tr></table>",
                            }
                        ],
                    }
                ],
            }
        ],
    }

    doc = make_parser()._convert_to_parsed_document(
        {"pdf_info": [page(0, [table_block])]}, Path("paper.pdf"), tmp_path
    )

    assert doc.tables[0].rows == [["layer", "output"], ["conv1", "112x112"]]
    assert doc.tables[0].headers == ["layer", "output"]
    assert doc.images == []  # no image_path on the body span


def test_convert_empty_pdf_info(tmp_path: Path):
    doc = make_parser()._convert_to_parsed_document({"pdf_info": []}, Path("x.pdf"), tmp_path)

    assert doc.num_pages == 0
    assert doc.text_blocks == []
    assert doc.images == []
    assert doc.tables == []


def test_converted_document_is_chunkable_with_headings(converted: ParsedDocument):
    chunks = SemanticChunker(chunk_size=512, chunk_overlap=50).chunk_with_structure(
        converted, doc_id="doc-1"
    )

    assert chunks
    assert "1. Introduction" in {c.metadata.get("section_heading") for c in chunks}
    assert any("Table 1. Architectures for ImageNet." in c.content for c in chunks)


def test_parse_pdf_uses_pdf_mid_data_from_pipe(tmp_path: Path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n% fake pdf for FakeUNIPipe\n%%EOF\n")
    output_dir = tmp_path / "out"
    parser = make_parser(output_dir)
    FakeUNIPipe.created = []
    parser.UNIPipe = FakeUNIPipe

    doc = parser.parse_pdf(pdf_path)

    (pipe,) = FakeUNIPipe.created
    assert pipe.calls == ["classify", "analyze", "parse"]
    assert pipe.image_writer.path == str(output_dir / "images" / "paper")

    assert doc.num_pages == 3
    assert [b.content for b in doc.text_blocks][:2] == [
        "Deep Residual Learning for Image Recognition",
        "1. Introduction",
    ]
    assert [(i.page, i.image_data) for i in doc.images] == [(0, FIGURE_BYTES), (1, TABLE_BYTES)]
    assert len(doc.tables) == 1
    assert doc.metadata["source"] == str(pdf_path)


def test_parse_pdf_raises_when_pipe_produces_no_result(tmp_path: Path):
    class EmptyPipe(FakeUNIPipe):
        def pipe_parse(self) -> None:
            self.calls.append("parse")

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
    parser = make_parser(tmp_path / "out")
    parser.UNIPipe = EmptyPipe

    with pytest.raises(RuntimeError, match="no parse result"):
        parser.parse_pdf(pdf_path)


def test_parse_pdf_turns_mineru_exit_into_runtime_error(tmp_path: Path):
    class BrokenModelPipe(FakeUNIPipe):
        def pipe_analyze(self) -> None:
            # magic_pdf.model.pp_structure_v2 does this when paddleocr fails to import
            exit(1)

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
    parser = make_parser(tmp_path / "out")
    parser.UNIPipe = BrokenModelPipe

    with pytest.raises(RuntimeError, match="MinerU aborted"):
        parser.parse_pdf(pdf_path)


def test_parse_pdf_validates_path(tmp_path: Path):
    parser = make_parser(tmp_path / "out")

    with pytest.raises(ValueError, match="File not found"):
        parser.parse_pdf(tmp_path / "missing.pdf")

    not_pdf = tmp_path / "notes.txt"
    not_pdf.write_bytes(b"Not a PDF")
    with pytest.raises(ValueError, match="not a PDF"):
        parser.parse_pdf(not_pdf)


def test_missing_mineru_raises_import_error_with_install_hint(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "magic_pdf", None)  # makes "import magic_pdf..." fail
    # setenv first so monkeypatch restores the original (unset) state after the
    # constructor sets MINERU_MODEL_SOURCE
    monkeypatch.setenv("MINERU_MODEL_SOURCE", "")
    monkeypatch.delenv("MINERU_MODEL_SOURCE")

    with pytest.raises(ImportError, match="magic-pdf\\[cpu\\]==0.6.1"):
        RAGAnythingParser()
