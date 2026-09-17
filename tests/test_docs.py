"""Document flow tests: .docx parse/rebuild round-trip, table preservation,
batched cell rewriting with individual fallback. No model downloads —
a fake pipe stands in for generation; .docx fixtures are built with
stdlib zipfile (no new deps).
"""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dochumanize import (  # noqa: E402
    eligible_units,
    humanize_document,
    parse_docx,
    plain_preview,
    write_docx,
)
from generator import RewriteCandidate  # noqa: E402
from utils import SamplingParams, extract_payload  # noqa: E402

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _p(text: str) -> ET.Element:
    p = ET.Element(f"{{{W}}}p")
    r = ET.SubElement(p, f"{{{W}}}r")
    t = ET.SubElement(r, f"{{{W}}}t")
    t.text = text
    return p


def make_docx(path: Path) -> Path:
    """Minimal .docx: 65-word para, short heading, 2x2 table, normal para."""
    ET.register_namespace("w", W)
    body = ET.Element(f"{{{W}}}body")
    long_para = " ".join(f"word{i}" for i in range(65)) + "."
    body.append(_p(long_para))
    body.append(_p("Short head"))
    tbl = ET.SubElement(body, f"{{{W}}}tbl")
    ET.SubElement(tbl, f"{{{W}}}tblPr")
    grid = ET.SubElement(tbl, f"{{{W}}}tblGrid")
    for _ in range(2):
        ET.SubElement(grid, f"{{{W}}}gridCol", {"{{{}}}w".format(W): "3000"})
    rows = [["Name", "2024-01-01"],
            ["The quick brown fox jumps over the lazy dog today indeed.", "42"]]
    for row in rows:
        tr = ET.SubElement(tbl, f"{{{W}}}tr")
        for cell in row:
            tc = ET.SubElement(tr, f"{{{W}}}tc")
            tc.append(_p(cell))
    body.append(_p("Final paragraph with plenty of ordinary words filling space for the test."))
    doc = ET.Element(f"{{{W}}}document")
    doc.append(body)
    ctypes = ('<?xml version="1.0" encoding="UTF-8"?>'
              '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
              '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.'
              'relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>'
              '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.'
              'wordprocessingml.document.main+xml"/></Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
            'officeDocument" Target="word/document.xml"/></Relationships>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", ctypes)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", ET.tostring(doc, encoding="UTF-8", xml_declaration=True))
    return path


class _FakeCritic:
    def human_likeness_score(self, text):
        return 0.9 if len(text.split()) > 6 else 0.2

    def explain(self, text):
        return SimpleNamespace(guidance=lambda: "")


class _FakeRewriter:
    def rewrite(self, payload, params=None, num_candidates=1, guidance=None,
                extra_instruction=None):
        assert extra_instruction, "batch path must request line-by-line mode"
        lines = []
        for ln in payload.split("\n"):
            ln = re.sub(r"^\d+[.)]\s+", "", ln).strip()
            if ln:
                lines.append(" ".join(reversed(ln.split())))
        return [RewriteCandidate(text="\n".join(lines),
                                 params=params or SamplingParams())]


class _BadBatchRewriter(_FakeRewriter):
    def rewrite(self, payload, params=None, num_candidates=1, guidance=None,
                extra_instruction=None):
        if payload.strip().count("\n") >= 1:
            return [RewriteCandidate(text="single muddled line",
                                     params=params or SamplingParams())]
        return super().rewrite(payload, params, num_candidates, guidance, extra_instruction)


class _FakePipe:
    def __init__(self, rewriter=None):
        self.rewriter = rewriter or _FakeRewriter()
        self.critic = _FakeCritic()

    def run(self, raw, **kwargs):
        payload = extract_payload(raw)
        best = " ".join(reversed(payload.split()))
        return SimpleNamespace(best_text=best, best_score=0.9,
                               best_similarity=1.0, best_is_copy=False)


RUN_KWARGS = {"target_score": 0.85, "max_iters": 1, "num_candidates": 1,
              "min_similarity": 0.35, "polish_passes": 0}


def test_docx_round_trip_preserves_table(tmp_path):
    src = make_docx(tmp_path / "in.docx")
    doc = parse_docx(src)
    assert doc.n_paragraphs == 3 and doc.n_tables == 1
    units = eligible_units(doc)
    # 65-word para + normal para + long cell; heading/labels/dates skipped
    assert len(units) == 3, [u["label"] for u in units]
    from dochumanize import para_text

    long_before = para_text([b[1] for b in doc.blocks if b[0] == "p"][0])

    report = humanize_document(_FakePipe(), doc, RUN_KWARGS)
    assert all(r["action"] == "rewritten" for r in report), report

    out = write_docx(doc, tmp_path / "out.docx")
    again = parse_docx(out)
    assert again.n_tables == 1 and again.n_paragraphs == 3
    table = [b[1] for b in again.blocks if b[0] == "table"][0]
    assert len(table.rows) == 2 and all(len(r) == 2 for r in table.rows)
    texts = [[c.text for c in row] for row in table.rows]
    assert texts[0] == ["Name", "2024-01-01"]  # labels untouched
    assert texts[1][1] == "42"  # numeric untouched
    assert texts[1][0] != "The quick brown fox jumps over the lazy dog today indeed."
    paras = [b[1] for b in again.blocks if b[0] == "p"]
    assert para_text(paras[1]) == "Short head"  # heading untouched
    assert para_text(paras[0]) != long_before  # long para went through the pipe
    # structure intact: grid + entries preserved
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        assert {"[Content_Types].xml", "_rels/.rels", "word/document.xml"} <= names
        xml = zf.read("word/document.xml").decode("utf-8")
        assert xml.count("gridCol") == 2
    assert "[Table 1]" in plain_preview(again) and " | " in plain_preview(again)


def test_batch_failure_falls_back_to_individual(tmp_path):
    src = make_docx(tmp_path / "in.docx")
    doc = parse_docx(src)
    report = humanize_document(_FakePipe(_BadBatchRewriter()), doc, RUN_KWARGS)
    rewritten = [r for r in report if r["action"] == "rewritten"]
    # batch mismatched, but every unit recovered via the individual path
    assert len(rewritten) == 3, report


def test_parse_rejects_garbage(tmp_path):
    bad = tmp_path / "note.txt"
    bad.write_text("just text", encoding="utf-8")
    try:
        parse_docx(bad)
        raise AssertionError("should have raised")
    except ValueError:
        pass


def _sdt_docx(path: Path) -> Path:
    """Doc with sdt-wrapped para + table containing a nested table."""
    ET.register_namespace("w", W)

    def p(text):
        return _p(text)

    body = ET.Element(f"{{{W}}}body")
    sdt = ET.SubElement(body, f"{{{W}}}sdt")
    content = ET.SubElement(sdt, f"{{{W}}}sdtContent")
    content.append(p("Wrapped paragraph with enough words to count as content here today."))
    tbl = ET.SubElement(body, f"{{{W}}}tbl")
    tr = ET.SubElement(tbl, f"{{{W}}}tr")
    tc1 = ET.SubElement(tr, f"{{{W}}}tc")
    tc1.append(p("Outer cell with enough words to qualify for processing today."))
    tc2 = ET.SubElement(tr, f"{{{W}}}tc")
    tc2.append(p("x"))
    nested = ET.SubElement(tc2, f"{{{W}}}tbl")
    ntr = ET.SubElement(nested, f"{{{W}}}tr")
    ntc = ET.SubElement(ntr, f"{{{W}}}tc")
    ntc.append(p("Inner nested cell with enough words to qualify today."))
    doc = ET.Element(f"{{{W}}}document")
    doc.append(body)
    ctypes = ('<?xml version="1.0" encoding="UTF-8"?>'
              '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
              '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.'
              'relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>'
              '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.'
              'wordprocessingml.document.main+xml"/></Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
            'officeDocument" Target="word/document.xml"/></Relationships>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", ctypes)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", ET.tostring(doc, encoding="UTF-8", xml_declaration=True))
    return path


def test_parse_handles_sdt_and_nested_tables(tmp_path):
    from dochumanize import eligible_units

    doc = parse_docx(_sdt_docx(tmp_path / "sdt.docx"))
    kinds = [b[0] for b in doc.blocks]
    assert kinds == ["p", "table", "table"]  # sdt para, outer table, nested table
    units = eligible_units(doc)
    assert len(units) == 3, [u["label"] for u in units]  # sdt para + both long cells
