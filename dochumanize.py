"""Document flow: upload .docx, keep tables, humanize the content inside.

A .docx is a zip of XML — parsed here with stdlib only (no new deps).
Structure contract:

* Top-level paragraphs and tables are found in document order, including
  content wrapped in structured-document tags (Word loves those) and
  tables nested inside table cells (processed as their own blocks).
* Tables keep every structural element (grid, widths, styles); only the
  text inside cells is replaced.
* Short cells (labels, numbers, dates) and short headings are kept as-is:
  a two-word label carries no AI-style signal worth the rewrite time and
  rewriting it risks breaking meaning ("Date" -> "Daytime"? no).
* Run-level formatting inside a rewritten paragraph (mid-sentence bold) is
  flattened to plain text; paragraph-level formatting is preserved.

Speed: table cells are short, so they are rewritten in line-batches
(one generation for many cells) with per-line validation and individual
fallback. Long paragraphs go through the full pipeline one by one.
"""
from __future__ import annotations

import logging
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _qn(tag: str) -> str:
    """Clark notation for a WordprocessingML tag ('w:body' or 'body')."""
    return f"{{{W_NS}}}{tag.split(':')[-1]}"


ET.register_namespace("w", W_NS)

# Policy (processing rules, not detection verdicts — those stay learned).
CELL_MIN_WORDS = 4      # shorter cells (labels/numbers/dates) are kept
PARA_MIN_WORDS = 8      # shorter paragraphs (headings) are kept
LONG_UNIT_WORDS = 60    # longer units go through the full pipeline alone
BATCH_MAX_CHARS = 500   # short-unit batch budget per generation
_NUMERIC_RE = re.compile(r"^[\d\s,.$%/\-–—:;()×x+]+$")
_LINE_NUM_RE = re.compile(r"^\d+[.)]\s+")


@dataclass
class CellRef:
    tc_elem: ET.Element
    para_elems: list[ET.Element]
    text: str
    row: int
    col: int


@dataclass
class TableRef:
    tbl_elem: ET.Element
    rows: list[list[CellRef]]


@dataclass
class ParsedDocx:
    """Live document: tree is mutated in place, then written back."""

    source: Path
    tree: ET.ElementTree
    blocks: list = field(default_factory=list)  # ("p", elem) | ("table", TableRef)

    @property
    def n_paragraphs(self) -> int:
        return sum(1 for b in self.blocks if b[0] == "p")

    @property
    def n_tables(self) -> int:
        return sum(1 for b in self.blocks if b[0] == "table")


def para_text(p_elem: ET.Element) -> str:
    """All run text in a paragraph (hyperlinks flattened)."""
    return "".join(t.text or "" for t in p_elem.iter(_qn("w:t"))).strip()


def set_para_text(p_elem: ET.Element, text: str) -> None:
    """Replace paragraph content with one plain run (keeps pPr)."""
    for r in list(p_elem.findall(_qn("w:r"))):
        p_elem.remove(r)
    for link in list(p_elem.findall(_qn("w:hyperlink"))):
        for r in list(link.findall(_qn("w:r"))):
            link.remove(r)
    run = ET.SubElement(p_elem, _qn("w:r"))
    t = ET.SubElement(run, _qn("w:t"))
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text


def _table_ref(tbl_elem: ET.Element) -> TableRef | None:
    rows: list[list[CellRef]] = []
    for ri, tr in enumerate(tbl_elem.findall(_qn("w:tr"))):
        cells: list[CellRef] = []
        for ci, tc in enumerate(tr.findall(_qn("w:tc"))):
            paras = tc.findall(_qn("w:p"))
            text = " ".join(para_text(p) for p in paras).strip()
            if text:
                cells.append(CellRef(tc, paras, text, ri, ci))
        if cells:
            rows.append(cells)
    return TableRef(tbl_elem, rows) if rows else None


def parse_docx(path: str | Path) -> ParsedDocx:
    """Parse body paragraphs + tables, in document order.

    Recurses into structured-document tags (w:sdt) that Word uses to wrap
    content, and treats tables nested in cells as their own blocks (cell
    text only ever covers direct paragraphs, so nothing gets corrupted).
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Document not found: {src}")
    try:
        with zipfile.ZipFile(src) as zf:
            raw = zf.read("word/document.xml")
    except Exception as exc:
        raise ValueError(
            f"Not a readable .docx file: {src} ({exc}). "
            "Old .doc files and PDFs are not supported — save/convert as .docx first."
        ) from exc
    tree = ET.ElementTree(ET.fromstring(raw))
    body = tree.getroot().find(_qn("w:body"))
    if body is None:
        raise ValueError(f"No document body found in {src}")
    blocks: list = []

    def walk(parent: ET.Element) -> None:
        for child in list(parent):
            if child.tag == _qn("w:p"):
                if para_text(child):
                    blocks.append(("p", child))
            elif child.tag == _qn("w:tbl"):
                ref = _table_ref(child)
                if ref:
                    blocks.append(("table", ref))
                for nested in child.iter(_qn("w:tbl")):
                    if nested is child:
                        continue
                    nref = _table_ref(nested)
                    if nref:
                        blocks.append(("table", nref))
            elif child.tag == _qn("w:sdt"):
                content = child.find(_qn("w:sdtContent"))
                if content is not None:
                    walk(content)

    walk(body)
    if not blocks:
        n_p = len(body.findall(f".//{_qn('w:p')}"))
        n_tbl = len(body.findall(f".//{_qn('w:tbl')}"))
        raise ValueError(
            f"No usable text in {src} (saw {n_p} paragraph and {n_tbl} table tags — "
            "the file may use an unsupported layout).")
    return ParsedDocx(source=src, tree=tree, blocks=blocks)


def write_docx(doc: ParsedDocx, out_path: str | Path) -> Path:
    """Write the (mutated) document, copying every other zip entry intact."""
    out = Path(out_path)
    body_xml = ET.tostring(doc.tree.getroot(), encoding="UTF-8", xml_declaration=True)
    with zipfile.ZipFile(doc.source) as zin:
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = body_xml if item.filename == "word/document.xml" else zin.read(item.filename)
                zout.writestr(item, data)
    logger.info("dochumanize: wrote %s", out)
    return out


def _words(text: str) -> list[str]:
    return text.split()


def eligible_units(doc: ParsedDocx) -> list[dict]:
    """Collect humanizable units: long-enough paras + cells (skip numeric)."""
    units: list[dict] = []
    for bi, (kind, ref) in enumerate(doc.blocks):
        if kind == "p":
            text = para_text(ref)
            if len(_words(text)) >= PARA_MIN_WORDS:
                units.append({"kind": "para", "ref": ref, "text": text,
                              "label": f"para {bi + 1}"})
        else:
            for row in ref.rows:
                for cell in row:
                    if len(_words(cell.text)) < CELL_MIN_WORDS:
                        continue
                    if _NUMERIC_RE.match(cell.text):
                        continue
                    units.append({"kind": "cell", "ref": cell, "text": cell.text,
                                  "label": f"table R{cell.row + 1}C{cell.col + 1}"})
    return units


def _apply_unit_text(unit: dict, new_text: str) -> None:
    new_text = " ".join(new_text.split())
    if not new_text:
        return
    if unit["kind"] == "para":
        set_para_text(unit["ref"], new_text)
    else:
        cell: CellRef = unit["ref"]
        set_para_text(cell.para_elems[0], new_text)
        for extra in cell.para_elems[1:]:
            set_para_text(extra, "")


def humanize_document(pipe, doc: ParsedDocx, run_kwargs: dict,
                      progress: Callable[[int, int, str], None] | None = None) -> list[dict]:
    """Humanize every eligible unit; mutate doc in place; return a report.

    `pipe` needs `.run(wrapped, **run_kwargs)`, `.rewriter.rewrite(...)`,
    and `.critic` (score + explain().guidance()). Units above
    LONG_UNIT_WORDS go through the full pipeline; short ones are batched
    line-by-line (one generation, per-line validation, individual fallback).
    """
    from utils import is_near_copy, lexical_similarity, wrap_payload

    from pipeline import MAX_LENGTH_RATIO

    floor = run_kwargs.get("min_similarity", 0.35)
    units = eligible_units(doc)
    total = len(units)
    report: list[dict] = []
    long = [u for u in units if len(_words(u["text"])) >= LONG_UNIT_WORDS]
    short = [u for u in units if len(_words(u["text"])) < LONG_UNIT_WORDS]

    def tick(label: str):
        if progress:
            progress(len(report), total, label)

    for u in long:
        tick(u["label"])
        try:
            res = pipe.run(wrap_payload(u["text"]), **run_kwargs)
            if res.best_is_copy or res.best_similarity < floor:
                raise RuntimeError("no acceptable rewrite")
            before = float(pipe.critic.human_likeness_score(u["text"]))
            _apply_unit_text(u, res.best_text)
            report.append({"label": u["label"], "action": "rewritten",
                           "before": round(before, 3), "after": round(res.best_score, 3),
                           "sim": round(res.best_similarity, 3)})
        except Exception as exc:
            logger.warning("dochumanize: %s kept original (%s)", u["label"], exc)
            report.append({"label": u["label"], "action": f"kept ({exc})"})

    batches: list[list[dict]] = []
    current, chars = [], 0
    for u in short:
        size = len(u["text"]) + 8
        if current and chars + size > BATCH_MAX_CHARS:
            batches.append(current)
            current, chars = [], 0
        current.append(u)
        chars += size
    if current:
        batches.append(current)

    for batch in batches:
        tick(f"batch ({len(batch)} cells/lines)")
        numbered = [f"{i + 1}. {u['text']}" for i, u in enumerate(batch)]
        try:
            guide = ""
            try:
                guide = pipe.critic.explain("\n".join(numbered)).guidance() or ""
            except Exception:
                pass
            outs = pipe.rewriter.rewrite(
                "\n".join(numbered), num_candidates=1, guidance=guide or None,
                extra_instruction=("Keep the response line by line: output exactly one "
                                   "rewritten line per input line, in the same order, "
                                   "with no extra commentary."))
            lines = [ln.strip() for ln in outs[0].text.split("\n")]
            lines = [_LINE_NUM_RE.sub("", ln).strip() for ln in lines if ln.strip()]
        except Exception as exc:
            logger.warning("dochumanize: batch failed (%s); trying lines individually", exc)
            lines = []
        if len(lines) != len(batch):
            lines = []
        for u, new in zip(batch, lines or [None] * len(batch)):
            if (new and not is_near_copy(u["text"], new)
                    and lexical_similarity(u["text"], new) >= floor
                    and len(_words(new)) <= MAX_LENGTH_RATIO * max(len(_words(u["text"])), 1)):
                before = float(pipe.critic.human_likeness_score(u["text"]))
                after = float(pipe.critic.human_likeness_score(new))
                _apply_unit_text(u, new)
                report.append({"label": u["label"], "action": "rewritten",
                               "before": round(before, 3), "after": round(after, 3),
                               "sim": round(lexical_similarity(u["text"], new), 3)})
                tick(u["label"])
                continue
            # Individual fallback through the full pipeline.
            tick(u["label"])
            try:
                res = pipe.run(wrap_payload(u["text"]), **run_kwargs)
                if res.best_is_copy or res.best_similarity < floor:
                    raise RuntimeError("no acceptable rewrite")
                before = float(pipe.critic.human_likeness_score(u["text"]))
                _apply_unit_text(u, res.best_text)
                report.append({"label": u["label"], "action": "rewritten",
                               "before": round(before, 3), "after": round(res.best_score, 3),
                               "sim": round(res.best_similarity, 3)})
            except Exception as exc:
                logger.warning("dochumanize: %s kept original (%s)", u["label"], exc)
                report.append({"label": u["label"], "action": f"kept ({exc})"})
    return report


def plain_preview(doc: ParsedDocx) -> str:
    """Readable text snapshot (tables as | separated rows) for display."""
    out: list[str] = []
    ti = 0
    for kind, ref in doc.blocks:
        if kind == "p":
            out.append(para_text(ref))
        else:
            ti += 1
            out.append(f"[Table {ti}]")
            for row in ref.rows:
                cells = []
                for cell in row:
                    text = " ".join(para_text(p) for p in cell.para_elems if para_text(p))
                    cells.append(text)
                out.append(" | ".join(cells))
    return "\n\n".join(out)
