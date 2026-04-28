# Materials Converter — Design Spec

**Date:** 2026-04-28
**Author:** Polk Wagner (with Claude)
**Status:** Draft — pending review

---

## 1. Problem Statement

We have a single-format PDF→markdown converter (`pdf_to_markdown.py`) built around Docling, optimized for legal casebooks and academic PDFs. The gap: we routinely need the same clean, position-marked markdown output from DOCX memos, PPTX lecture decks, and HTML articles for AI ingestion (LLM context, semantic search, downstream skill chains like eddie/factual-pipeline-orchestrator).

**Success looks like:**
- Single CLI converts PDF, DOCX, PPTX, and HTML to markdown via auto-detection.
- Position markers are reliable and uniquely citeable (`<!-- Page N -->`, `<!-- Slide N -->`, `<!-- Section K: Title -->`).
- PPTX speaker notes are captured (per the user's explicit requirement).
- Cheap structural verification runs on every conversion and catches catastrophic silent failures (empty output, lost pages, missing notes); deep verification is opt-in.
- Existing PDF behavior is preserved exactly through the refactor (no regressions on the casebook corpus).

**Out of scope (this spec):** ePub, RTF, `.doc` (legacy Word), image OCR as a primary input, parallelism across files, deep-verifier rebuild beyond what currently exists. These are deferred until the core ships.

## 2. Empirical Foundations

Before writing this design, the installed Docling 2.65.0 was probed directly to verify what it does and does not expose. The findings below are **measured, not assumed**:

| Format | Feature | Docling 2.65.0 | Extra library needed? |
|---|---|---|---|
| PPTX | Speaker notes | Yes — extracted into `ContentLayer.FURNITURE`, excluded from default markdown export. Including `FURNITURE` in `included_content_layers` emits notes inline at slide-end. | **No** |
| DOCX | Headings, text, tables | Yes | No |
| DOCX | Comments | **No** — `grep` on `msword_backend.py` for `comment` returns zero matches | Yes — `python-docx` |
| DOCX | Footnotes/endnotes | **No** | Yes — `python-docx` |
| DOCX | Tracked changes | **No** | Yes — `python-docx` |
| HTML | Conversion to markdown | Yes | No (optional `beautifulsoup4` to pre-strip nav/script for sane retention ratios) |
| PDF | Pages, tables, headings | Yes (current code path) | PyMuPDF + RapidFuzz (existing) |

**Implication:** the dependency additions are leaner than originally proposed.
- `python-docx` — mandatory, only used inside `formats/docx.py` for comments/footnotes/revisions.
- `beautifulsoup4` — optional, declared in `requirements.txt` but only imported when `--strip-html-noise` flag is set.
- `python-pptx` — **not added**; Docling handles it.

## 3. Alternatives Considered

| Alternative | Why rejected |
|---|---|
| **Use only Docling, no extra deps.** | Loses DOCX comments, footnotes, tracked changes. The probe confirmed Docling's DOCX backend doesn't touch them. For legal/teaching memos, those are the most-edited bits — the workflow demands them. |
| **Pandoc instead of Docling.** | The current PDF pipeline (provenance-based page-marker insertion via Docling element tree → PyMuPDF fuzzy fallback) is the project's most valuable existing logic and depends on Docling's element-level provenance model. Switching engines discards that. Pandoc is also weaker on PDFs and does not produce element-level provenance. |
| **Microsoft `markitdown`.** | Younger, narrower feature set, no element-level provenance, no PDF page-label awareness. Worse on the central PDF use case. |
| **Per-format CLIs (`pdf-to-md`, `docx-to-md`, ...) instead of one unified `convert`.** | Fragments muscle memory; doesn't compose well in batch mode where a class folder has mixed extensions. |
| **Big-bang refactor (one PR with everything).** | Fragile — if any piece slips, all slip. Replaced with the staged plan in §8. |

## 4. Decisions (Locked)

These were settled through the brainstorming flow:

1. **Formats in scope:** PDF, DOCX, PPTX, HTML.
2. **CLI shape:** single unified `convert` command, auto-detected by extension, with format-specific flags allowed.
3. **Position markers (hybrid):**
   - PDF → `<!-- Page N -->` (using PDF page labels, current behavior)
   - PPTX → `<!-- Slide N -->`
   - DOCX → `<!-- Section K: <heading-text> -->` (numbered, see §5.2)
   - HTML → `<!-- Section K: <heading-text> -->` (numbered)
4. **Speaker notes:** inline at end of each slide, marked `<!-- Speaker notes -->`. `--notes-only` flag emits a clean transcript with slide numbers, dropping bullet content.
5. **Verification:** two-tier — cheap-always + deep on `--verify`. Failure semantics in §6.4.
6. **DOCX auxiliary content (lean default):**
   - **Default = lean.** Comments dropped (not in output). Tracked changes shown as accepted-final (insertions kept, deletions dropped). Images dropped, replaced with `<!-- image -->` placeholder matching Docling's existing convention. **Footnotes always preserved** as `[^N]` markdown footnotes with an auto-generated `## Footnotes` section at end — footnotes are body content (e.g., legal citations), not editorial overlay.
   - **`--full` flag** opts into the maximalist behavior: comments appended in a `## Reviewer Comments` appendix with anchored references (`[C1]` inline; `[C1] **Reviewer name** (date): "comment text"` in appendix). No effect on footnotes (already preserved) or images (use `--keep-images` for that).
   - **`--keep-images`** extracts embedded images to `<output>_files/` and references them with `![](output_files/img-N.png)`.
   - **`--show-revisions`** renders tracked changes as `[+ added +]` / `[- removed -]` (independent of `--full`).
7. **Architecture:** per-format modules + shared core (see §5).
8. **Project name:** `materials-md` (CLI binary: `convert`).

## 5. Architecture

### 5.1 Module layout

```
convert.py                       # CLI entry point: argparse + dispatch by extension
materials/
  __init__.py
  core/
    base.py                      # BaseConverter ABC; ConversionResult dataclass;
                                 #   ConversionOptions dataclass
    output.py                    # Filename rules, output directory creation, log paths
    verify.py                    # Cheap-check primitives (count parity, retention ratio,
                                 #   non-empty); deep-check primitives shared across formats
  formats/
    pdf.py                       # PDFConverter — wraps existing pdf_to_markdown.py logic
    docx.py                      # DOCXConverter — Docling + python-docx
    pptx.py                      # PPTXConverter — Docling with FURNITURE layer enabled
    html.py                      # HTMLConverter — Docling (+ optional bs4 pre-clean)
console.py                       # Rich UX (existing — minor refactor to share Panel/Table
                                 #   helpers with new format-specific reports)
pdf_to_markdown.py               # Deprecation shim → convert.py (removed in stage 5)
verify_cli.py                    # Standalone verifier CLI (folds verify_conversion.py
                                 #   and verify_page_markers.py into one entry point)
requirements.txt
README.md
CLAUDE.md
docs/superpowers/specs/          # this spec lives here
tests/
  fixtures/                      # see §7.2
  test_pdf.py
  test_docx.py
  test_pptx.py
  test_html.py
  test_verify.py
  test_cli.py                    # exercises convert.py dispatch + flags
```

**No `core/markers.py`** in this layout. Each format owns its own marker insertion because the strategies differ fundamentally (PDF: 3-strategy fuzzy fallback; PPTX: walk slides via `iterate_items` with FURNITURE layer; DOCX/HTML: heading walk). The only shared piece is `core/output.py::sanitize_heading_text`, used by DOCX and HTML for marker text. When two formats independently grow code that genuinely overlaps beyond that, that's the moment to extract — not before.

### 5.2 Marker uniqueness and heading escaping

Section markers for DOCX and HTML are **numbered** to ensure intra-document uniqueness:

```
<!-- Section 1: Discussion -->
... body ...
<!-- Section 2: Conclusion -->
... body ...
```

Numbering is sequential by document order, restarting at 1 per document. The heading text is included for human readability. Citation form: "Section 3 of memo" maps unambiguously to `<!-- Section 3: ... -->`.

**Heading escaping rule (mandatory).** Heading text within a marker is sanitized before insertion to avoid breaking HTML comment syntax and to keep markers compact:

1. Collapse any run of two-or-more `-` characters to a single `-` (prevents `-->` and `--` from terminating or destabilizing the comment).
2. Strip backticks (` ` ` `) — they confuse downstream renderers when they straddle a comment boundary.
3. Strip newlines and tabs; collapse internal whitespace to single spaces.
4. Truncate to 80 characters (excluding the `<!-- Section K: ` and ` -->` framing). Append `…` if truncated.
5. If the result is empty after sanitization, use `(untitled)`.

Example: a DOCX heading `Why we use --no-foo (and \`--strict\`)` becomes `<!-- Section 4: Why we use -no-foo (and -strict) -->`.

Implementation lives in `core/output.py` as `sanitize_heading_text(text: str) -> str`; both `formats/docx.py` and `formats/html.py` call it.

For PDF and PPTX, page/slide numbers are already unique within a document so no escaping is needed there.

### 5.3 Format-specific notes

**PDF (`formats/pdf.py`):** lifts and shifts the existing 3-stage marker insertion (Docling tokens → provenance → PyMuPDF + RapidFuzz fallback) verbatim. Stage 1 of the implementation plan covers the move; behavior must be identical to today's `pdf_to_markdown.py`, verified by running the existing suite of casebook conversions and diffing outputs.

**DOCX (`formats/docx.py`):** Docling produces the prose backbone. python-docx is opened separately on the same file to extract:
- Comments via `document.part.related_parts` (look for the `comments` part) → list of `(comment_id, author, date, text, anchor_paragraph_index)`.
- Footnotes via `document.part.related_parts` (look for the `footnotes` part) → emitted as `[^N]` markdown footnotes with a `## Footnotes` section appended.
- Revision elements (`w:ins`, `w:del`) walked via lxml on `document.element.body`.

**Comment-extraction scope (stage 3):** only top-level Word comments from `comments.xml` are captured. Modern Word also writes `commentsExtended.xml` (threaded replies and resolved/unresolved state), `commentsExtensible.xml`, and `commentsIds.xml`. python-docx's high-level API doesn't expose these uniformly, so `--full` output emits a top-level comment but its replies are silently dropped. This is a **known limitation** — documented in `convert.py --help` and the README — to be addressed in a follow-up after stage 5. If a memo has heavy reviewer threading, fall back to the original `.docx` for review.

Default behavior strips comments and accepts revisions but **always preserves footnotes**. `--full` adds the comments appendix; `--show-revisions` renders revisions inline. Section markers inserted by walking Docling's element tree for `section_header` items and assigning sequential numbers (per §5.2). **DOCX with no headings:** single fallback marker `<!-- Section 1: (untitled) -->` at top of document, mirroring the HTML rule in §5.4.

**PPTX (`formats/pptx.py`):** Single Docling pass with `included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}`. Slide markers inserted by walking the document tree per slide group; speaker-note text identified by `content_layer == FURNITURE` within a slide group and re-emitted with `<!-- Speaker notes -->` prefix. `--notes-only` flag walks slides and emits only `<!-- Slide N -->\n<notes text>` for each slide that has notes.

**HTML (`formats/html.py`):** Docling does the conversion. With `--strip-html-noise`, beautifulsoup4 strips `<script>`, `<style>`, `<nav>`, `<footer>`, `<aside>`, and elements with `class` matching `/sidebar|advert|cookie|consent/i` before handing the cleaned HTML to Docling. Section markers inserted on H1/H2 headings only; H3+ are not numbered (they live inside their parent section). Documented edge cases below.

### 5.4 HTML edge cases

Documented as known limitations in `convert.py --help` and README:

- **JS-rendered pages:** static HTML only — pages that hydrate content client-side will produce minimal output regardless of `--strip-html-noise`. Both Docling and bs4 see only the bytes on disk; neither runs a browser. Recommendation: render to static HTML upstream (e.g., browser "Save Page As → Web Page, Complete") before converting.
- **Multiple `<h1>` tags:** each gets its own `<!-- Section K: ... -->` marker; numbering continues across them.
- **`<h3>` with no `<h2>` parent:** ignored for marker purposes (sectioning happens at H1/H2 only). The H3 still appears in the markdown as `### ...`.
- **No headings at all:** single section marker `<!-- Section 1: (untitled) -->` at top of document.

## 6. Verification

### 6.1 Cheap checks (always on)

Run synchronously after conversion, fast (<100ms typical), block writes only on output-empty.

| Check | All formats | PDF only | PPTX only | DOCX only | HTML only |
|---|---|---|---|---|---|
| Output non-empty (>0 chars) | ✓ | | | | |
| Word retention ratio in band | ✓ ≥75% | | | ≥90% | ≥60% |
| Position markers present where expected | ✓ | Page count match within ±5% | Slide count exact | Section count > 0 if doc has H1/H2 | Section count > 0 if doc has H1/H2 |
| | | | Notes-bearing slide count match | Comment count match (when `--full`) | |

### 6.2 Deep checks (`--verify`)

| Check | Format |
|---|---|
| Per-marker fuzzy validation against source page | PDF |
| Per-slide content+notes round-trip parity | PPTX |
| Per-comment anchor resolves to correct paragraph | DOCX (`--full`) |
| Footnote cross-references resolve | DOCX (`--full`) |
| Heading hierarchy preserved (no orphan H3) | HTML, DOCX |

### 6.3 Sidecar report

When `--verify` runs, a `<output>.verify.json` sidecar is written next to the markdown with structured results: `{format, checks_run, results: [{name, status, detail}], duration_ms}`. This enables `find converted/ -name '*.verify.json' | xargs jq '.results[] | select(.status=="FAIL")'` audits across a corpus.

### 6.4 Failure semantics

| Layer | Single-file mode | Batch mode |
|---|---|---|
| Cheap-check FAIL on output-empty | Exit code 2; do not write markdown | Skip file; log; aggregate into summary; continue (controllable via `--continue-on-error` / `--no-continue-on-error`; default = continue) |
| Cheap-check FAIL on retention/count drift | Exit code 1; **write markdown** with a `<!-- VERIFY: WARN -->` header line | Same; aggregate into batch summary |
| Deep-check FAIL (`--verify`) | Exit code 1; write markdown; sidecar `.verify.json` records failure | Same |
| Pass | Exit 0 | Exit 0 if all files pass; 1 if any failed |

`--continue-on-error` only applies in batch mode (single-file always exits non-zero on failure). `--strict` flag escalates retention/count drift to a hard fail (no markdown written), useful for automated pipelines that prefer strict semantics.

## 7. Test infrastructure

### 7.1 Decision

Tests are added in stage 1 alongside the PDF refactor. No CI yet (matches current "no tests, no linter config" baseline). Tests run via `./venv/bin/python -m pytest tests/`.

### 7.2 Fixtures

- Minimal fixtures committed in `tests/fixtures/` — under 100KB each. Every fixture is **scripted**, not hand-clicked, so regeneration is reproducible.
- One fixture per format, each built by a committed script in `tests/fixtures/build/`:
  - `tests/fixtures/sample.pdf` (3 pages, Roman-numeral-prefixed page label, one table) — `build/build_pdf.py` uses ReportLab.
  - `tests/fixtures/sample_with_comments.docx` (one paragraph, one footnote, one Word comment, ~30KB) — `build/build_docx.py` uses python-docx for the prose and lxml to inject `<w:commentRangeStart>`/`<w:commentRangeEnd>` and a `comments.xml` part directly into the .docx archive (python-docx does not insert comments natively).
  - `tests/fixtures/sample_with_notes.pptx` (3 slides, 2 with speaker notes) — `build/build_pptx.py` uses python-pptx.
  - `tests/fixtures/sample_article.html` (article with H1/H2/H3, one `<nav>` to test stripping, one heading containing `--` to exercise §5.2 escaping) — `build/build_html.py` writes static HTML.
- A `tests/fixtures/README.md` documents what each fixture exercises and how to regenerate (`python tests/fixtures/build/build_<format>.py`). Both the fixture and its builder are committed.

### 7.3 Test contract

A passing test means: (a) conversion runs without exception; (b) the cheap verifier reports PASS; (c) marker count matches the fixture's expected count; (d) for DOCX/PPTX, key auxiliary content (comments, notes) is present iff its flag is set.

## 8. Incremental delivery

Five stages. Each stage is independently shippable and useful.

### Stage 1 — Refactor PDF to new module structure

**Goal:** zero behavior change; existing PDF conversions produce byte-identical output.

- Create `convert.py`, `materials/core/{base,output,verify}.py`, `materials/formats/pdf.py`.
- Move PDF logic from `pdf_to_markdown.py` into `formats/pdf.py` and `core/`.
- `pdf_to_markdown.py` becomes a 5-line deprecation shim: prints `[DEPRECATED] use convert.py — pdf_to_markdown.py will be removed in stage 5` to stderr, then forwards `sys.argv` to `convert.main()` (the shim sets `sys.argv[0] = "convert.py"` so argparse error messages reference the new tool).
- Add `tests/test_pdf.py` exercising `tests/fixtures/sample.pdf` (the scripted fixture from §7.2); cheap verifier ported.
- Migration test (`tests/test_pdf.py::test_migration_byte_identical`): run `tests/fixtures/sample.pdf` through both `git show 3ec3edb:pdf_to_markdown.py` (the pre-refactor entry point, captured as a copy in `tests/fixtures/legacy_pdf_to_markdown.py`) and the new `convert.py`, assert byte-identical markdown output.
- Update `CLAUDE.md` to describe the new module layout (`convert.py`, `materials/core/`, `materials/formats/pdf.py`) and the deprecation shim. Remove stale claims about `pdf_to_markdown.py` being the main converter.

**Ships:** PDF-only converter on the new architecture. Same UX, same flags, same output.

### Stage 2 — Add HTML

**Goal:** lowest-friction new format.

- Create `formats/html.py`. Pure Docling pipeline, optional bs4 pre-clean.
- Section markers (numbered) per §5.2; `core/output.py::sanitize_heading_text` lands here.
- HTML cheap verifier.
- HTML test + fixture (the §7.2 fixture exercises §5.2 escaping via a heading containing `--`).
- README updated with HTML usage; `CLAUDE.md` updated to add HTML to the supported-formats section and note the bs4 optional dependency.

**Ships:** PDF + HTML.

### Stage 3 — Add DOCX

**Goal:** memo/document workflow. Lean default, opt-in maximalist.

- Create `formats/docx.py`.
- python-docx integration for comments, footnotes, revisions. **Top-level comments only** in this stage; threaded replies are a documented limitation (§5.3).
- `--full`, `--show-revisions`, `--keep-images` flags.
- Section markers (numbered) using `core/output.py::sanitize_heading_text`; no-headings fallback per §5.3.
- DOCX cheap + deep verifiers.
- DOCX test + fixture (scripted via `tests/fixtures/build/build_docx.py` — one paragraph, one comment, one footnote).
- `requirements.txt` adds `python-docx>=1.1.0`.
- `CLAUDE.md` updated to document the DOCX format and its flags.

**Ships:** PDF + HTML + DOCX.

### Stage 4 — Add PPTX

**Goal:** lecture-deck workflow with speaker notes.

- Create `formats/pptx.py`.
- Docling with `included_content_layers={BODY, FURNITURE}`.
- Slide markers + `<!-- Speaker notes -->` markers.
- `--notes-only` flag.
- PPTX cheap + deep verifiers.
- PPTX test + fixture (scripted via `tests/fixtures/build/build_pptx.py`).
- **No new dependencies** (Docling handles it; verified empirically).
- `CLAUDE.md` updated to document PPTX support and speaker-notes behavior.

**Ships:** all four formats. Project goal achieved.

### Stage 5 — Polish, parallelism, removal

- Remove `pdf_to_markdown.py` deprecation shim.
- Add `materials/core/parallel.py` with `ProcessPoolExecutor`-based batch parallelism (per-worker model warmup; opt-in via `--workers N`, default 1).
- Performance benchmarking — Polk runs `tests/bench/run_bench.py` against a personal casebook corpus (≥20 files) and pastes results into the stage-5 PR description. If benchmark shows no win below 20 files, document that and keep default `--workers 1`. (The bench script is committed; the corpus is not — it's user content.)
- `verify_cli.py` standalone verifier consolidating `verify_conversion.py` + `verify_page_markers.py`; old scripts become deprecation shims (removed at next cycle).
- README final pass; `CLAUDE.md` final pass to remove all stage-1-through-4 staleness and reflect the shipped state.

**Ships:** the whole Python suite, polished.

### Stage 6 — Claude Code skill wrapper

**Goal:** make the converter accessible from natural-language conversation in Claude Code via a thin skill that shells out to the Python CLI.

- Create `~/.claude/skills/materials-md/SKILL.md` (a markdown instruction sheet, not Python code) — describes when to invoke the converter, how to map natural-language requests to flags, and where to put output.
- Skill is a **thin dispatcher**, not a reimplementation. It does nothing more than recognize the request, pick flags, shell out to `~/Penn Law Dropbox/Polk Wagner/code/pdf-converter/convert.py`, and return the output path.
- Document composition with existing skills: convert → eddie, convert → polk-document, convert → factual-pipeline-orchestrator.
- No new dependencies; the Python tool does all the work.

**Ships:** natural-language entry point to the converter, composable with the rest of the skill ecosystem.

### Stage 7 (deferred) — Subagent batch orchestration

Not in scope for this project until evidence justifies it. Possible future work: a skill that orchestrates "convert this whole folder, then route each output through eddie / a verifier / a follow-up skill," using parallel subagents per file. Decision deferred until stages 1-6 ship and the manual workflow is observed to be painful enough to need orchestration.

## 8.5 Architecture decision: Python core + thin skill wrapper

The deterministic conversion engine is a Python project (stages 1-5). A Claude Code skill wraps it as a thin dispatcher (stage 6). Reasoning:

- **Determinism wins for legal materials.** The same PPTX must always produce the same markdown — a verifier system only catches silent failures if the underlying conversion is reproducible. A Claude-mediated conversion would re-summarize content nondeterministically, costs hundreds of dollars per casebook in tokens, runs minutes-to-hours instead of seconds, and silently drops content the verifier can't predict.
- **The existing PDF code is battle-tested.** The 3-strategy marker insertion (Docling tokens → provenance → PyMuPDF + RapidFuzz) is real engineering. Rebuilding it as a Claude skill would be a regression.
- **Composability.** A Python CLI runs in cron, in Make, in CI, in shell pipelines. A skill only runs inside Claude Code. The Python project is the durable artifact.
- **The skill earns its keep at the natural-language boundary.** It maps "give me just the lecture text" → `--notes-only`, "keep all reviewer comments" → `--full`, and routes output to follow-up skills (eddie, polk-document, factual-pipeline-orchestrator). It does not replicate conversion logic.

The trap to avoid is a skill that *re-implements* conversion in Claude. The skill must be a dispatcher. Conversion logic lives in Python; only flag-mapping and routing live in the skill.

## 9. Tradeoffs (made explicit)

- **Per-format files vs shared core:** chosen split optimizes for *isolation and ease of adding a 5th format later*. Cost: some boilerplate (each format file imports the same set of types from `core/base`). Acceptable.
- **Lean DOCX default vs maximalist:** prioritizes the AI-ingestion use case (the stated primary goal). Cost: workflows that *do* want comments/footnotes need `--full`. Acceptable — those are batch-runnable with a saved alias.
- **Numbered section markers:** verbosity (longer markers) traded for citation uniqueness. Worth it.
- **No `core/markers.py`:** code duplication acceptable risk vs premature abstraction. The duplication is small (each format's marker insertion is local logic, not a library); if real overlap appears in stages 3-4, extraction is cheap.
- **Stages 5's parallelism gated on benchmark:** rather than committing to a "wins past 20 files" claim, we measure and document the actual threshold. If parallelism doesn't pay off for typical batches, ship without it and document why.

## 10. Out-of-scope reaffirmed

- ePub, RTF, `.doc` (legacy Word), image OCR as primary input — not in this spec; reconsider after stage 5 ships.
- A GUI — no.
- Cloud-hosted conversion service — no.
- Format conversion in the other direction (markdown → PDF/DOCX/PPTX) — no; that's `md-to-pdf` and `polk-document` skills.

## 11. Deprecation and documentation lifecycle

- `pdf_to_markdown.py` deprecation shim ships in **stage 1**.
- Removed in **stage 5**.
- `verify_conversion.py` and `verify_page_markers.py` deprecation shims ship in **stage 5**, removed at the next material change after stage 5.
- **CLAUDE.md update rule:** every stage that changes the architecture also updates `CLAUDE.md` to reflect the architecture as of that stage's merge. No stage merges with a stale `CLAUDE.md`. This rule applies retroactively to any stage description above that didn't already include a `CLAUDE.md` line.
