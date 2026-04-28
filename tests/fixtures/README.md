# Test fixtures

Each fixture is **scripted** — committed alongside its builder so it can be regenerated
deterministically. To rebuild a fixture, run its builder from the repo root:

    ./venv/bin/python tests/fixtures/build/build_pdf.py

| Fixture | Purpose | Builder |
|---|---|---|
| `sample.pdf` | 3-page PDF with Roman-numeral page labels (front matter) and a table | `build/build_pdf.py` |
| `sample_article.html` | Article with H1/H2/H3, a `<nav>` element, and headings containing `--` and backticks (exercises §5.2 escaping) | `build/build_html.py` |
| `sample_with_comments.docx` | Memo with H1/H2 headings, one footnote, one Word comment anchored to "contested claim", one tracked-changes ins/del pair | `build/build_docx.py` |
| `sample_with_notes.pptx` | 3-slide deck where slides 1 and 2 have speaker notes; slide 3 does not | `build/build_pptx.py` |

All four formats now have fixtures.
