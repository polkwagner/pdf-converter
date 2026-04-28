"""Build tests/fixtures/sample_article.html — a static HTML article that
exercises:
  - H1/H2/H3 heading hierarchy (H1/H2 should produce section markers; H3 should not)
  - A <nav> element (tests --strip-html-noise removal)
  - A heading containing "--" (tests core/output.py::sanitize_heading_text dash collapse)
  - A heading containing backticks (tests sanitize backtick stripping)

Run from repo root:
    ./venv/bin/python tests/fixtures/build/build_html.py
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "sample_article.html"

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sample Article</title>
</head>
<body>
<nav class="site-nav">
  <a href="/">Home</a> | <a href="/about">About</a>
</nav>

<article>
<h1>Article Title</h1>
<p>This is the introduction paragraph of the article. It establishes context
for what follows and contains enough words to make the word-retention
verification check meaningful.</p>

<h2>First Section</h2>
<p>Body of the first section. Lorem ipsum dolor sit amet, consectetur
adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna
aliqua. Enim ad minim veniam, quis nostrud exercitation ullamco laboris.</p>

<h3>A subsection that should not get its own marker</h3>
<p>This H3 lives inside the first section. The marker insertion logic should
ignore H3+ (sectioning happens at H1/H2 only).</p>

<h2>Why we use --strict mode</h2>
<p>This heading contains a double-dash to exercise the sanitize_heading_text
escaping rule from spec §5.2. The marker should read "Section 3: Why we use
-strict mode" (single dash).</p>

<h2>Code example with `backticks` in title</h2>
<p>This heading contains backticks to exercise the backtick-stripping rule.</p>

<footer>
  <p>Site footer with copyright junk that should be stripped.</p>
</footer>
</article>
</body>
</html>
"""


def build() -> Path:
    OUT.write_text(HTML, encoding="utf-8")
    return OUT


if __name__ == "__main__":
    p = build()
    size_kb = p.stat().st_size / 1024
    print(f"Wrote {p} ({size_kb:.1f} KB)")
