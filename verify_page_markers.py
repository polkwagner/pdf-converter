#!/usr/bin/env python3
"""
Verify that page markers in converted markdown match actual PDF pages.
"""

import sys
import re
import fitz  # PyMuPDF
from pathlib import Path

def normalize_text(text):
    """Normalize text for comparison."""
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

def extract_page_markers(md_file):
    """Extract all page markers and their positions from markdown."""
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()

    markers = []
    for match in re.finditer(r'<!-- Page (\d+) -->', content):
        page_num = int(match.group(1))
        position = match.start()
        markers.append((page_num, position))

    return markers, content

def verify_page_markers(pdf_file, md_file, sample_size=50):
    """Verify page markers by sampling pages."""
    # Extract markers
    markers, md_content = extract_page_markers(md_file)

    print(f"Found {len(markers)} page markers in markdown")
    print(f"Checking if they're in sequential order...")

    # Check if markers are in sequential order
    marker_nums = [m[0] for m in markers]
    out_of_order = []
    for i in range(len(marker_nums) - 1):
        if marker_nums[i] >= marker_nums[i+1]:
            out_of_order.append((i, marker_nums[i], marker_nums[i+1]))

    if out_of_order:
        print(f"\n⚠ WARNING: {len(out_of_order)} page markers are OUT OF ORDER:")
        for idx, (i, curr, next_page) in enumerate(out_of_order[:10]):
            print(f"  Position {i}: Page {curr} followed by Page {next_page}")
        if len(out_of_order) > 10:
            print(f"  ... and {len(out_of_order) - 10} more")
    else:
        print("✓ All page markers are in sequential order")

    # Open PDF
    doc = fitz.open(pdf_file)
    total_pages = len(doc)

    print(f"\nPDF has {total_pages} pages")
    print(f"Markdown has markers for {len(markers)} pages")

    # Check for missing pages
    marker_set = set(marker_nums)
    missing_pages = [i for i in range(1, total_pages + 1) if i not in marker_set]

    if missing_pages:
        print(f"\n⚠ WARNING: {len(missing_pages)} pages are MISSING markers:")
        if len(missing_pages) <= 20:
            print(f"  Missing pages: {missing_pages}")
        else:
            print(f"  First 20 missing: {missing_pages[:20]}")
    else:
        print("✓ All pages have markers")

    # Spot check accuracy by sampling pages
    print(f"\n🔍 Spot-checking {sample_size} pages for accuracy...")

    # Sample pages evenly distributed
    import random
    random.seed(42)

    # Include first, last, and random middle pages
    sample_indices = [0, len(markers) - 1]  # First and last
    if len(markers) > sample_size:
        # Add evenly distributed samples
        step = len(markers) // (sample_size - 2)
        sample_indices.extend(range(step, len(markers) - 1, step))
    else:
        sample_indices = list(range(len(markers)))

    sample_indices = sorted(set(sample_indices))[:sample_size]

    correct = 0
    incorrect = 0
    blank_pages = 0

    for idx in sample_indices:
        page_num, pos = markers[idx]

        if page_num > total_pages:
            print(f"  ✗ Page {page_num}: Invalid page number (PDF only has {total_pages} pages)")
            incorrect += 1
            continue

        # Get PDF page text
        pdf_page = doc[page_num - 1]  # 0-indexed
        pdf_text = pdf_page.get_text().strip()

        if not pdf_text:
            blank_pages += 1
            continue  # Skip blank pages for verification

        # Get markdown text after this marker (next ~500 chars)
        next_marker_pos = markers[idx + 1][1] if idx + 1 < len(markers) else len(md_content)
        md_text_after = md_content[pos:min(pos + 2000, next_marker_pos)]

        # Skip page numbers and headers in PDF text (first few lines often just have page numbers)
        pdf_lines = pdf_text.split('\n')
        # Find first substantial line (more than just a page number or short header)
        substantial_start = 0
        for i, line in enumerate(pdf_lines):
            if len(line.strip()) > 20:  # First line with substantial content
                substantial_start = i
                break
        pdf_text_clean = '\n'.join(pdf_lines[substantial_start:])

        # Normalize both for comparison
        pdf_normalized = normalize_text(pdf_text_clean[:800])
        md_normalized = normalize_text(md_text_after[:1000])

        # Check if PDF text appears in markdown
        # Try multiple strategies for matching
        matched = False

        # Strategy 1: Check if first 100 chars of PDF appear anywhere in markdown
        if len(pdf_normalized) > 100:
            if pdf_normalized[:100] in md_normalized:
                matched = True

        # Strategy 2: Try first 50 chars
        if not matched and len(pdf_normalized) > 50:
            if pdf_normalized[:50] in md_normalized:
                matched = True

        # Strategy 3: Fuzzy match on beginning
        if not matched:
            from difflib import SequenceMatcher
            ratio = SequenceMatcher(None, pdf_normalized[:300], md_normalized[:300]).ratio()
            if ratio > 0.6:
                matched = True

        # Strategy 4: Try middle portion (skip potential intro/header)
        if not matched and len(pdf_normalized) > 200:
            pdf_middle = pdf_normalized[50:250]
            if len(pdf_middle) > 50 and pdf_middle in md_normalized:
                matched = True

        if matched:
            correct += 1
        else:
            incorrect += 1
            if incorrect <= 5:  # Show first 5 errors
                print(f"  ✗ Page {page_num}: Content mismatch")
                print(f"     PDF starts: {pdf_text_clean[:150]}")
                print(f"     MD after marker: {md_text_after[:150]}")

    doc.close()

    # Print summary
    print(f"\n{'='*60}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"Sampled pages:     {len(sample_indices)}")
    print(f"Correct:           {correct}")
    print(f"Incorrect:         {incorrect}")
    print(f"Blank pages:       {blank_pages}")
    print(f"Accuracy:          {correct/(correct+incorrect)*100:.1f}%")
    print(f"{'='*60}")

    return {
        'total_markers': len(markers),
        'out_of_order': len(out_of_order),
        'missing_pages': len(missing_pages),
        'sampled': len(sample_indices),
        'correct': correct,
        'incorrect': incorrect,
        'blank_pages': blank_pages
    }

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python verify_page_markers.py <pdf_file> <markdown_file>")
        sys.exit(1)

    pdf_file = sys.argv[1]
    md_file = sys.argv[2]

    verify_page_markers(pdf_file, md_file)
