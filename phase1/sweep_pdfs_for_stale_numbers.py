"""
sweep_pdfs_for_stale_numbers.py
Sweeps both action-plan PDFs for every stale number identified during the audit.
"""
import re
import fitz  # PyMuPDF

STALE_PATTERNS = [
    ("165,047 / 165047",   r"165[,.]?047"),
    ("23,265",             r"23[,.]?265"),
    ("8,908 nodes",        r"8[,.]?908"),
    ("14,357 edges",       r"14[,.]?357"),
    ("42,237 train",       r"42[,.]?237"),
    ("10,258 val",         r"10[,.]?258"),
    ("35.9% baseline",     r"35\.9\s*%?"),
    ("+16.1 pp",           r"16\.1\s*(pp|p\.p\.|percentage)?"),
    ("71.0% context",      r"71\.0\s*%?"),
    ("Unknown->Unknown",   r"Unknown\s*[→\-\>]+\s*Unknown"),
    ("212 / 200 type pairs", r"\b212\b|\b200\s+type\s+pairs"),
]

PDFS = [
    r"D:\FYP-B9\Agentic_TI_Action_Plan (1).pdf",
    r"D:\FYP-B9\Agentic_TI_Action_Plan_with_Feasibility_Study (1).pdf",
]

total_hits = 0
for pdf_path in PDFS:
    short_name = pdf_path.split("\\")[-1]
    print(f"\n{'='*70}")
    print(f"  {short_name}")
    print(f"{'='*70}")
    doc = fitz.open(pdf_path)
    pdf_hits = 0
    for page_num, page in enumerate(doc, 1):
        text = page.get_text()
        for label, pattern in STALE_PATTERNS:
            matches = [(m.start(), m.end()) for m in re.finditer(pattern, text, re.IGNORECASE)]
            if matches:
                # Extract surrounding context for each match
                for start, end in matches:
                    ctx_start = max(0, start - 60)
                    ctx_end   = min(len(text), end + 60)
                    ctx = text[ctx_start:ctx_end].replace("\n", " ").strip()
                    print(f"  p{page_num:3d}  [{label}]")
                    print(f"         ...{ctx}...")
                    pdf_hits += 1
    if pdf_hits == 0:
        print("  ✓  No stale numbers found.")
    else:
        print(f"\n  ⚠  {pdf_hits} stale hit(s) found — update before final export.")
    total_hits += pdf_hits
    doc.close()

print(f"\n{'='*70}")
print(f"  TOTAL STALE HITS ACROSS ALL PDFs: {total_hits}")
if total_hits == 0:
    print("  ✓  Both PDFs are clean.")
else:
    print("  ⚠  Update the above locations before submission.")
print(f"{'='*70}")
