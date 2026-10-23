"""Turn HybridAgent inspector HTML reports into plain text for a job log.

Keeps the text spans, tool calls, observations and each latent span's
pseudo-text. Of the per-step nearest-token tables it keeps only the steps whose
nearest-token cosine is below LOW (default 0.5), i.e. the places where the
readout was guessing.

usage: python dump_trace.py r1.html r2.html ...   (env LOW=0.5, MAXCHARS=12000)
"""
import html
import os
import re
import sys

LOW = float(os.environ.get("LOW", "0.5"))
MAXCHARS = int(os.environ.get("MAXCHARS", "12000"))
SECTION_STARTS = ("text mode", "tool call", "observation", "answer")


def to_lines(raw):
    t = re.sub(r"(?s)<(script|style).*?</\1>", "", raw)
    t = re.sub(r"(?i)<br\s*/?>|</(div|p|li|tr|h\d|pre|section|span)>", "\n", t)
    t = html.unescape(re.sub(r"<[^>]+>", "", t))
    t = re.sub(r"[ \t]+", " ", t)
    return [l for l in t.split("\n") if l.strip()]


def condense(lines):
    out, in_table, prev = [], False, ""
    for l in lines:
        if l.startswith("per-step nearest tokens"):
            in_table = True
            out.append(f"  [steps with nearest-token cosine < {LOW}:]")
            continue
        if l.startswith(SECTION_STARTS) or l.startswith("latent span") or l.startswith("</think>"):
            in_table = False
        if in_table:
            # cosine line looks like "0.325 bat (0.32), bag (0.30), ..."
            m = re.match(r"^(0\.\d{3})", l)
            if m and float(m.group(1)) < LOW:
                out.append(f"   {prev} -> {l}")
            prev = l
            continue
        out.append(l)
    return out


def key(path):
    nums = re.findall(r"\d+", os.path.basename(path))
    return int(nums[-1]) if nums else 0


for f in sorted(sys.argv[1:], key=key):
    with open(f, encoding="utf-8") as fh:
        text = "\n".join(condense(to_lines(fh.read())))
    print("##### TRACE", f)
    print(text[:MAXCHARS])
