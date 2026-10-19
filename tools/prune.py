#!/usr/bin/env python3
"""Prune the experiment switches of a CUDA source at their frozen default values.

This tool works like `unifdef`. It reads a source file and tracks object-like
`#define` values. It evaluates `#if`, `#ifdef`, `#ifndef`, `#elif`, `#else` and
`#endif`. It keeps only the live branch of every conditional that it can decide.
A conditional with an unknown identifier (for example `NB`, which comes from the
compiler command line) stays in the output. The tool then removes `#define`s
that nothing uses any more.

Usage: prune.py INPUT.cu OUTPUT.cu [--report] [--public]\n\n--public also applies HAND_FIXES, renames the tuning constants and adds the header.
"""
from __future__ import annotations

import re
import sys

DIRECTIVE = re.compile(r"^\s*#\s*(\w+)\s*(.*)$", re.S)
IDENT = re.compile(r"[A-Za-z_]\w*")


class Undecidable(Exception):
    pass


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//.*", "", text).strip()


TOKEN = re.compile(r"\s*(0[xX][0-9a-fA-F]+|\d+|[A-Za-z_]\w*|<<|>>|<=|>=|==|!=|&&|\|\||[-+*/%()&|^!~<>?:,])")


def tokenize(expr: str):
    pos, out = 0, []
    expr = expr.strip()
    while pos < len(expr):
        m = TOKEN.match(expr, pos)
        if not m:
            raise Undecidable(f"bad token in {expr!r}")
        tok = m.group(1)
        pos = m.end()
        # drop integer suffixes such as 1u or 1ull
        if re.fullmatch(r"(0[xX][0-9a-fA-F]+|\d+)", tok):
            sm = re.match(r"[uUlL]+", expr[pos:])
            if sm:
                pos += sm.end()
        out.append(tok)
    return out


class Evaluator:
    def __init__(self, macros: dict, funcs: set, tokens: list, depth: int = 0):
        self.macros, self.funcs, self.t, self.i, self.depth = macros, funcs, tokens, 0, depth

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self, want=None):
        tok = self.peek()
        if tok is None or (want is not None and tok != want):
            raise Undecidable(f"expected {want}, got {tok}")
        self.i += 1
        return tok

    def parse(self):
        v = self.ternary()
        if self.peek() is not None:
            raise Undecidable(f"trailing token {self.peek()}")
        return v

    def ternary(self):
        c = self.binary(0)
        if self.peek() == "?":
            self.take("?")
            a = self.ternary()
            self.take(":")
            b = self.ternary()
            return a if c else b
        return c

    LEVELS = [["||"], ["&&"], ["|"], ["^"], ["&"], ["==", "!="], ["<", ">", "<=", ">="], ["<<", ">>"], ["+", "-"], ["*", "/", "%"]]

    def binary(self, lvl):
        if lvl == len(self.LEVELS):
            return self.unary()
        v = self.binary(lvl + 1)
        while self.peek() in self.LEVELS[lvl]:
            op = self.take()
            # no short-circuit skipping of undecidable operands: keep it simple and strict
            w = self.binary(lvl + 1)
            v = {"||": lambda a, b: int(bool(a) or bool(b)), "&&": lambda a, b: int(bool(a) and bool(b)),
                 "|": lambda a, b: a | b, "^": lambda a, b: a ^ b, "&": lambda a, b: a & b,
                 "==": lambda a, b: int(a == b), "!=": lambda a, b: int(a != b),
                 "<": lambda a, b: int(a < b), ">": lambda a, b: int(a > b),
                 "<=": lambda a, b: int(a <= b), ">=": lambda a, b: int(a >= b),
                 "<<": lambda a, b: a << b, ">>": lambda a, b: a >> b,
                 "+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
                 "/": lambda a, b: int(a / b), "%": lambda a, b: a - int(a / b) * b}[op](v, w)
        return v

    def unary(self):
        tok = self.peek()
        if tok == "!":
            self.take()
            return int(not self.unary())
        if tok == "-":
            self.take()
            return -self.unary()
        if tok == "~":
            self.take()
            return ~self.unary()
        if tok == "(":
            self.take()
            v = self.ternary()
            self.take(")")
            return v
        return self.atom()

    def atom(self):
        tok = self.take()
        if re.fullmatch(r"0[xX][0-9a-fA-F]+", tok):
            return int(tok, 16)
        if tok.isdigit():
            return int(tok)
        if tok == "defined":
            paren = self.peek() == "("
            if paren:
                self.take("(")
            name = self.take()
            if paren:
                self.take(")")
            return self.defined(name)
        if tok in self.funcs:
            raise Undecidable(f"function-like macro {tok}")
        if tok in self.macros:
            if self.depth > 50:
                raise Undecidable("macro recursion")
            val = self.macros[tok]
            if val is None:
                raise Undecidable(f"empty macro {tok}")
            return Evaluator(self.macros, self.funcs, tokenize(val), self.depth + 1).parse()
        raise Undecidable(f"unknown identifier {tok}")

    def defined(self, name):
        if name in self.macros or name in self.funcs:
            return 1
        if name in EXTERNAL:
            raise Undecidable(f"defined({name}) is set outside")
        return 0


EXTERNAL = {"NB", "__CUDA_ARCH__", "__CUDACC__"}


def logical_lines(text: str):
    """Yield (list_of_raw_lines, joined_text) with backslash continuations merged."""
    raw = text.split("\n")
    i = 0
    while i < len(raw):
        start = i
        group = [raw[i]]
        while group[-1].rstrip().endswith("\\") and i + 1 < len(raw):
            i += 1
            group.append(raw[i])
        i += 1
        yield start, group


def prune(text: str, report: list | None = None, public: bool = False) -> str:
    macros: dict = {}
    funcs: set = set()
    out: list = []          # items: dict(start, lines, kind, name, config); kind in {"text", "define", "directive"}
    expand: dict = {}       # name -> (params or None, body): macros defined by a switch branch, not plain numbers
    tag: dict = {}          # original line index -> "define" | "switch" for a removed directive
    stack: list = []        # frames: dict(decided, taken, emit, name)
    log = report if report is not None else []

    def emitting():
        return all(f["emit"] for f in stack) if stack else True

    def evaluate(cond: str):
        try:
            return int(bool(Evaluator(macros, funcs, tokenize(strip_comments(cond))).parse()))
        except Undecidable as e:
            log.append(f"undecidable #if kept: {cond.strip()!r} ({e})")
            return None

    def add(start, group, kind, name=None, config=False):
        out.append({"start": start, "lines": group, "kind": kind, "name": name, "config": config})

    for start, group in logical_lines(text):
        joined = " ".join(g.rstrip("\\") for g in group)
        m = DIRECTIVE.match(joined)
        if not m:
            if emitting():
                add(start, group, "text")
            continue
        d, rest = m.group(1), m.group(2)
        parent = emitting()
        if d in ("if", "ifdef", "ifndef"):
            if not parent:
                stack.append({"decided": True, "taken": True, "emit": False, "name": None})
                continue
            name = None
            if d == "if":
                val = evaluate(rest)
            else:
                name = strip_comments(rest).split()[0]
                if name in EXTERNAL:
                    val = None
                    log.append(f"undecidable #{d} kept: {name}")
                else:
                    val = int(name in macros or name in funcs)
                    if d == "ifndef":
                        val = 1 - val
            if val is None:
                stack.append({"decided": False, "taken": False, "emit": True, "name": None})
                add(start, group, "directive")
            else:
                stack.append({"decided": True, "taken": bool(val), "emit": bool(val),
                              "name": name if d == "ifndef" else None})
                tag[start] = "switch"
        elif d in ("elif", "else"):
            f = stack[-1]
            outer = all(x["emit"] for x in stack[:-1]) if len(stack) > 1 else True
            if not f["decided"]:
                if outer:
                    add(start, group, "directive")
                continue
            if not outer:
                continue
            if f["taken"]:
                f["emit"] = False
            else:
                val = 1 if d == "else" else evaluate(rest)
                if val is None:
                    raise SystemExit(f"elif after decided branch is undecidable: {rest!r}")
                f["emit"] = bool(val)
                f["taken"] = bool(val)
        elif d == "endif":
            f = stack.pop()
            if not f["decided"] and all(x["emit"] for x in stack):
                add(start, group, "directive")
        elif not parent:
            continue
        elif d == "define":
            body = strip_comments(joined.split("define", 1)[1].strip())
            mm = re.match(r"([A-Za-z_]\w*)(\(.*?\))?\s*(.*)$", body, re.S)
            name, params, val = mm.group(1), mm.group(2), mm.group(3).strip()
            if any(not f["decided"] for f in stack):
                log.append(f"define inside undecided block (not tracked): {name}")
            elif params is not None:
                funcs.add(name)
                macros.pop(name, None)
            else:
                macros[name] = val if val else None
                funcs.discard(name)
            config = any(f["name"] == name for f in stack)
            if stack and not any(not f["decided"] for f in stack):
                if params is not None:
                    expand[name] = ([a.strip() for a in params[1:-1].split(",") if a.strip()], val)
                else:
                    try:
                        Evaluator(macros, funcs, tokenize(val)).parse() if val else (_ for _ in ()).throw(Undecidable("empty"))
                        expand.pop(name, None)
                    except Undecidable:
                        expand[name] = (None, val)
            add(start, group, "define", name, config)
        elif d == "undef":
            name = strip_comments(rest).split()[0]
            macros.pop(name, None)
            funcs.discard(name)
            add(start, group, "directive")
        elif d == "error":
            raise SystemExit(f"live #error: {joined.strip()}")
        else:
            add(start, group, "directive")
    if stack:
        raise SystemExit("unbalanced conditionals")
    return finish(out, tag, log, expand, public)


# Edits that the macro evaluator cannot make. Each entry is (old text, new text, expected count).
# They remove the last uses of frozen switch values and the comments that name removed switches.
# Every fix must match exactly the expected number of times, or the tool stops.
HAND_FIXES = [
    ("rmsd_to_smem<!R10_QKE >(x, (ln1 + (l) * H), sbuf, red);  // R10_QKE: no QKV cp.async group in the down -> QKV window (and none left over), so no wait_all",
     "rmsd_to_smem<true>(x, ln1 + l * H, sbuf, red);  // waits for the cp.async group of the QKV window", 1),
    ("{ rmsd_to_smem<false >(x, (ln2 + (l) * H), sbuf, red); do {} while (0); }",
     "rmsd_to_smem<false>(x, ln2 + l * H, sbuf, red);", 1),
    ("        if (l + 1 < NL) ;\n", "", 1),
    ("template <bool WAIT = false >", "template <bool WAIT = false>", 1),
    ("(size_t)(l) *", "(size_t)l *", 5), ("(size_t)(ll) *", "(size_t)ll *", 3),
    ("((gt) / (1.f + expf(-(gt))))", "(gt / (1.f + expf(-gt)))", 1),
    ("template <int G = 0, int U, class F>  // G: 0 plain loads, 1 (R10_PF 16, R10_W0LF 1) ld_v4_nc predicated on warp != 0, 2 (R10_W0LF 2) ld_v4_nc unpredicated",
     "template <int U, class F>", 1),
    ("gemv_load_lf<R10_LFQ>(", "gemv_load_lf(", 1),
    ("{    // 8 B (as V1q's V-cache copy)", "{    // 8 B", 1),
    ("// own Wo slice, K and V-cache loads as in V1q (group B)", "// own Wo slice, K and V-cache loads (group B)", 1),
    ("        if (R10_GUE_AT == 0 || !fused) issue_gue(l);  // layer l's gate / up rows: first statement of the phase (non-fused CTAs also with GUE_AT 1)",
     "        if (!fused) issue_gue(l);  // layer l's gate / up rows: non-fused CTAs issue at the start of the phase", 1),
    ("            // prep roles: warp R10_PFS + 0: q head fh, + 1: k of kv-head fg, + 2: v copy (R10_PFS is 1 with R10_PF 32: warp 0 is idle)",
     "            // prep roles: warp 0: q head fh, warp 1: k of kv-head fg, warp 2: v copy", 1),
    ("(unsigned)(warp - R10_PFS) < 2u", "warp < 2", 2),
    ("warp == R10_PFS ?", "warp == 0 ?", 3),
    ("warp == R10_PFS + 1 &&", "warp == 1 &&", 1),
    ("warp == R10_PFS + 2)", "warp == 2)", 1),
    ("sGU fits the 86016 B of dynamic smem (_dyn_smem)", "sGU fits the 86016 B of dynamic smem", 1),
    ("sD fits 43008 B (_dyn_smem)", "sD fits 43008 B of dynamic smem", 1),
]

HEADER = """// Qwen3-0.6B decode step, batch 1, position 128, as one persistent cooperative kernel.
// Main ideas:
//  - One launch for the whole step. One 512-thread CTA per SM. Grid barriers separate the dependent phases.
//  - A grid barrier is a release-add on a counter plus a relaxed poll. The weight loads of the next phase are issued between arrive and wait.
//  - Activations stay in fp32. The GEMV keeps its activations in registers and sums the lane results with a transpose butterfly.
//  - Fused attention: one CTA per q-head slice computes the softmax over all 129 positions and a split-K O-projection that red.adds into x.
//  - The gate / up rows and the down rows of a layer are TMA bulk copies into dynamic shared memory, ahead of use.
//  - RMSNorm of the next GEMV input is deferred. The row sums are scaled after the dot, so one __syncthreads suffices.
//  - The LM head keeps two rows in flight per warp. The argmax is a per-CTA reduction plus a last-arriver reduction.
"""

RENAMES = [(r"\bR8_SPIN_", "SPIN_"), (r"\bR8_STAGE_BYTES\b", "STAGE_BYTES"), (r"\br8_stage\b", "stage"), (r"\bR10_LMH\b", "LM_ROWS")]


def split_comment(line: str):
    i = line.find("//")
    return (line, "") if i < 0 else (line[:i], line[i:])


def expand_text(code: str, expand: dict, depth: int = 0) -> str:
    """Expand the macros of `expand` in one line of code. Calls must fit on one line."""
    if depth > 20:
        raise SystemExit("macro expansion too deep")
    out, pos = [], 0
    for m in IDENT.finditer(code):
        if m.start() < pos or m.group(0) not in expand:
            continue
        params, body = expand[m.group(0)]
        end = m.end()
        if params is None:
            repl = body
        else:
            if code[end:].lstrip()[:1] != "(":
                raise SystemExit(f"function-like macro {m.group(0)} used without a call")
            i = code.index("(", end)
            depth_p, j, args, cur = 0, i, [], ""
            while True:
                ch = code[j]
                if ch == "(":
                    depth_p += 1
                    if depth_p > 1:
                        cur += ch
                elif ch == ")":
                    depth_p -= 1
                    if depth_p == 0:
                        break
                    cur += ch
                elif ch == "," and depth_p == 1:
                    args.append(cur.strip())
                    cur = ""
                else:
                    cur += ch
                j += 1
                if j >= len(code):
                    raise SystemExit(f"call of {m.group(0)} does not close on its line")
            if cur.strip() or args:
                args.append(cur.strip())
            if len(args) != len(params):
                raise SystemExit(f"{m.group(0)}: {len(args)} arguments, {len(params)} parameters")
            sub = dict(zip(params, args))
            repl = IDENT.sub(lambda t: sub.get(t.group(0), t.group(0)), body)
            end = j + 1
        repl = expand_text(repl, expand, depth + 1)
        out.append(code[pos:m.start()])
        out.append(repl)
        pos = end
        if not repl and code[pos:pos + 1] == " ":
            pos += 1
    out.append(code[pos:])
    return "".join(out)


def expand_line(line: str, expand: dict) -> str:
    code, comment = split_comment(line)
    new = expand_text(code, expand)
    if new == code:
        return line
    if not new.strip() or new.strip() == ";":
        return ";"
    return new.rstrip() + (("  " + comment) if comment else "")


def finish(out: list, tag: dict, log: list, expand: dict, public: bool = False) -> str:
    """Remove #defines that no remaining line uses, then the comments that belonged to removed switches."""
    for it in out:
        if it["kind"] == "text":
            it["lines"] = [expand_line(ln, expand) for ln in it["lines"]]
    if public:
        out = apply_fixes(out)
    out = [it for it in out if not (it["kind"] == "text" and len(it["lines"]) == 1 and it["lines"][0].strip() == ";")]
    while True:
        used = set()
        for it in out:
            body = "\n".join(it["lines"])
            if it["kind"] == "define":
                body = body.split("define", 1)[1]
                parts = body.split(None, 1)
                body = parts[1] if len(parts) > 1 else ""
            used.update(IDENT.findall(strip_comments(body)))
        dead = [k for k, it in enumerate(out) if it["kind"] == "define" and it["name"] not in used]
        if not dead:
            break
        for k in reversed(dead):
            log.append(f"removed unused define {out[k]['name']}")
            tag[out[k]["start"]] = "define"
            del out[k]
    # the trailing comment of a switch default describes the switch, not the code
    for it in out:
        if it["kind"] == "define" and it["config"]:
            it["lines"] = [re.sub(r"\s*//.*$", "", ln) for ln in it["lines"]]
    # a run of comment-only lines that touches a removed switch directive belongs to that switch
    def comment_only(it):
        return it["kind"] == "text" and len(it["lines"]) == 1 and it["lines"][0].strip().startswith("//")
    keep = [True] * len(out)
    k = 0
    while k < len(out):
        if not comment_only(out[k]):
            k += 1
            continue
        e = k
        while e + 1 < len(out) and comment_only(out[e + 1]) and out[e + 1]["start"] == out[e]["start"] + 1:
            e += 1
        after_config = k > 0 and out[k - 1]["kind"] == "define" and out[k - 1]["config"] \
            and out[k - 1]["lines"][-1] and out[k]["start"] == out[k - 1]["start"] + len(out[k - 1]["lines"])
        if after_config or tag.get(out[k]["start"] - 1) or tag.get(out[e]["start"] + 1):
            for q in range(k, e + 1):
                keep[q] = False
                log.append(f"dropped switch comment: {out[q]['lines'][0].strip()[:70]}")
        k = e + 1
    out = [it for it, kp in zip(out, keep) if kp]
    result = "\n".join("\n".join(it["lines"]) for it in out)
    if public:
        for pat, rep in RENAMES:
            result = re.sub(pat, rep, result)
        for old, new in (("#define SPIN_CYC 490", "// %clock cycles that thread 0 waits before it polls a grid barrier (one value per barrier)\n#define SPIN_CYC 490"),
                         ("#define LM_ROWS 2", "// LM-head rows in flight per warp\n#define LM_ROWS 2")):
            assert result.count(old) == 1, old
            result = result.replace(old, new)
        result = HEADER + result.lstrip("\n")
        result = re.sub(r"\n{3,}", "\n\n", result)
    return result


def apply_fixes(out: list) -> list:
    """Apply HAND_FIXES to the text items and the defines. A fix may span one line only."""
    counts = [0] * len(HAND_FIXES)
    for it in out:
        if it["kind"] != "text":
            continue
        new = []
        for ln in it["lines"]:
            for k, (old, rep, _) in enumerate(HAND_FIXES):
                if old.endswith("\n"):
                    continue
                n = ln.count(old)
                if n:
                    counts[k] += n
                    ln = ln.replace(old, rep)
            new.append(ln)
        it["lines"] = new
    # whole-line deletions (old text ends with a newline)
    for k, (old, rep, _) in enumerate(HAND_FIXES):
        if not old.endswith("\n"):
            continue
        for it in out:
            if it["kind"] == "text" and len(it["lines"]) == 1 and it["lines"][0] + "\n" == old:
                it["lines"] = [";"]
                counts[k] += 1
    for k, (old, rep, want) in enumerate(HAND_FIXES):
        if counts[k] != want:
            raise SystemExit(f"hand fix {k} matched {counts[k]} times, expected {want}: {old[:60]!r}")
    return out


def main(argv):
    src, dst = argv[1], argv[2]
    log: list = []
    result = prune(open(src).read(), log, public="--public" in argv)
    open(dst, "w").write(result)
    if "--report" in argv:
        print("\n".join(log))
    print(f"{src}: {len(open(src).read().splitlines())} lines -> {dst}: {len(result.splitlines())} lines")


if __name__ == "__main__":
    main(sys.argv)
