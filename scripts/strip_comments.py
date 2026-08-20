#!/usr/bin/env python3
"""Strip comments from source files in the repo.

This script removes COMMENT tokens from Python files using the stdlib
`tokenize` module (preserving docstrings and string literals) and removes
shell comments from `.sh` scripts (preserving shebang lines). It skips
vendor, virtualenv, assets, results, and other non-source folders.

Backups are written as `*.bak` next to the original file.
"""
import io
import os
import sys
import tokenize

ROOTS = ["src", "scripts", "tests"]
EXCLUDE_DIRS = {" .venv", ".venv", "assets", "results_sim", "kalari_gallery", "venv", "build", "dist"}


def should_skip_dir(d):
    return any(part in EXCLUDE_DIRS for part in d.split(os.sep))


def strip_py(path):
    with open(path, 'rb') as f:
        src = f.read()
    try:
        tokens = tokenize.tokenize(io.BytesIO(src).readline)
    except Exception:
        # If tokenize fails, skip file
        return False

    out = []
    prev_end = (1, 0)
    for tok in tokens:
        ttype = tok.type
        tstring = tok.string
        if ttype == tokenize.ENCODING:
            continue
        if ttype == tokenize.COMMENT:
            # skip comments
            continue
        out.append((ttype, tstring))

    # Reconstruct source from tokens
    try:
        new_src = tokenize.untokenize(out)
    except Exception:
        return False

    # Write backup and replace file
    bak = path + '.bak'
    if not os.path.exists(bak):
        os.rename(path, bak)
        with open(path, 'wb') as f:
            f.write(new_src)
        return True
    else:
        # backup exists, just overwrite
        with open(path, 'wb') as f:
            f.write(new_src)
        return True


def strip_sh(path):
    changed = False
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    for i, line in enumerate(lines):
        if i == 0 and line.startswith('#!'):
            new_lines.append(line)
            continue
        s = line
        out = []
        in_sq = in_dq = False
        for ch in s:
            if ch == "'" and not in_dq:
                in_sq = not in_sq
                out.append(ch)
                continue
            if ch == '"' and not in_sq:
                in_dq = not in_dq
                out.append(ch)
                continue
            if ch == '#' and not in_sq and not in_dq:
                # start of comment outside quotes
                break
            out.append(ch)
        new_line = ''.join(out).rstrip() + ('\n' if s.endswith('\n') else '')
        if new_line != line:
            changed = True
        new_lines.append(new_line)

    if changed:
        bak = path + '.bak'
        if not os.path.exists(bak):
            os.rename(path, bak)
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
    return changed


def main():
    repo_root = os.getcwd()
    modified = []
    for root in ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # prune excluded dirs
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
            if should_skip_dir(dirpath):
                continue
            for fn in filenames:
                path = os.path.join(dirpath, fn)
                if path.endswith('.py'):
                    ok = strip_py(path)
                    if ok:
                        modified.append(path)
                elif path.endswith('.sh') or fn.endswith('.sh'):
                    ok = strip_sh(path)
                    if ok:
                        modified.append(path)

    if modified:
        print('Modified files:')
        for m in modified:
            print(m)
    else:
        print('No files modified')


if __name__ == '__main__':
    main()
