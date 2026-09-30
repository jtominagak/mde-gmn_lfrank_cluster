#!/usr/bin/env python3
"""Move top-level definitions out of a pipeline entry script into a mde_gmn module.

This is the tool the Phase 2 refactor uses (docs/phase2-refactor-handoff.md).
It exists so each extraction is mechanical and reviewable instead of retyped:
bodies are moved VERBATIM, so `git show` for the commit is a pure move plus an
import line, and any behaviour change would be a bug in this script rather than
a transcription slip.

What one run does:
  1. reads the named definitions out of the entry script, in file order;
  2. writes (or appends to) scripts/src/mde_gmn/<module>.py, with the header
     text you pass in --header-file;
  3. works out which of the entry script's own imports the moved code needs,
     and puts them at the top of the new module;
  4. deletes those definitions from the entry script and adds
     `from mde_gmn.<module> import ...` for the names the rest of the script
     still uses;
  5. refuses to proceed if the moved code still depends on definitions that
     live in the entry script and are not being moved with it -- that
     dependency has to move first, or the module boundary is wrong.

Always dry-run first:
    python3 scripts/dev/split_module.py --module mde \\
        --entry scripts/src/tsls-mde-gmn.py \\
        --names run_mde_for_target,run_multi_mde_for_targets \\
        --header-file /tmp/mde_header.txt --dry-run

Then drop --dry-run, run the suite (see the handoff doc), and commit.
"""
import argparse
import ast
import sys
from pathlib import Path

PACKAGE = Path('scripts/src/mde_gmn')


def top_level(tree):
    """{name: node} for every top-level def/class/CONSTANT assignment."""
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node
    return out


def import_table(tree):
    """{bound name: the import statement that binds it} for the whole file."""
    table = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                table[alias.asname or alias.name.split('.')[0]] = ast.unparse(node)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                stmt = f'from {node.module} import {alias.name}'
                if alias.asname:
                    stmt += f' as {alias.asname}'
                table[alias.asname or alias.name] = stmt
    return table


def loaded_names(nodes):
    """Every name the given AST nodes read (attribute roots included)."""
    used = set()
    for node in nodes:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                used.add(sub.id)
            elif isinstance(sub, ast.Attribute):
                root = sub.value
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name):
                    used.add(root.id)
    return used


def wrap_import(module, names, width=95):
    line = f'from mde_gmn.{module} import ' + ', '.join(names)
    if len(line) <= width:
        return line
    out, cur = f'from mde_gmn.{module} import (\n', '    '
    for name in names:
        piece = name + ', '
        if len(cur) + len(piece) > width - 3:
            out += cur.rstrip() + '\n'
            cur = '    '
        cur += piece
    return out + cur.rstrip().rstrip(',') + ')'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--module', required=True, help='target module name inside mde_gmn/')
    ap.add_argument('--entry', required=True, help='entry script to move code OUT of')
    ap.add_argument('--names', required=True, help='comma-separated top-level names to move')
    ap.add_argument('--header-file', help='file holding the new module\'s docstring header')
    ap.add_argument('--extra-import', action='append', default=[],
                    help='import line the moved code needs that the entry script lacks '
                         '(repeatable), e.g. "from mde_gmn import columns"')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    names = [n.strip() for n in args.names.split(',') if n.strip()]
    entry_path = Path(args.entry)
    src = entry_path.read_text()
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    defs, imports = top_level(tree), import_table(tree)

    missing = [n for n in names if n not in defs]
    if missing:
        sys.exit(f'not top-level definitions in {entry_path}: {missing}')

    moving = [defs[n] for n in names]
    used = loaded_names(moving)

    # Dependencies that stay behind would break the moved code.
    stranded = sorted(n for n in used if n in defs and n not in names)
    if stranded:
        sys.exit('these definitions must move first (or belong in this module too):\n  '
                 + ', '.join(stranded))

    needed_imports = sorted({imports[n] for n in used if n in imports})
    ordered = sorted(moving, key=lambda n: n.lineno)
    body = '\n\n'.join(''.join(lines[n.lineno - 1:n.end_lineno]).rstrip('\n') for n in ordered)

    header = Path(args.header_file).read_text().rstrip('\n') + '\n' if args.header_file else ''
    module_text = header
    if needed_imports or args.extra_import:
        module_text += '\n' + '\n'.join(needed_imports + args.extra_import) + '\n'
    module_text += '\n\n' + body + '\n'

    # Delete from the entry script, bottom-up so line numbers stay valid.
    kept = list(lines)
    for node in sorted(moving, key=lambda n: n.lineno, reverse=True):
        del kept[node.lineno - 1:node.end_lineno]
    remaining = ''.join(kept)

    still_used = sorted(n for n in loaded_names([ast.parse(remaining)]) if n in names)
    if still_used:
        anchor_candidates = [
            f'from mde_gmn.{args.module} import ',
            "import cleaned_helper_functions as tsls  # noqa: E402 -- import after sys.path edit above\n",
        ]
        new_line = wrap_import(args.module, still_used)
        existing = next((a for a in anchor_candidates[:1] if a in remaining), None)
        if existing:
            sys.exit(f'{entry_path} already imports from mde_gmn.{args.module}; merge by hand')
        anchor = anchor_candidates[1]
        if remaining.count(anchor) != 1:
            sys.exit('could not find the import anchor in the entry script')
        remaining = remaining.replace(anchor, anchor + new_line + '\n')

    target = PACKAGE / f'{args.module}.py'
    print(f'module   : {target} ({len(module_text.splitlines())} lines, {len(names)} definitions)')
    print(f'imports  : {needed_imports + args.extra_import}')
    print(f'entry    : {entry_path} {len(lines)} -> {len(remaining.splitlines())} lines')
    print(f're-import: {still_used or "(nothing -- entry no longer uses these)"}')
    if args.dry_run:
        print('\n--- dry run, nothing written ---')
        return
    if target.exists():
        sys.exit(f'{target} exists; extend it by hand or pick another module name')
    target.write_text(module_text)
    entry_path.write_text(remaining)
    for path in (target, entry_path):
        ast.parse(path.read_text())      # both files must still parse
    print('\nwritten. Now run the test suite (docs/phase2-refactor-handoff.md) and commit.')


if __name__ == '__main__':
    main()
