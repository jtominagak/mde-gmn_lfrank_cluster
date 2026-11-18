#!/usr/bin/env python3
"""clean_sync.py -- keep clean_scripts/ in step with scripts/ and tests/.

clean_scripts/ is the shareable copy of the pipeline: the same behaviour,
restructured to be readable (see clean_scripts/README.md). It is NOT a
mechanical copy, so a change in scripts/ cannot be synced by a script. What a
script CAN do is know, exactly, which changes have not been ported yet. That
is this file.

For every file in scripts/ and tests/ it knows the clean counterpart (MAPPING)
or why there is none (IGNORE), and it records in clean_sync_state.json the
git blob of the version that was last ported. A file whose content differs
from that blob has drifted. The drift is classified:

  CODE     the code changed (docstring-stripped AST for .py, comment-stripped
           lines otherwise). Must be ported.
  TEXT     only comments/docstrings changed. Port if the explanation matters
           to a reader of clean_scripts, then mark synced.
  NEW      a mapped file with no recorded baseline. Must be ported.
  DELETED  a recorded file is gone. Remove/adjust its counterpart.
  UNCLASSIFIED  a file in scripts/ or tests/ that is neither in MAPPING nor in
           IGNORE. Decide which, and add it to the lists below.

Usage (run from anywhere; python3 >= 3.6, stdlib only):
  clean_sync.py [status]               what still needs porting (exit 1 if any)
  clean_sync.py diff [PATH ...]        the change to port, per drifted file
  clean_sync.py mark-synced PATH ...   record PATH's current content as ported
  clean_sync.py mark-synced --all      ... for every mapped file
       --note TEXT                     why, when a change is deliberately not mirrored
  clean_sync.py hook-stop | hook-post-edit | hook-pre-commit
                                       Claude Code / git hook entry points

mark-synced writes each blob into the git object store (git hash-object -w),
so the baseline stays diffable even if it was never committed. Once the same
content is committed, the commit keeps that blob alive permanently.

The porting procedure itself lives in .claude/skills/sync-clean-scripts/SKILL.md.
"""
import ast
import datetime
import difflib
import fnmatch
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE_FILE = os.path.join(REPO, 'scripts', 'dev', 'clean_sync_state.json')
WATCHED_ROOTS = ('scripts', 'tests')

# (pattern, [clean counterparts], porting note). First match wins.
# '{name}' in a counterpart is replaced by the source file's basename.
MAPPING = [
    ('scripts/src/cleaned_helper_functions.py', ['clean_scripts/src/edm.py'],
     'renamed edm.py (callers import it as `edm`, not `tsls`); the dead '
     'mean_in_time_grid/multi_simplex/rolling_simplex/top_neurons are not carried'),
    ('scripts/src/gpu_budget.py', ['clean_scripts/src/gpu_budget.py'], ''),
    ('scripts/src/gpu_edm.py', ['clean_scripts/src/gpu_edm.py'], ''),
    ('scripts/src/no-kedm-optimization-parallelized-mde--tsls-mde-gmn.py',
     ['clean_scripts/src/run_parallel_mde.py'], ''),
    ('scripts/src/tsls-mde-gmn.py', ['clean_scripts/src/run_serial_mde.py'], ''),
    ('scripts/src/mde_gmn/network.py',
     ['clean_scripts/src/mde_gmn/network.py', 'clean_scripts/src/mde_gmn/waves.py'],
     'the Stage 1/Stage 2 pool loops live in waves.run_mde_batch()'),
    ('scripts/src/mde_gmn/network_parallel.py',
     ['clean_scripts/src/mde_gmn/network_parallel.py', 'clean_scripts/src/mde_gmn/waves.py'],
     'the precompute pool loop lives in waves.run_mde_batch()'),
    ('scripts/src/mde_gmn/config.py',
     ['clean_scripts/src/mde_gmn/config.py', 'clean_scripts/batch/saion_parallel_mde.sbatch',
      'clean_scripts/batch/serial_mde.sbatch', 'clean_scripts/docs/parameters.md'],
     'a new TSLS_MDE_GMN_* variable also goes in BOTH sbatch export loops'),
    ('scripts/src/mde_gmn/*.py', ['clean_scripts/src/mde_gmn/{name}'], ''),
    ('scripts/batch/saion-no-kedm-optimization-parallelized-mde--tsls-mde-gmn.sbatch',
     ['clean_scripts/batch/saion_parallel_mde.sbatch', 'clean_scripts/batch/serial_mde.sbatch',
      'clean_scripts/docs/parameters.md'],
     'serial_mde.sbatch is its CPU twin: shared parameter values stay identical. '
     'Long rationale goes to docs/parameters.md, not the sbatch'),
    ('scripts/batch/submit.sh', ['clean_scripts/batch/submit.sh'], ''),
    ('scripts/batch/generate_job_description.sh',
     ['clean_scripts/batch/generate_job_description.sh'],
     'the clean version diffs every .py under src/, not just the entry script'),
    ('scripts/saion/saion_env.sh', ['clean_scripts/saion/cluster_env.sh'],
     'no account-specific paths: environment variables with working defaults'),
    ('scripts/saion/setup_gpu_venv.sh', ['clean_scripts/saion/setup_gpu_venv.sh'],
     'the clean version installs env/sitecustomize.py from the tree'),
    ('scripts/saion/submit_saion.sh', ['clean_scripts/saion/submit_saion.sh'], ''),
    ('scripts/saion/submit_saion_scope_pair.sh',
     ['clean_scripts/saion/submit_saion_scope_pair.sh'], ''),
    ('scripts/env/sitecustomize.py', ['clean_scripts/env/sitecustomize.py'],
     'code stays byte-identical below the module docstring'),
    ('tests/_pipeline_test_utils.py', ['clean_scripts/tests/_pipeline_test_utils.py'], ''),
    ('tests/test_*.py', ['clean_scripts/tests/{name}'],
     'adapt paths and renamed modules; drop any half that exercises simple-mde-gmn.py'),
]

# (pattern, why it has no clean counterpart). Checked before MAPPING.
IGNORE = [
    ('scripts/src/simple-mde-gmn.py', 'the 1 Hz pipeline: out of scope by design'),
    ('scripts/src/tsls_example.py', 'example script, out of scope'),
    ('scripts/backup/*', 'superseded backups'),
    ('scripts/calibration/*', 'one-off calibration; its results are in clean_scripts/docs/parameters.md'),
    ('scripts/gpu_prototypes/*', 'one-off GPU prototypes'),
    ('scripts/dev/*', 'local-development tooling, including this file'),
    ('scripts/batch/simple-mde-gmn.sbatch', 'the 1 Hz pipeline'),
    ('scripts/batch/old_simple_mde_gmn.sbatch', 'the 1 Hz pipeline'),
    ('scripts/batch/tsls_example.sbatch', 'example, out of scope'),
    ('scripts/batch/tsls-mde-gmn-gpu-*.sbatch', 'near-duplicate variants, collapsed into saion_parallel_mde.sbatch'),
    ('scripts/batch/no-kedm-optimization-parallelized-mde--tsls-mde-gmn.sbatch',
     'the deigo variant, collapsed into saion_parallel_mde.sbatch'),
    ('scripts/batch/tsls-mde-gmn.sbatch',
     'the deigo serial run; clean serial_mde.sbatch is the CPU twin of the saion parallel sbatch, not a port of this'),
    ('scripts/batch/saion-preflight.sbatch', 'this account\'s venv preflight'),
    ('scripts/saion/gpu_capacity_probe.py', 'one-off probe'),
    ('scripts/env/README.md', 'clean_scripts/env/README.md is written for that tree'),
    ('scripts/env/requirements-*-frozen.txt', 'version manifests of this cluster\'s venvs'),
]

BLOCKING = ('CODE', 'NEW', 'DELETED', 'UNCLASSIFIED')


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------
def git(*args, **kw):
    r = subprocess.run(['git', '-C', REPO] + list(args), stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, input=kw.get('input'))
    if r.returncode != 0 and not kw.get('ok_to_fail'):
        raise RuntimeError('git %s: %s' % (' '.join(args), r.stderr.decode(errors='replace').strip()))
    return r


def blob_ids(paths, write=False):
    """Git blob id of each path's working-tree content."""
    if not paths:
        return {}
    args = ['hash-object', '--stdin-paths'] + (['-w'] if write else [])
    out = git(*args, input='\n'.join(paths).encode()).stdout.decode().split()
    return dict(zip(paths, out))


def blob_text(blob):
    r = git('cat-file', '-p', blob, ok_to_fail=True)
    return r.stdout.decode('utf-8', errors='replace') if r.returncode == 0 else None


def watched_files():
    out = git('ls-files', '--cached', '--others', '--exclude-standard', '--', *WATCHED_ROOTS).stdout
    return sorted(p for p in out.decode().splitlines() if os.path.isfile(os.path.join(REPO, p)))


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------
def counterparts(path):
    for pattern, targets, note in MAPPING:
        if fnmatch.fnmatch(path, pattern):
            name = os.path.basename(path)
            return [t.replace('{name}', name) for t in targets], note
    return None, None


def ignored(path):
    return any(fnmatch.fnmatch(path, pattern) for pattern, _ in IGNORE)


def _is_docstring(node):
    if not isinstance(node, ast.Expr):
        return False
    v = node.value
    if isinstance(v, ast.Constant) and isinstance(v.value, str):
        return True
    return sys.version_info < (3, 8) and isinstance(v, getattr(ast, 'Str', ()))


def code_view(path, text):
    """The text with comments and docstrings removed, for CODE vs TEXT."""
    if path.endswith('.py'):
        try:
            tree = ast.parse(text)
        except SyntaxError:
            pass
        else:
            for node in ast.walk(tree):
                body = getattr(node, 'body', None)
                if isinstance(body, list) and body and _is_docstring(body[0]):
                    node.body = body[1:] or [ast.Pass()]
            return ast.dump(tree)
    keep = []
    for line in text.splitlines():
        s = line.strip()
        if not s or (s.startswith('#') and not s.startswith(('#!', '#SBATCH'))):
            continue
        keep.append(' '.join(_strip_inline_comment(s).split()))
    return '\n'.join(keep)


def _strip_inline_comment(line):
    """Drop a trailing ` # ...` that is not inside quotes (shell-ish files)."""
    single = double = False
    for i, ch in enumerate(line):
        if ch == "'" and not double:
            single = not single
        elif ch == '"' and not single:
            double = not double
        elif ch == '#' and not single and not double and i > 0 and line[i - 1] in ' \t':
            return line[:i].rstrip()
    return line


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (IOError, ValueError):
        return {'files': {}}


def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=1, sort_keys=True)
        f.write('\n')


def scan():
    """[(kind, path, counterparts, note)] for every file that is not in sync."""
    state = load_state()['files']
    files = watched_files()
    mapped = [p for p in files if not ignored(p) and counterparts(p)[0]]
    current = blob_ids(mapped)
    report = []
    for p in files:
        if ignored(p):
            continue
        targets, note = counterparts(p)
        if targets is None:
            report.append(('UNCLASSIFIED', p, [], 'add it to MAPPING or IGNORE in scripts/dev/clean_sync.py'))
            continue
        rec = state.get(p)
        if rec is None:
            report.append(('NEW', p, targets, note))
            continue
        if rec['blob'] == current[p]:
            continue
        old = blob_text(rec['blob'])
        with open(os.path.join(REPO, p), encoding='utf-8', errors='replace') as f:
            new = f.read()
        if old is not None and code_view(p, old) == code_view(p, new):
            report.append(('TEXT', p, targets, note))
        else:
            report.append(('CODE', p, targets, note))
    for p in sorted(set(state) - set(files)):
        targets, note = counterparts(p)
        report.append(('DELETED', p, targets or [], note))
    return report


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def format_report(report):
    if not report:
        return 'clean_scripts/ is in sync with scripts/ and tests/.'
    lines = []
    for kind, p, targets, note in report:
        lines.append('  %-12s %s' % (kind, p))
        if targets:
            lines.append('               -> ' + ', '.join(targets))
        if note:
            lines.append('               (' + note + ')')
    n = sum(1 for r in report if r[0] in BLOCKING)
    head = ('clean_scripts/ is behind on %d file(s):' % n if n
            else 'clean_scripts/ code is in sync; comment/docstring-only changes to review:')
    return head + '\n' + '\n'.join(lines)


def cmd_status(_args):
    report = scan()
    print(format_report(report))
    return 1 if any(r[0] in BLOCKING for r in report) else 0


def cmd_diff(args):
    state = load_state()['files']
    report = {r[1]: r for r in scan()}
    paths = args or sorted(p for p, r in report.items() if r[0] != 'UNCLASSIFIED')
    for p in paths:
        kind, _, targets, note = report.get(p, ('IN SYNC', p, counterparts(p)[0] or [], ''))
        print('=' * 78)
        print('%s  %s  ->  %s' % (kind, p, ', '.join(targets)))
        if note:
            print('note: ' + note)
        print('=' * 78)
        full = os.path.join(REPO, p)
        new = open(full, encoding='utf-8', errors='replace').read() if os.path.exists(full) else ''
        rec = state.get(p)
        old = blob_text(rec['blob']) if rec else ''
        if old is None:
            print('(baseline blob %s is no longer in the object store; compare the file '
                  'against its counterpart directly)' % rec['blob'])
            continue
        sys.stdout.writelines(difflib.unified_diff(
            old.splitlines(True), new.splitlines(True),
            'a/%s (last ported)' % p, 'b/%s (now)' % p))
    return 0


def cmd_mark_synced(args):
    note = ''
    if '--note' in args:
        i = args.index('--note')
        note = ' '.join(args[i + 1:i + 2])
        args = args[:i] + args[i + 2:]
    files = watched_files()
    if args == ['--all']:
        unclassified = [p for p in files if not ignored(p) and counterparts(p)[0] is None]
        if unclassified:
            print('refusing --all: classify these first (MAPPING or IGNORE):\n  ' + '\n  '.join(unclassified))
            return 1
        paths = [p for p in files if not ignored(p)]
    else:
        paths = [os.path.relpath(os.path.abspath(a), REPO) for a in args]
        bad = [p for p in paths if ignored(p) or counterparts(p)[0] is None]
        if bad or not paths:
            print('not mapped files: %s' % (bad or 'none given'))
            return 1
    state = load_state()
    present = [p for p in paths if os.path.exists(os.path.join(REPO, p))]
    blobs = blob_ids(present, write=True)
    today = datetime.date.today().isoformat()
    head = git('rev-parse', '--short', 'HEAD').stdout.decode().strip()
    for p in paths:
        if p in blobs:
            state['files'][p] = {'blob': blobs[p], 'synced_on': today, 'head': head, 'note': note}
        else:
            state['files'].pop(p, None)
    if args == ['--all']:
        for p in list(state['files']):
            if p not in blobs:
                del state['files'][p]
    save_state(state)
    print('marked %d file(s) as ported%s' % (len(paths), (' -- ' + note) if note else ''))
    return 0


def _hook_input():
    try:
        return json.loads(sys.stdin.read() or '{}')
    except ValueError:
        return {}


def _hooks_active():
    return (os.environ.get('CLEAN_SYNC_HOOK', 'on') != 'off'
            and os.path.isdir(os.path.join(REPO, 'clean_scripts')))


def cmd_hook_stop(_args):
    data = _hook_input()
    if data.get('stop_hook_active') or not _hooks_active():
        return 0
    blocking = [r for r in scan() if r[0] in BLOCKING]
    if blocking:
        print(json.dumps({'decision': 'block', 'reason': (
            format_report(blocking) + '\n\nPort these into clean_scripts/ before finishing, '
            'following .claude/skills/sync-clean-scripts/SKILL.md (the /sync-clean-scripts '
            'skill). If a change must NOT be mirrored, record why instead: '
            "python3 scripts/dev/clean_sync.py mark-synced PATH --note '...'. "
            'If the user asked to defer syncing, say so and stop.')}))
    return 0


def cmd_hook_post_edit(_args):
    data = _hook_input()
    if not _hooks_active():
        return 0
    f = (data.get('tool_input') or {}).get('file_path') or (data.get('tool_input') or {}).get('notebook_path')
    if not f:
        return 0
    p = os.path.relpath(os.path.abspath(f), REPO)
    if not p.startswith(tuple(r + os.sep for r in WATCHED_ROOTS)) or ignored(p):
        return 0
    targets, note = counterparts(p)
    if targets is None:
        msg = ('%s is new to clean-sync: decide whether it belongs in clean_scripts/ and add it '
               'to MAPPING or IGNORE in scripts/dev/clean_sync.py.' % p)
    else:
        msg = ('%s has a clean counterpart: %s.%s Mirror this change there, in clean_scripts\' '
               'style, before finishing (see /sync-clean-scripts); the Stop hook checks.'
               % (p, ', '.join(targets), (' Note: ' + note + '.') if note else ''))
    print(json.dumps({'hookSpecificOutput': {'hookEventName': 'PostToolUse',
                                             'additionalContext': msg}}))
    return 0


def cmd_hook_pre_commit(_args):
    if not _hooks_active():
        return 0
    blocking = [r for r in scan() if r[0] in BLOCKING]
    if blocking:
        sys.stderr.write('\n[clean-sync] warning -- committing anyway:\n' + format_report(blocking)
                         + '\n[clean-sync] port with /sync-clean-scripts in Claude Code, or see '
                           'scripts/dev/clean_sync.py\n\n')
    return 0


COMMANDS = {
    'status': cmd_status, 'diff': cmd_diff, 'mark-synced': cmd_mark_synced,
    'hook-stop': cmd_hook_stop, 'hook-post-edit': cmd_hook_post_edit,
    'hook-pre-commit': cmd_hook_pre_commit,
}


def main(argv):
    cmd = argv[0] if argv else 'status'
    if cmd not in COMMANDS:
        print(__doc__)
        return 2
    if cmd.startswith('hook-'):
        # A hook must never break the session or the commit it is attached to.
        try:
            return COMMANDS[cmd](argv[1:])
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write('[clean-sync] hook error (ignored): %r\n' % (exc,))
            return 0
    return COMMANDS[cmd](argv[1:])


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
