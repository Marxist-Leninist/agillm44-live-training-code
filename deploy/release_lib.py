"""Dependency-free release validation. Never imports or starts the trainer."""
from __future__ import annotations
import ast
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path

SCHEMA = 'agillm44.github-release.v1'
CAPABILITIES = ('AR', 'dynamic-SAT-var', 'NAT', 'Direct56', 'exact-resume')

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('w', encoding='utf-8') as stream:
        os.chmod(temporary, 0o600)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)

def source_checks(path: Path) -> dict:
    require(not path.is_symlink() and path.is_file(), 'source must be a regular file')
    text = path.read_text(encoding='utf-8')
    tree = ast.parse(text, filename=str(path))
    compile(tree, str(path), 'exec')
    folded = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_AGILLM_SF_SOURCE' for t in node.targets):
            require(isinstance(node.value, ast.Constant) and isinstance(node.value.value, str), 'folded source must be literal')
            compile(node.value.value, str(path) + '#folded-' + str(folded), 'exec')
            folded += 1
    for token in ('SATHead', 'NATHead', 'gate_conf', 'dblock_satvar_enabled', '_dblock_direct56', 'require_exact_resume_state', '_on_terminate_signal', '_agillm43_sanitize_json_floats'):
        require(token in text, 'missing capability marker: ' + token)
    for name, pattern in {'github-token': r'gh[pousr]_[A-Za-z0-9_]{30,}', 'hf-token': r'hf_[A-Za-z0-9]{30,}', 'private-key': r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----'}.items():
        require(re.search(pattern, text) is None, 'possible secret detected: ' + name)
    return {'sha256': digest(path), 'folded_modules_compiled': folded, 'syntax': 'passed', 'capability_markers': 'passed', 'secret_patterns': 'passed'}

def validate_release(root: Path) -> dict:
    manifest = json.loads((root / 'release.json').read_text())
    require(manifest.get('schema') == SCHEMA, 'unknown release schema')
    require(manifest.get('source') == 'trainer/agillm44.py', 'unexpected source path')
    require(manifest.get('resume_schema') == 'agillm43.training.resume.v2', 'resume schema changed')
    require(manifest.get('capabilities') == list(CAPABILITIES), 'capability contract changed')
    require(manifest.get('deployment') == 'checkpoint-safe', 'unexpected deployment mode')
    for key in ('source_sha256', 'parent_source_sha256'):
        require(isinstance(manifest.get(key), str) and re.fullmatch(r'[0-9a-f]{64}', manifest[key]) is not None, 'invalid ' + key)
    checks = source_checks(root / manifest['source'])
    require(checks['sha256'] == manifest['source_sha256'], 'source digest mismatch')
    return manifest

def argument(argv: list[str], key: str) -> str:
    require(argv.count(key) == 1, 'missing or duplicate argument: ' + key)
    i = argv.index(key)
    require(i + 1 < len(argv) and not argv[i + 1].startswith('--'), 'missing argument value: ' + key)
    return argv[i + 1]

def replace_argument(argv: list[str], key: str, value: str) -> list[str]:
    argument(argv, key)
    result = list(argv); result[result.index(key) + 1] = value
    return result

def validate_runtime(argv: list[str], hot: dict) -> None:
    require('train' in argv and '--require_exact_resume_state' in argv and '--dblock' in argv, 'exact-resume DBlock required')
    for flag in ('--fresh', '--reset_optimizer_on_resume', '--lr_schedule_reset_on_resume', '--lr_schedule_reanchor_on_resume'):
        require(flag not in argv, 'state reset forbidden: ' + flag)
    for flag in ('--dblock_ar_prob', '--dblock_sat_prob', '--dblock_nat_prob'):
        value = float(argument(argv, flag)); require(math.isfinite(value) and value > 0, 'all three modes must remain active')
    for flag in ('--sat_every', '--nat_every'):
        require(int(argument(argv, flag)) > 0, 'SAT/NAT cadence disabled')
    require(int(hot.get('dblock_satvar_enabled', 0)) == 1, 'dynamic SAT-var must remain enabled')
    require(hot.get('owner_satvar_must_choose') is True, 'SAT-var must retain its choose-both contract')
    distribution = dict((int(k), float(v)) for k, v in (part.split(':') for part in str(hot.get('dblock_satvar_block_probs', '')).split(',')))
    require(set(distribution) == {1, 2} and all(math.isfinite(v) and v > 0 for v in distribution.values()), 'SAT-var cannot be fixed at one or two tokens')
    require(int(argument(argv, '--dblock_direct56_enabled')) == 1, 'Direct56 must remain enabled')

def verified_checkpoint(directory: Path, minimum_step: int) -> tuple[dict, Path]:
    pointer = directory / 'training_latest.json'
    require(not pointer.is_symlink(), 'symlink pointer refused')
    raw = pointer.read_bytes(); m = json.loads(raw)
    require(m.get('schema') == 'agillm43.training.resume.v2' and m.get('role') == 'training', 'not a training resume pointer')
    require(type(m.get('step')) is int and m['step'] >= minimum_step, 'checkpoint is stale')
    name = m['checkpoint_name']; shard_dir = m['shard_dir']
    require(Path(name).name == name and name not in ('', '.', '..') and shard_dir == name + '.shards', 'unsafe checkpoint names')
    rows = [(name, m['checkpoint_main_sha256'], m['checkpoint_main_nbytes']), (name + '.tokenizer.json', m['tokenizer_sidecar_sha256'], m['tokenizer_sidecar_nbytes'])]
    for row in m['shards']:
        require(Path(row['name']).name == row['name'] and row['name'] not in ('', '.', '..'), 'unsafe shard name')
        rows.append((shard_dir + '/' + row['name'], row['sha256'], row['nbytes']))
    require(len(rows) >= 5 and len({r[0] for r in rows}) == len(rows), 'missing or duplicate checkpoint files')
    required = {'core_base.pt', 'ar.pt', 'sat.pt', 'nat.pt', 'opt.pt', 'scaler.pt', 'detachable_50m_base.pt', 'agillm5_detachable_50m_bridge.pt', 'agillm5_detachable_50m_opt.pt', 'agillm5_detachable_50m_scaler.pt', 'agillm5_detachable_50m_rng.pt', 'agillm5_detachable_50m_counters.pt'}
    required |= {'core_block_%04d.pt' % i for i in range(28)}
    required |= {'detachable_50m_block_%04d.pt' % i for i in range(22)}
    require(len(rows) == 64 and len(m['shards']) == 62, 'GB10 Direct56 checkpoint must contain all 64 files')
    require(required.issubset({row['name'] for row in m['shards']}), 'incomplete model/optimizer state')
    for relative, expected, size in rows:
        path = directory / relative
        require(not path.is_symlink() and not path.parent.is_symlink() and path.resolve().is_relative_to(directory.resolve()), 'unsafe checkpoint path')
        before = path.stat()
        require(stat.S_ISREG(before.st_mode) and before.st_size == int(size), 'checkpoint file size mismatch')
        require(digest(path) == expected, 'checkpoint file hash mismatch: ' + relative)
        after = path.stat()
        require((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns), 'checkpoint mutated during verification')
    require(pointer.read_bytes() == raw, 'checkpoint pointer changed during verification')
    return m, directory / name
