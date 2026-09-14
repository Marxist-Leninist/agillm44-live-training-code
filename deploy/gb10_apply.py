#!/usr/bin/env python3
"""Apply a CI-qualified, SHA-pinned trainer. Credentials never leave this host.
Same-byte first adoption does not restart training. Changed code gets a clean
checkpoint stop, complete file verification, exact-state restart, and health check.
A failed trial rolls code/state back to the retained pre-trial checkpoint.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from release_lib import atomic_json, argument, digest, replace_argument, require, validate_release, validate_runtime, verified_checkpoint

BASE = Path('/workspace/agillm44-github-cd')
RUN = Path('/workspace/gb10_hf_training_20260914')
START_LOCK = Path('/workspace/gb10_resume_control_20260914/start.lock')

def identity(pid: int) -> dict:
    proc = Path('/proc') / str(pid)
    fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    require(fields[0] != 'Z', 'trainer is a zombie')
    argv = [s.decode() for s in (proc / 'cmdline').read_bytes().split(b'\0') if s]
    source = next(Path(s) for s in argv if s.endswith('.py') and 'agillm' in s)
    return {'pid': pid, 'start_ticks': fields[19], 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'argv': argv, 'source': str(source), 'source_sha256': digest(source)}

def trainers() -> list[dict]:
    result = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            argv = [s.decode() for s in (proc / 'cmdline').read_bytes().split(b'\0') if s]
            if argv and 'python' in Path(argv[0]).name and 'train' in argv and any(s.endswith('.py') and 'agillm' in s for s in argv):
                result.append(identity(int(proc.name)))
        except (OSError, ValueError, StopIteration, UnicodeError):
            continue
    return result

def same_process(item: dict) -> bool:
    try:
        current = identity(item['pid'])
        return all(current[k] == item[k] for k in ('pid', 'start_ticks', 'boot_id', 'argv', 'source_sha256'))
    except (OSError, ValueError, StopIteration):
        return False

def status(item: dict) -> dict:
    s = json.loads((RUN / 'current_run_status.json').read_text())
    require(time.time() - (RUN / 'current_run_status.json').stat().st_mtime < 120, 'stale monitor')
    require(s.get('pid') == item['pid'] and s.get('process_alive') is True, 'monitor PID mismatch')
    require(str(s.get('process_identity', {}).get('start_ticks')) == str(item['start_ticks']), 'monitor incarnation mismatch')
    require(s.get('training_progress_confirmed') is True and s.get('progress_age_sec', 1e9) < 120, 'no recent training progress')
    h = s['latest_heartbeat']
    require(h.get('finite') is True, 'non-finite heartbeat')
    return h

def active_runtime(item: dict) -> dict:
    proc = Path('/proc') / str(item['pid'])
    env = dict(s.decode().split('=', 1) for s in (proc / 'environ').read_bytes().split(b'\0') if b'=' in s)
    require('AGILLM_HOT_CONFIG' in env, 'missing active SAT-var hot configuration')
    hot_path = Path(env['AGILLM_HOT_CONFIG']); hot = json.loads(hot_path.read_text())
    if env.get('AGILLM_HOT_CONFIG_SHA256'):
        require(digest(hot_path) == env['AGILLM_HOT_CONFIG_SHA256'], 'active hot-config digest mismatch')
    validate_runtime(item['argv'], hot)
    return env


def graceful_stop(item: dict, timeout: int = 900) -> None:
    require(same_process(item), 'trainer identity changed before stop')
    caught = next(line.split()[1] for line in Path('/proc', str(item['pid']), 'status').read_text().splitlines() if line.startswith('SigCgt:'))
    require(int(caught, 16) & (1 << (signal.SIGTERM - 1)), 'SIGTERM checkpoint handler not installed')
    fd = os.pidfd_open(item['pid'])
    try:
        require(same_process(item), 'trainer changed after pidfd open')
        signal.pidfd_send_signal(fd, signal.SIGTERM)
    finally:
        os.close(fd)
    deadline = time.monotonic() + timeout
    while same_process(item):
        require(time.monotonic() < deadline, 'clean stop timed out; no force kill was sent')
        time.sleep(2)

def pin_checkpoint(save: Path, pin: Path, minimum: int) -> tuple[dict, Path]:
    m, ckpt = verified_checkpoint(save, minimum)
    pin.mkdir(parents=True, exist_ok=True)
    names = [m['checkpoint_name'], m['checkpoint_name'] + '.tokenizer.json'] + [m['shard_dir'] + '/' + row['name'] for row in m['shards']]
    for name in names:
        dst = pin / name; dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists(): os.link(save / name, dst)
    atomic_json(pin / 'training_latest.json', m)
    return verified_checkpoint(pin, minimum)

def launch(transaction: dict, source: Path, checkpoint: Path, label: str) -> dict:
    require(not trainers(), 'another trainer is already running')
    fd = os.open(START_LOCK, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not trainers(), 'another launcher won the start race')
        argv = list(transaction['old']['argv'])
        index = next(i for i, value in enumerate(argv) if value == transaction['old']['source'])
        argv[index] = str(source)
        argv = replace_argument(argv, '--resume', str(checkpoint))
        env = json.loads(Path(transaction['private_environment']).read_text())
        log = Path(transaction['directory']) / (label + '.log')
        with log.open('ab', buffering=0) as stream:
            child = subprocess.Popen(argv, env=env, cwd=transaction['cwd'], stdin=subprocess.DEVNULL, stdout=stream, stderr=stream, start_new_session=True, pass_fds=(fd,))
        time.sleep(2)
        require(child.poll() is None, 'trainer exited during startup; see host-local trial log')
        item = identity(child.pid)
        require(item['argv'] == argv, 'launched command mismatch')
        target = RUN / 'trainer.pid'; temporary = target.with_name(target.name + '.cd-tmp')
        temporary.write_text(str(child.pid) + '\n'); os.replace(temporary, target)
        return item
    finally:
        os.close(fd)

def wait_health(item: dict, minimum_step: int, timeout: int = 600) -> dict:
    deadline = time.monotonic() + timeout
    advances = 0; previous = minimum_step; latest = {}
    while time.monotonic() < deadline:
        require(same_process(item), 'candidate exited or process identity changed')
        try:
            latest = status(item)
            if latest['step'] > previous:
                advances += 1; previous = latest['step']
            if advances >= 3: return latest
        except (OSError, KeyError, ValueError):
            pass
        time.sleep(5)
    raise ValueError('candidate did not pass three finite, advancing heartbeat checks')

def receipt(state: str, sha: str, **fields) -> dict:
    value = {'schema': 'agillm44.github-deployment.v1', 'state': state, 'commit': sha, 'checked_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields}
    atomic_json(BASE / 'status.json', value)
    print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)
    return value

def rollback_or_raise(error: Exception, transaction: dict, transaction_path: Path, manifest: dict, commit: str) -> None:
    old = transaction['old']
    current = trainers()
    candidate = transaction.get('candidate')
    if candidate and same_process(candidate):
        graceful_stop(candidate)
        current = trainers()
    if not current and transaction.get('checkpoint'):
        _, rollback_ckpt = verified_checkpoint(Path(transaction['checkpoint']).parent, transaction['checkpoint_step'])
        restored = launch(transaction, Path(old['source']), rollback_ckpt, 'rollback')
        h = wait_health(restored, transaction['checkpoint_step'])
        atomic_json(BASE / 'rejected' / (commit + '.json'), {'reason': str(error), 'source_sha256': manifest['source_sha256']})
        transaction.update(phase='ROLLED_BACK', rollback=restored); atomic_json(transaction_path, transaction)
        receipt('ROLLED_BACK', commit, reason=str(error), pid=restored['pid'], step=h['step'], rollback_checkpoint_step=transaction['checkpoint_step'])
        return
    receipt('BLOCKED_OR_RECOVERY_REQUIRED', commit, reason=str(error), live_pids=[item['pid'] for item in current])
    raise error

def apply(root: Path, commit: str) -> None:
    require(re.fullmatch(r'[0-9a-f]{40}', commit) is not None, 'invalid Git commit')
    manifest = validate_release(root); source = root / manifest['source']
    transaction_path = BASE / 'transactions' / commit / 'transaction.json'
    if (BASE / 'rejected' / (commit + '.json')).exists():
        receipt('REJECTED_PREVIOUSLY', commit); return
    live = trainers(); require(len(live) <= 1, 'multiple trainers; refusing to choose one')
    transaction = json.loads(transaction_path.read_text()) if transaction_path.exists() else None
    if live and live[0]['source_sha256'] == manifest['source_sha256']:
        item = live[0]
        active_runtime(item)
        if transaction and transaction.get('phase') == 'STARTED':
            try:
                h = wait_health(item, transaction['checkpoint_step'])
                transaction['phase'] = 'SUCCESS'; atomic_json(transaction_path, transaction)
            except Exception as error:
                rollback_or_raise(error, transaction, transaction_path, manifest, commit)
                return
        else:
            h = status(item)
        state = 'DEPLOYED' if item['source'] == str(source) else 'BOUND_IDENTICAL_NO_RESTART'
        receipt(state, commit, source_sha256=manifest['source_sha256'], live_source=item['source'], canonical_release_source=str(source), pid=item['pid'], step=h['step'], trainer_restarted=bool(transaction), performance_qualification='NOT_MEASURED_BY_CI', training_state_modified=bool(transaction))
        return
    if transaction is None:
        if not live:
            receipt('WAITING_FOR_EXISTING_TRAINER', commit, reason='No unrequested cold start; the existing training lane controls its launch.')
            return
        old = live[0]
        if old['source_sha256'] != manifest['parent_source_sha256']:
            receipt('LIVE_SOURCE_DRIFT', commit, pid=old['pid'], expected_parent_sha256=manifest['parent_source_sha256'], observed_source_sha256=old['source_sha256'], live_source=old['source'], trainer_restarted=False, reason='Rebase release on the active source; experimental/live code was not overwritten.')
            return
        baseline = status(old)
        proc = Path('/proc') / str(old['pid'])
        env = active_runtime(old)
        # CLI parsing in the target runtime, without starting a training command.
        check = subprocess.run([old['argv'][0], str(source), 'train', '--help'], env=env, capture_output=True, timeout=120)
        require(check.returncode == 0, 'candidate CLI/runtime smoke failed; live trainer untouched')
        directory = transaction_path.parent; directory.mkdir(parents=True, exist_ok=True)
        private = directory / 'environment.private.json'; atomic_json(private, env)
        transaction = {'old': old, 'phase': 'PREPARED', 'directory': str(directory), 'private_environment': str(private), 'cwd': str(proc.joinpath('cwd').resolve()), 'save_dir': argument(old['argv'], '--save_dir'), 'minimum_step': baseline['step'], 'requested_epoch': time.time()}
        atomic_json(transaction_path, transaction)
    old = transaction['old']
    try:
        if transaction['phase'] in ('PREPARED', 'STOP_REQUESTED'):
            if live:
                require(same_process(old), 'a different trainer is active; no cutover attempted')
                transaction['phase'] = 'STOP_REQUESTED'; atomic_json(transaction_path, transaction)
                graceful_stop(old)
            require(not trainers(), 'another trainer started; cutover aborted')
            save = Path(transaction['save_dir'])
            require((save / 'training_latest.json').stat().st_mtime >= transaction['requested_epoch'] - 1, 'no fresh shutdown checkpoint')
            m, checkpoint = pin_checkpoint(save, transaction_path.parent / 'rollback_checkpoint', transaction['minimum_step'])
            transaction.update(phase='CHECKPOINT_VERIFIED', checkpoint=str(checkpoint), checkpoint_step=m['step'])
            atomic_json(transaction_path, transaction)
        require(transaction['phase'] == 'CHECKPOINT_VERIFIED', 'unexpected transaction state; inspect local transaction')
        candidate = launch(transaction, source, Path(transaction['checkpoint']), 'candidate')
        transaction.update(phase='STARTED', candidate=candidate); atomic_json(transaction_path, transaction)
        h = wait_health(candidate, transaction['checkpoint_step'])
        transaction['phase'] = 'SUCCESS'; atomic_json(transaction_path, transaction)
        receipt('DEPLOYED', commit, source_sha256=manifest['source_sha256'], live_source=str(source), pid=candidate['pid'], resume_step=transaction['checkpoint_step'], step=h['step'], trainer_restarted=True, checkpoint_verified=True, performance_qualification='NOT_MEASURED_BY_CI')
    except Exception as error:
        rollback_or_raise(error, transaction, transaction_path, manifest, commit)

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument('--release', required=True, type=Path); parser.add_argument('--commit', required=True)
    args = parser.parse_args(); BASE.mkdir(parents=True, exist_ok=True)
    with (BASE / 'deploy.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            apply(args.release.resolve(), args.commit)
        except Exception as error:
            receipt('BLOCKED_OR_RECOVERY_REQUIRED', args.commit, reason=str(error))

if __name__ == '__main__':
    main()
