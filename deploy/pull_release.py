#!/usr/bin/env python3
"""SG-side pull deployment. Only successful push-to-main CI for current HEAD.
Install this controller separately; it deliberately does not self-update from PRs.
GitHub/cloud credentials remain on SG; GB10 receives source and a public manifest.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from release_lib import atomic_json, require, validate_release


def run(argv: list[str], timeout: int = 60) -> bytes:
    result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if result.returncode:
        # Do not include potentially sensitive command output in a public receipt.
        raise RuntimeError('command failed: ' + Path(argv[0]).name + ' rc=' + str(result.returncode))
    return result.stdout


def github(endpoint: str) -> dict:
    return json.loads(run(['gh', 'api', endpoint]))


def qualified_run(repo: str, head: str, workflow: str) -> dict | None:
    runs = github(f'repos/{repo}/actions/workflows/{workflow}/runs?branch=main&event=push&per_page=20')['workflow_runs']
    candidates = [r for r in runs if r.get('head_sha') == head and r.get('head_branch') == 'main' and r.get('event') == 'push' and r.get('path') == '.github/workflows/' + workflow and r.get('head_repository', {}).get('full_name') == repo]
    if not candidates:
        return None
    latest = max(candidates, key=lambda r: (r['id'], r.get('run_attempt', 1)))
    return latest if latest.get('status') == 'completed' and latest.get('conclusion') == 'success' else None


def tick(config_path: Path) -> None:
    c = json.loads(config_path.read_text()); root = Path(c['state_dir'])
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'pull.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        repo = c['repository']; require(re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo) is not None, 'invalid repository')
        head = github(f'repos/{repo}/git/ref/heads/main')['object']['sha']
        require(re.fullmatch(r'[0-9a-f]{40}', head) is not None, 'invalid HEAD')
        qualified = qualified_run(repo, head, 'trainer-ci.yml')
        if not qualified:
            atomic_json(root / 'status.json', {'state': 'WAITING_FOR_SUCCESSFUL_MAIN_CI', 'commit': head, 'checked_epoch': time.time()})
            print('WAITING_FOR_SUCCESSFUL_MAIN_CI ' + head); return
        release = root / 'releases' / head; (release / 'trainer').mkdir(parents=True, exist_ok=True)
        for name in ('release.json', 'trainer/agillm44.py'):
            payload = run(['gh', 'api', f'repos/{repo}/contents/{name}?ref={head}', '-H', 'Accept: application/vnd.github.raw+json'])
            temporary = release / (name + '.tmp'); temporary.write_bytes(payload); os.replace(temporary, release / name)
        manifest = validate_release(release)
        # Recheck HEAD after downloads, never deploy a stale queued commit.
        require(github(f'repos/{repo}/git/ref/heads/main')['object']['sha'] == head, 'HEAD advanced during staging')
        host = c['host']; port = str(c['port']); key = c['ssh_key']
        require(re.fullmatch(r'[A-Za-z0-9.-]+', host) is not None and port.isdigit(), 'invalid SSH destination')
        remote = '/workspace/agillm44-github-cd/releases/' + head
        common = ['-i', key, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15']
        ssh = ['ssh', *common, '-p', port, 'root@' + host]
        run([*ssh, 'mkdir -p ' + shlex.quote(remote + '/trainer')])
        for name in ('release.json', 'trainer/agillm44.py'):
            run(['scp', '-q', *common, '-P', port, str(release / name), 'root@' + host + ':' + remote + '/' + name], timeout=120)
        command = 'python3 /workspace/agillm44-github-cd/controller/gb10_apply.py --release ' + shlex.quote(remote) + ' --commit ' + head
        output = run([*ssh, command], timeout=3300)
        records = [json.loads(line) for line in output.decode().splitlines() if line.startswith('{')]
        require(bool(records), 'missing deployment receipt')
        receipt = records[-1]
        receipt.update(ci_run_id=qualified['id'], ci_run_url=qualified['html_url'], repository=repo, source_sha256=manifest['source_sha256'], checked_epoch=time.time())
        atomic_json(root / 'status.json', receipt)
        atomic_json(root / 'receipts' / (head + '.json'), receipt)
        print(json.dumps(receipt, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    try: tick(args.config)
    except Exception as error:
        print(json.dumps({'state': 'ERROR', 'error': str(error)}, sort_keys=True), flush=True)
        raise

if __name__ == '__main__': main()
