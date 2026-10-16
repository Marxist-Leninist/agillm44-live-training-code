import sys
f=sys.argv[1]; s=open(f).read()
def rep(a,b):
    global s
    assert s.count(a)==1, a
    s=s.replace(a,b)
# 1) byte data -> vocab 256 (a 129280x8192 fp32 tied table + AdamW is ~17GB: cannot fit a 12GB card; bytes never exceed 255)
rep("    if a.ctx:\n        c['ctx'] = a.ctx\n",
    "    if a.ctx:\n        c['ctx'] = a.ctx\n    if a.vocab:\n        c['vocab'] = a.vocab\n    elif a.data.startswith(('file:', 'parquet:')) and c['vocab'] > 256:\n        print(f\"byte data: vocab {c['vocab']} -> 256 (UTF-8 bytes; the big table would not fit the GPU)\", flush=True)\n        c['vocab'] = 256\n")
rep("    sp.add_argument('--stream'", "    sp.add_argument('--vocab', type=int, default=0, help='override the preset vocab (byte data defaults to 256)')\n    sp.add_argument('--hf-every-h', type=float, default=6.0, help='push to HF at most this often (local saves keep --save-min)')\n    sp.add_argument('--stream'")
# 2) HF upload throttle
rep("    if HF_REPO:\n        import subprocess\n",
    "    stamp = os.path.join(run, 'hf_last_upload')\n    if HF_REPO and os.path.exists(stamp) and time.time() - os.path.getmtime(stamp) < HF_EVERY_H * 3600:\n        return  # frugal with HF: local save only\n    if HF_REPO:\n        import subprocess\n        open(stamp, 'w').write(str(step))\n")
rep("HF_REPO = None\n", "HF_REPO = None\nHF_EVERY_H = 6.0\n")
rep("    global HF_REPO\n    HF_REPO = a.hf_repo or None\n", "    global HF_REPO, HF_EVERY_H\n    HF_REPO = a.hf_repo or None\n    HF_EVERY_H = a.hf_every_h\n")
open(f,'w').write(s); print('patched', f)
