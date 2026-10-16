# muP-style width scaling of the fp AdamW lr in terntf.py (fixes the d=8192 divergence; see PROGRESS.md)
import sys
f = sys.argv[1]; s = open(f).read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a
    s = s.replace(a, b)
rep("    torch.manual_seed(a.seed)\n",
    "    if getattr(a, 'mup_base', 0) and c['d'] > a.mup_base:  # Adam moves each coord ~lr: logits shift ~lr*d\n"
    "        a.lr = a.lr * a.mup_base / c['d']\n"
    "        print(f\"width scaling: fp lr -> {a.lr:.3g} (x{a.mup_base}/{c['d']})\", flush=True)\n"
    "    torch.manual_seed(a.seed)\n")
rep("    sp.add_argument('--sign-lr'",
    "    sp.add_argument('--mup-base', type=int, default=256, help='fp lr is scaled by mup_base/d when d > mup_base (0 = off)')\n"
    "    sp.add_argument('--sign-lr'")
open(f, 'w').write(s); print('patched', f)
