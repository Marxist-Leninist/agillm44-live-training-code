#!/bin/bash
ps -eo cmd | grep -q '^/opt/conda/bin/python terntf.py train' && { echo TRAINER_ALREADY_RUNNING; exit 3; }
cd /workspace/terntf
grep -q 'hf_every_h' terntf.py || { cp terntf.py terntf.py.fused_unpatched; /opt/conda/bin/python /workspace/patch64.py terntf.py || exit 4; }
export HF_TOKEN=$(cat /workspace/.secrets/hf_token_marxistleninist 2>/dev/null) PYTORCH_ALLOC_CONF=expandable_segments:True LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
CTX=${CTX:-2048}
setsid nohup /opt/conda/bin/python terntf.py train --preset 64b --stream 1 --data 'parquet:/workspace/fwedu/sample/10BT/*.parquet' --batch 1 --ctx $CTX --log-every ${LOGE:-20} --save-min 30 --hf-repo MarxistLeninist/AGILLM-64B-tern --run /workspace/tern64b_run --resume >> /workspace/tern64b_run.log 2>&1 < /dev/null &
echo $! > /workspace/tern64b_run.pid; echo launched $(cat /workspace/tern64b_run.pid)
