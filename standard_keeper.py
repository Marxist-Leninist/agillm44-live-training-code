#!/usr/bin/env python3
"""Canonical Standard 1.1B exact-resume keeper for two-model GB10 training.
Never signals RPV16, never falls back to an older checkpoint, and never prints
private launcher environment. Retains the established Standard save directory,
which is independent of RPV16 and already covered by its checkpoint uploader.
"""
from __future__ import annotations
import argparse, datetime, fcntl, hashlib, json, math, os, pathlib, py_compile
import signal, shutil, subprocess, sys, time
from typing import Any
P=pathlib.Path
ROOT=P('/workspace'); HOME=ROOT/'dual_training_20260920'
SAVE=ROOT/'gb10_nvfp4_openai_20260915/live_canary'
BASE_LAUNCH=ROOT/'gb10_memory_innovation_20260915/retention4/active_launch.private.json'
SOURCE=ROOT/'gb10_memory_innovation_20260915/retention4/agillm44_retention4.py'
CONTRACT=ROOT/'GB10_DUAL_TRAINING_OWNER_CONTRACT_20260920.json'
STATUS=HOME/'standard_keeper_state.json'; LOG=HOME/'standard_trainer.log'
MIN_SEEN=306866100224; MIN_STEP=2915022


def now()->str:return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read_json(p:P)->dict[str,Any]:return json.loads(p.read_text())
def atomic(p:P,data:dict[str,Any],mode:int=0o600)->None:
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.chmod(mode);os.replace(tmp,p)
def digest(p:P)->str:
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def processes()->list[dict[str,Any]]:
    result=[]
    for d in P('/proc').iterdir():
        if not d.name.isdigit():continue
        try:a=[x.decode(errors='replace') for x in (d/'cmdline').read_bytes().split(b'\0') if x]
        except OSError:continue
        if not a or 'python' not in P(a[0]).name.lower():continue
        # Inspect executable/script argv only, not substrings inside shell code.
        script=next((s for s in a[1:4] if s.endswith('.py')), '')
        is_train='train' in a
        low=script.lower()
        kind=None
        if is_train and any(s in low for s in ('agillm44_retention4','agillm44_hf_resume','agillm43_pretrain','agillm44_standard')):kind='standard'
        elif is_train and ('rpv16_quality_' in low or 'quality_gate' in low):kind='quality_test'
        elif is_train and 'agillm_gb10_1pf' in low and '--save-dir' in a:
            if a[a.index('--save-dir')+1]=='/workspace/agillm-gb10-1pf-targetfix-active':kind='rpv16_production'
        if kind:result.append({'pid':int(d.name),'kind':kind,'script':script})
    return result

def resources()->dict[str,Any]:
    lines=P('/proc/meminfo').read_text().splitlines()
    available=next(int(x.split()[1])*1024 for x in lines if x.startswith('MemAvailable:'))
    return {'mem_available_gib':round(available/2**30,2),'disk_free_gib':round(shutil.disk_usage(ROOT).free/2**30,2)}

def stops()->list[str]:
    return [str(p) for p in (HOME/'STOP_STANDARD', ROOT/'dualtrack_standard_1p1b/STOP_STANDARD_DUAL',ROOT/'STOP_LIVE_KEEP') if p.exists()]

def authorized()->bool:
    try:
        d=read_json(CONTRACT)
        return d.get('mode')=='side-by-side' and d.get('explicit_user_reversal') is True and d.get('instance_id')==51049010
    except (OSError,ValueError):return False

def checkpoint(verify_hashes:bool=True)->dict[str,Any]:
    p=SAVE/'training_latest.json'; m=read_json(p)
    assert m.get('role')=='training', 'Training pointer required, not serving or rescue branch'
    assert int(m['step'])>=MIN_STEP and int(m['seen_tok'])>=MIN_SEEN, 'Refusing older Standard lineage'
    ck=P(m['checkpoint_path']); assert ck.parent.resolve()==SAVE.resolve(), 'Unexpected checkpoint root'
    parts=[{'path':str(ck),'name':ck.name,'nbytes':m['checkpoint_main_nbytes'],'sha256':m['checkpoint_main_sha256']}]+m['shards']
    assert len(m['shards'])>=62, 'Incomplete checkpoint shards'
    for part in parts:
        fp=P(part['path'])
        assert fp.is_file() and fp.stat().st_size==part['nbytes'], 'Missing/wrong-size checkpoint part: '+fp.name
        if verify_hashes:assert digest(fp)==part['sha256'], 'Checkpoint SHA mismatch: '+fp.name
    tok=P(str(ck)+'.tokenizer.json');assert tok.is_file() and tok.stat().st_size>0, 'Tokenizer sidecar missing'
    assert m.get('continuation_state_schema')=='agillm43.continuation.state.v2', 'Exact continuation state required'
    assert m.get('continuation_state_sha256'), 'Missing continuation integrity marker'
    return {'checkpoint':str(ck),'step':int(m['step']),'seen_tok':int(m['seen_tok']),
            'manifest_sha256':digest(p),'checkpoint_main_sha256':m['checkpoint_main_sha256'],
            'continuation_state_sha256':m['continuation_state_sha256'],'parts':len(parts),
            'package_bytes':sum(x['nbytes'] for x in parts),'tokenizer_sha256':digest(tok)}

def prepare(verify_hashes:bool=True)->tuple[list[str],dict[str,str],str,dict[str,Any]]:
    assert authorized(), 'Current explicit dual-training contract missing'
    ck=checkpoint(verify_hashes)
    d=read_json(BASE_LAUNCH);argv=list(d['argv']);env=os.environ.copy()
    env.update({str(k):str(v) for k,v in d.get('env',{}).items()})
    assert str(SOURCE) in argv and 'train' in argv, 'Unexpected Standard source'
    assert '--require_exact_resume_state' in argv, 'Exact resume is mandatory'
    assert argv[argv.index('--save_dir')+1]==str(SAVE), 'Do not merge RPV16 and Standard checkpoints'
    assert int(argv[argv.index('--target_tokens')+1])==400000000000
    for flag in ('--dblock_ar_prob','--dblock_sat_prob','--dblock_nat_prob'):
        assert float(argv[argv.index(flag)+1])>0, 'All three objective families must remain active'
    for flag in ('--lr_schedule_reset_on_resume','--lr_schedule_reanchor_on_resume'):
        assert flag not in argv, 'Do not reset existing learning-rate history'
    argv[argv.index('--resume')+1]=ck['checkpoint']
    def setarg(flag:str,val:str)->None:
        if flag in argv:
            argv[argv.index(flag)+1]=str(val)
        else:
            argv.extend([flag,str(val)])
    # Side-by-side GB10 co-tenancy: Standard keeps moving without starving AGILLMB10/RPV16.
    # Original active launch used batch 224 when Standard owned the box; dual-track starts conservative.
    dual_batch=os.environ.get('AGILLM_STANDARD_DUAL_BATCH_SIZE','4')
    setarg('--batch_size', dual_batch)
    setarg('--disk_free_floor_gb', os.environ.get('AGILLM_STANDARD_DUAL_DISK_FLOOR_GB','10.0'))
    setarg('--max_ckpts', os.environ.get('AGILLM_STANDARD_DUAL_MAX_CKPTS','2'))
    setarg('--save_every_sec', os.environ.get('AGILLM_STANDARD_DUAL_SAVE_EVERY_SEC','1800'))
    env['AGILLM43_BATCH_SIZE']=dual_batch
    env['AGILLM_ALLOW_RETENTION4_DURING_1PF']='1'
    env['AGILLM43_SAT_VARIABLE_STRIDE2_THRESHOLD']='0.15'
    env['AGILLM_RUN_ID']='standard1p1b-dual-canonical-20260920'
    # CPU feeder concurrency only; model/batch/objective/optimizer settings unchanged.
    env['OMP_NUM_THREADS']='6';env['MKL_NUM_THREADS']='6'
    py_compile.compile(str(SOURCE),doraise=True)
    cwd=d.get('cwd') or '/workspace/gb10_hf_training_20260914'
    assert P(cwd).is_dir()
    receipt={'schema':'agillm.standard.dual-exact-launch.v1','prepared_utc':now(),**ck,
             'trainer_source':str(SOURCE),'trainer_source_sha256':digest(SOURCE),
             'save_dir':str(SAVE),'cwd':cwd,'argv':argv,
             'changed_from_original':['resume -> newest canonical training pointer','SAT-variable threshold -> owner0.15','dedicated run id','CPU feeder threads ->6','dual co-tenant batch_size override'],
             'target_tokens':400000000000,'reset_weights':False,'reset_optimizer':False,
             'reset_data_cursor':False,'reset_schedule':False,'sat_variable_threshold':0.15,
             'exact_resume_enforced_by_trainer':True,'launched':False}
    atomic(HOME/'standard_launch_receipt.json',receipt)
    return argv,env,cwd,receipt


def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');ap.add_argument('--status',action='store_true')
    ap.add_argument('--min-mem-gib',type=float,default=30);ap.add_argument('--min-disk-gib',type=float,default=30)
    args=ap.parse_args(); HOME.mkdir(exist_ok=True)
    if args.status:
        print(json.dumps({'status':read_json(STATUS) if STATUS.exists() else None,'resources':resources(),'processes':processes(),'stops':stops()},indent=2));return 0
    if args.check:
        _,_,_,r=prepare();print(json.dumps({k:v for k,v in r.items() if k!='argv'},indent=2));return 0
    locks=[]
    try:
        for p in (HOME/'standard-training.lock',ROOT/'dualtrack_standard_1p1b/control/start.lock'):
            p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a+')
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    except BlockingIOError:
        print('Another canonical Standard launcher/keeper owns the duplicate-writer lock.',flush=True);return 0
    child:subprocess.Popen|None=None; stopping=False; current_receipt={}; retry_at=0.0; failures=0; last_print=''
    def write(state:str,**extra:Any)->None:
        nonlocal last_print
        data={'schema':'agillm.standard.dual-keeper.v1','utc':now(),'keeper_pid':os.getpid(),
              'state':state,'trainer_pid':child.pid if child and child.poll() is None else None,
              'save_dir':str(SAVE),'log':str(LOG),**extra}
        atomic(STATUS,data)
        concise=json.dumps({k:v for k,v in data.items() if k not in ('utc','resources')},sort_keys=True)
        if concise!=last_print:print(json.dumps(data,sort_keys=True),flush=True);last_print=concise
    def on_signal(sig,frame):
        nonlocal stopping
        stopping=True
        if child is not None and child.poll() is None:child.terminate()
    signal.signal(signal.SIGTERM,on_signal);signal.signal(signal.SIGINT,on_signal)
    while True:
        if child is not None and child.poll() is None:
            if stopping or stops():
                if not stopping:child.terminate();stopping=True
                write('saving_own_standard_on_stop',stops=stops());time.sleep(5);continue
            try:
                s=read_json(SAVE/'run_state.json')
                fresh=int(s.get('pid',0))==child.pid and int(s.get('seen_tok',0))>current_receipt['seen_tok']
                loss=s.get('loss');loss=float(loss) if isinstance(loss,(float,int)) and math.isfinite(loss) else None
                write('training_verified' if fresh else 'starting_exact_resume',
                      checkpoint_step=current_receipt['step'],checkpoint_seen_tok=current_receipt['seen_tok'],
                      step=s.get('step') if fresh else None,seen_tok=s.get('seen_tok') if fresh else None,
                      loss=loss if fresh else None,resources=resources(),rpv16=[p for p in processes() if p['kind']=='rpv16_production'])
            except (OSError,ValueError,KeyError) as e:write('starting_exact_resume',detail=type(e).__name__)
            time.sleep(10);continue
        if child is not None:
            rc=child.returncode;write('standard_exited',returncode=rc)
            child=None
            if stopping:return 0
            try:
                if int(read_json(SAVE/'training_latest.json')['seen_tok'])>=400000000000:
                    write('standard_target_complete');return 0
            except (OSError,ValueError,KeyError):pass
            failures+=1;retry_at=time.monotonic()+min(600,30*2**min(failures,5))
        if stopping:return 0
        ps=processes();res=resources();reasons=[]
        if not authorized():reasons.append('dual_contract_missing')
        if stops():reasons.append('explicit_standard_stop')
        if any(p['kind']=='standard' for p in ps):reasons.append('existing_standard_trainer_no_duplicate')
        if any(p['kind']=='quality_test' for p in ps):reasons.append('active_bounded_quality_test')
        if not any(p['kind']=='rpv16_production' for p in ps):reasons.append('await_existing_rpv16_production_restore')
        if res['mem_available_gib']<args.min_mem_gib:reasons.append('actual_memory_headroom')
        if res['disk_free_gib']<args.min_disk_gib:reasons.append('atomic_checkpoint_disk_headroom')
        if time.monotonic()<retry_at:reasons.append('restart_backoff')
        if reasons:
            write('waiting',reasons=reasons,resources=res,processes=ps);time.sleep(15);continue
        try:
            write('verifying_exact_checkpoint',resources=res)
            argv,env,cwd,current_receipt=prepare()
            # Hashing may take seconds. Recheck live state immediately before spawn.
            ps=processes();res=resources()
            if (any(p['kind'] in ('standard','quality_test') for p in ps)
                or not any(p['kind']=='rpv16_production' for p in ps)
                or res['mem_available_gib']<args.min_mem_gib or res['disk_free_gib']<args.min_disk_gib
                or stops() or not authorized()):continue
            lf=LOG.open('ab',buffering=0)
            child=subprocess.Popen(argv,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=lf,stderr=subprocess.STDOUT,
                start_new_session=True,pass_fds=tuple(f.fileno() for f in locks))
            lf.close();current_receipt.update(launched=True,launched_utc=now(),trainer_pid=child.pid)
            atomic(HOME/'standard_launch_receipt.json',current_receipt)
            write('starting_exact_resume',checkpoint_step=current_receipt['step'],checkpoint_seen_tok=current_receipt['seen_tok'])
        except Exception as e:
            failures+=1;retry_at=time.monotonic()+min(600,30*2**min(failures,5))
            write('launch_error_no_reset',error=type(e).__name__+': '+str(e)[:400])
        time.sleep(5)
if __name__=='__main__':raise SystemExit(main())
