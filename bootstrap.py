#!/usr/bin/env python3
"""Discover a compatible stable PyTorch stack in a fresh isolated environment.

No global installation, driver changes, root requirements or hidden CPU fallback.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import urllib.request

INDEX='https://download.pytorch.org/whl/'
PYPI='https://pypi.org/simple'
PROFILES={
 'core':['numpy','scipy','pandas','scikit-learn','matplotlib','pillow','tqdm','pyyaml',
         'tensorboard','psutil','requests','ipykernel'],
 'vision':['timm','opencv-python-headless'],
 'llm':['transformers','accelerate','datasets','peft','safetensors','sentencepiece','huggingface-hub'],
}


def capture(command,timeout=30,check=False,env=None):
    return subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                          timeout=timeout,check=check,env=env)


def system_info():
    info={'os':platform.system(),'release':platform.release(),'architecture':platform.machine(),
          'libc':platform.libc_ver(),'python':platform.python_version(),'python_executable':sys.executable,
          'cpu_count':os.cpu_count(),'slurm_job_id':os.getenv('SLURM_JOB_ID'),
          'slurm_available':bool(shutil.which('sinfo')),'gpu':None}
    if hasattr(platform,'freedesktop_os_release') and sys.platform=='linux':
        try:info['distribution']=platform.freedesktop_os_release()
        except OSError:pass
    if shutil.which('nvidia-smi'):
        report=capture(['nvidia-smi'],timeout=20)
        match=re.search(r'CUDA Version:\s*(\d+)\.(\d+)',report.stdout)
        if report.returncode==0 and match:
            query=capture(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],timeout=20)
            capability=capture(['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader'],timeout=20)
            info['gpu']={'driver_cuda_max':list(map(int,match.groups())),
                         'devices':query.stdout.strip().splitlines() if query.returncode==0 else [],
                         'compute_capabilities':capability.stdout.strip().splitlines() if capability.returncode==0 else []}
    if shutil.which('nvcc'):
        info['nvcc_information_only']=capture(['nvcc','--version']).stdout.strip()
    if info['slurm_available']:
        info['slurm_version']=capture(['sinfo','--version']).stdout.strip()
    return info


def cuda_tuple(channel):
    digits=channel[2:]
    return int(digits[:-1]),int(digits[-1])


def choose_channels(html,info,device):
    if device=='cpu':return ['cpu']
    gpu=info.get('gpu')
    if not gpu:raise ValueError('No usable NVIDIA driver/GPU detected. Use --device cpu explicitly for CPU-only setup.')
    maximum=tuple(gpu['driver_cuda_max'])
    channels=set(re.findall(r'href=["\'](?:https://download\.pytorch\.org/whl/|/whl/)?(cu\d{3})/?["\']',html))
    eligible=[c for c in channels if (11,8)<=cuda_tuple(c)<=maximum]
    caps=[]
    for value in gpu.get('compute_capabilities',[]):
        if re.fullmatch(r'\d+\.\d+',value.strip()):caps.append(float(value))
    # CUDA 13 dropped pre-Turing architectures; Blackwell first supported at 12.8.
    if caps:
        if min(caps)<7.5:eligible=[c for c in eligible if cuda_tuple(c)<(13,0)]
        if max(caps)>=10.:eligible=[c for c in eligible if cuda_tuple(c)>=(12,8)]
    if not eligible:raise ValueError('No conservative official CUDA wheel channel fits this driver/GPU. No driver will be changed.')
    return sorted(eligible,key=cuda_tuple,reverse=True)


def stable_version(output):
    # pip index (without --pre) filters prereleases and incompatible Python/wheel tags.
    match=re.search(r'^torch\s+\((\d+\.\d+\.\d+(?:\+[A-Za-z0-9.]+)?)\)',output,re.M)
    return match.group(1) if match else None


def version_key(version):
    return tuple(map(int,version.split('+')[0].split('.')))


def pip_env():
    env=dict(os.environ)
    # Use documented public indexes, not a global/private pip configuration.
    env.update(PIP_CONFIG_FILE=os.devnull,PIP_INDEX_URL=PYPI,PIP_EXTRA_INDEX_URL='',
               PIP_DISABLE_PIP_VERSION_CHECK='1',PYTHONNOUSERSITE='1')
    return env


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def smoke_source(device,profiles):
    imports=['numpy','scipy','pandas','sklearn','matplotlib','PIL','tqdm','yaml','tensorboard','psutil','requests','ipykernel']
    if 'vision' in profiles:imports+=['timm','cv2']
    if 'llm' in profiles:imports+=['transformers','accelerate','datasets','peft','safetensors','sentencepiece','huggingface_hub']
    return '''import json, importlib, torch, torchvision, torchaudio
for name in %r: importlib.import_module(name)
device = %r
if device == 'cuda':
    assert torch.cuda.is_available(), 'CUDA is not available in the selected wheel'
    assert torch.cuda.device_count() > 0
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            a=torch.randn(128,128,device='cuda',requires_grad=True)
            (a@a.T).square().mean().backward()
            torch.cuda.synchronize()
net=torch.nn.Conv2d(3,8,3).to(device)
x=torch.randn(2,3,32,32,device=device,requires_grad=True)
net(x).square().mean().backward()
boxes=torch.tensor([[0.,0.,1.,1.],[0.,0.,2.,2.]],device=device)
torchvision.ops.nms(boxes,torch.tensor([.9,.8],device=device),.5)
if device=='cuda': torch.cuda.synchronize()
print(json.dumps({'torch':torch.__version__,'torchvision':torchvision.__version__,
 'torchaudio':torchaudio.__version__,'cuda_runtime':torch.version.cuda,
 'device':device,'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,
 'arch_list':torch.cuda.get_arch_list() if device=='cuda' else [],'smoke_passed':True}))
''' % (imports,device)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--apply',action='store_true',help='Create a fresh venv and actually install; otherwise inspect and plan only')
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    p.add_argument('--profile',action='append',choices=['vision','llm'],default=None)
    p.add_argument('--destination',default=str(Path.home()/'autodl-tmp/task-guard-envs'))
    p.add_argument('--max-attempts',type=int,default=2)
    args=p.parse_args(argv)
    if sys.version_info<(3,10):raise ValueError('Use Python 3.10+ to run bootstrap. Select a newer AutoDL image or HPC Python module first.')
    if args.max_attempts<1:raise ValueError('max-attempts must be positive')
    info=system_info()
    print(json.dumps({'detected':info},ensure_ascii=False,indent=2),flush=True)
    if info['slurm_available'] and not info['slurm_job_id'] and args.apply:
        raise ValueError('Slurm detected outside an allocation. Run --apply inside a compute job, not on the HPC login node.')
    device='cpu' if args.device=='cpu' else 'cuda'
    if device=='cuda':
        with urllib.request.urlopen(INDEX,timeout=20) as response:html=response.read(2_000_000).decode('utf-8')
    else:html=''
    channels=choose_channels(html,info,device)
    candidates=[]
    for channel in channels:
        result=capture([sys.executable,'-m','pip','index','versions','torch','--index-url',INDEX+channel],
                       timeout=90,env=pip_env())
        version=stable_version(result.stdout) if result.returncode==0 else None
        if version:candidates.append({'channel':channel,'torch':version})
    candidates.sort(key=lambda c:(version_key(c['torch']),cuda_tuple(c['channel']) if c['channel']!='cpu' else (0,0)),reverse=True)
    if not candidates:raise ValueError('No compatible stable torch wheel found for this Python/OS/driver; check network and Python version.')
    profiles=args.profile or ['vision']
    packages=PROFILES['core']+list(dict.fromkeys(pkg for name in profiles for pkg in PROFILES[name]))
    plan={'hardware':info,'device':device,'profiles':profiles,'packages':packages,'candidates':candidates,
          'policy':'Latest stable torch found across conservative driver-compatible official channels; runtime smoke required.',
          'drivers_modified':False,'global_environment_modified':False}
    print(json.dumps({'plan':plan,'apply':args.apply},ensure_ascii=False,indent=2),flush=True)
    if not args.apply:return 0
    destination=Path(args.destination).expanduser().resolve()
    destination.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    report=destination/('setup_'+stamp)
    report.mkdir()
    write(report/'plan.json',plan)
    failures=[]
    for number,candidate in enumerate(candidates[:args.max_attempts],1):
        directory=report/(str(number)+'_'+candidate['channel'])
        try:
            subprocess.run([sys.executable,'-m','venv',str(directory)],check=True)
            python=directory/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
            def pip(*arguments):
                subprocess.run([str(python),'-m','pip',*arguments],check=True,env=pip_env())
            pip('install','--upgrade','pip','--index-url',PYPI)
            pip('install','--only-binary=:all:','--report',str(directory/'torch-resolve.json'),
                '--index-url',INDEX+candidate['channel'],'torch=='+candidate['torch'],'torchvision','torchaudio')
            versions=capture([str(python),'-c',"import importlib.metadata as m; print('\\n'.join(p+'=='+m.version(p) for p in ['torch','torchvision','torchaudio']))"],check=True)
            constraint=directory/'torch-constraints.txt'
            constraint.write_text(versions.stdout,encoding='utf-8')
            pip('install','--only-binary=:all:','--index-url',PYPI,'--constraint',str(constraint),
                '--report',str(directory/'extras-resolve.json'),*packages)
            pip('check')
            smoke=directory/'smoke.py';smoke.write_text(smoke_source(device,profiles),encoding='utf-8')
            with (directory/'smoke.log').open('w',encoding='utf-8') as log:
                subprocess.run([str(python),str(smoke)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
            frozen=capture([str(python),'-m','pip','freeze'],check=True,env=pip_env())
            (directory/'requirements.lock.txt').write_text(frozen.stdout,encoding='utf-8')
            result={'status':'ready','python':str(python),'environment':str(directory),'candidate':candidate,
                    'previous_failures':failures,'smoke_log':str(directory/'smoke.log')}
            write(report/'result.json',result)
            write(destination/'latest.json',result)
            print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)
            print('Activate: '+str(directory/('Scripts/activate' if os.name=='nt' else 'bin/activate')),flush=True)
            return 0
        except (subprocess.SubprocessError,OSError) as error:
            failures.append({'candidate':candidate,'error_type':type(error).__name__,'directory':str(directory)})
            write(report/'failures.json',failures)
            print('Candidate failed; its files are retained for diagnosis. Trying next eligible candidate.',flush=True)
    write(report/'result.json',{'status':'failed','failures':failures})
    raise RuntimeError('No tested environment succeeded; existing environments were not changed. Inspect setup report.')


if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,RuntimeError,OSError,subprocess.SubprocessError) as error:
        print(str(error),file=sys.stderr)
        sys.exit(2)
