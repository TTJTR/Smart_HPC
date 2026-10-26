#!/usr/bin/env python3
"""Reuse a working framework first; supplement or create isolated environments.

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
IMPORT_NAMES={'scikit-learn':'sklearn','pillow':'PIL','pyyaml':'yaml',
              'opencv-python-headless':'cv2','huggingface-hub':'huggingface_hub'}


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


def smoke_source(device,profiles,include_extras=True):
    imports=['numpy','scipy','pandas','sklearn','matplotlib','PIL','tqdm','yaml','tensorboard','psutil','requests','ipykernel']
    if 'vision' in profiles:imports+=['timm','cv2']
    if 'llm' in profiles:imports+=['transformers','accelerate','datasets','peft','safetensors','sentencepiece','huggingface_hub']
    if not include_extras:imports=[]
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


def inspect_current(device,packages,profiles):
    """Read installed metadata and test torch itself before contacting an index."""
    script='''import json, importlib.util, importlib.metadata as m, site
names=%r
aliases=%r
versions={}
missing=[]
for name in names:
    module=aliases.get(name,name.replace('-','_'))
    try:
        if importlib.util.find_spec(module) is None: raise ImportError(module)
        try: versions[name]=m.version(name)
        except m.PackageNotFoundError:
            if name=='opencv-python-headless': versions[name]='provided-by-cv2'
            else: raise
    except (ImportError,ValueError,m.PackageNotFoundError):missing.append(name)
print(json.dumps({'versions':versions,'missing':missing,'source_sites':site.getsitepackages()+[site.getusersitepackages()]}))
''' % (['torch','torchvision','torchaudio']+packages,IMPORT_NAMES)
    result=capture([sys.executable,'-c',script],timeout=30)
    if result.returncode:
        return {'status':'unusable','reason':'metadata_probe_failed'}
    inventory=json.loads(result.stdout.strip().splitlines()[-1])
    if 'torch' not in inventory['versions']:
        return dict(inventory,status='unusable',reason='torch_missing')
    test="""import json, torch
device=%r
if device=='cuda': assert torch.cuda.is_available(), 'Existing torch cannot use CUDA'
x=torch.randn(64,64,device=device,requires_grad=True)
(x@x.T).square().mean().backward()
net=torch.nn.Conv2d(3,4,3).to(device)
net(torch.randn(1,3,16,16,device=device)).sum().backward()
if device=='cuda': torch.cuda.synchronize()
print(json.dumps({'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'device':device}))
""" % device
    tested=capture([sys.executable,'-c',test],timeout=90)
    if tested.returncode:
        return dict(inventory,status='unusable',reason='existing_torch_runtime_failed',diagnostic=tested.stdout[-6000:])
    inventory['runtime']=json.loads(tested.stdout.strip().splitlines()[-1])
    if inventory['missing']:
        return dict(inventory,status='supplement',reason='packages_missing')
    full=capture([sys.executable,'-c',smoke_source(device,profiles)],timeout=180)
    if full.returncode:
        return dict(inventory,status='unusable',reason='existing_stack_smoke_failed',diagnostic=full.stdout[-6000:])
    return dict(inventory,status='reuse',reason='existing_stack_verified',smoke=json.loads(full.stdout.strip().splitlines()[-1]))


def supplement_existing(directory,existing,device,profiles):
    """Overlay missing libraries without writing to the supplied base environment."""
    subprocess.run([sys.executable,'-m','venv',str(directory)],check=True)
    python=directory/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    sites=capture([str(python),'-c','import sysconfig; print(sysconfig.get_path("purelib"))'],check=True)
    # A venv made from another venv does not inherit its packages automatically.
    # A .pth in the new environment reuses that interpreter's existing site dirs.
    inherited=[str(Path(path).resolve()) for path in existing['source_sites'] if Path(path).is_dir()]
    (Path(sites.stdout.strip())/'task_guard_base.pth').write_text('\n'.join(dict.fromkeys(inherited))+'\n',encoding='utf-8')
    constraint=directory/'base-torch-constraints.txt'
    constraint.write_text(''.join(name+'=='+version+'\n' for name,version in existing['versions'].items()
                                if name in ('torch','torchvision','torchaudio')),encoding='utf-8')
    def pip(*arguments):
        subprocess.run([str(python),'-m','pip',*arguments],check=True,env=pip_env())
    pip('install','--upgrade','pip','--index-url',PYPI)
    missing_core=[name for name in existing['missing'] if name in ('torchvision','torchaudio')]
    missing_extras=[name for name in existing['missing'] if name not in missing_core]
    if missing_core:
        runtime=existing['runtime']['cuda_runtime']
        channel='cu'+runtime.replace('.','') if device=='cuda' and runtime else 'cpu'
        pip('install','--only-binary=:all:','--index-url',INDEX+channel,'--constraint',str(constraint),
            '--report',str(directory/'core-resolve.json'),*missing_core)
    if missing_extras:
        pip('install','--only-binary=:all:','--index-url',PYPI,'--constraint',str(constraint),
            '--report',str(directory/'extras-resolve.json'),*missing_extras)
    full=capture([str(python),'-c',smoke_source(device,profiles)],timeout=180)
    (directory/'smoke.log').write_text(full.stdout,encoding='utf-8')
    if full.returncode:raise RuntimeError('Supplement environment failed runtime validation')
    # Ensure pip did not replace any preinstalled member of the torch stack.
    stack=capture([str(python),'-c',"import json,importlib.metadata as m; print(json.dumps({p:m.version(p) for p in ['torch','torchvision','torchaudio']}))"],check=True)
    versions=json.loads(stack.stdout.strip())
    for name,version in existing['versions'].items():
        if name in versions and versions[name]!=version:
            raise RuntimeError('Supplement changed a pinned framework version')
    frozen=capture([str(python),'-m','pip','freeze'],check=True,env=pip_env())
    (directory/'requirements.lock.txt').write_text(frozen.stdout,encoding='utf-8')
    return {'status':'ready','mode':'supplement','python':str(python),'environment':str(directory),
            'base_python':sys.executable,'installed_packages':existing['missing'],
            'smoke_log':str(directory/'smoke.log'),'base_environment_modified':False}


def reuse_or_supplement(args,existing,device,profiles):
    """Return True when reuse policy handles the request; no global mutation."""
    print(json.dumps({'existing_environment':existing,'apply':args.apply},ensure_ascii=False,indent=2),flush=True)
    if existing['status']=='unusable':return False
    if not args.apply:return True
    destination=Path(args.destination).expanduser().resolve()
    destination.mkdir(parents=True,exist_ok=True)
    report=destination/('reuse_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    report.mkdir()
    write(report/'inspection.json',existing)
    if existing['status']=='reuse':
        frozen=capture([sys.executable,'-m','pip','freeze'],check=True,env=pip_env())
        (report/'requirements.lock.txt').write_text(frozen.stdout,encoding='utf-8')
        result={'status':'ready','mode':'reuse','python':sys.executable,
                'environment':sys.prefix,'installed_packages':[],'base_environment_modified':False,
                'versions':existing['versions']}
    else:
        try:
            result=supplement_existing(report/'overlay',existing,device,profiles)
        except (subprocess.SubprocessError,OSError,RuntimeError) as error:
            write(report/'result.json',{'status':'failed','mode':'supplement','error_type':type(error).__name__})
            print('Supplement failed; base environment retained. Falling back to a fresh isolated environment.',flush=True)
            return False
    write(report/'result.json',result)
    write(destination/'latest.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return True


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--apply',action='store_true',help='Reuse verified framework, supplement missing libraries, or create a fresh venv')
    p.add_argument('--fresh',action='store_true',help='Skip reuse and resolve a new stable framework in an isolated venv')
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
    profiles=args.profile or ['vision']
    packages=PROFILES['core']+list(dict.fromkeys(pkg for name in profiles for pkg in PROFILES[name]))
    if not args.fresh:
        existing=inspect_current(device,packages,profiles)
        if reuse_or_supplement(args,existing,device,profiles):return 0
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
