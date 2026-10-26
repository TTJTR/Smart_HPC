#!/usr/bin/env python3
"""AutoDL task runner: detached execution, WeChat notifications, opt-in shutdown.

Standard library only. No shutdown or notification is performed at import time.
"""
import argparse
import base64
import contextlib
from datetime import datetime, timezone
import getpass
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.parse
import uuid

VERSION = '0.1.0'
CONFIG = Path.home()/'.config/autodl-task-guard/config.json'
STATE = Path.home()/'.local/state/autodl-task-guard'
SHUTDOWN = '/usr/bin/shutdown'
TERMINAL = {'succeeded', 'failed', 'timed_out', 'cancelled', 'launch_failed'}


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    descriptor = os.open(temporary, os.O_WRONLY|os.O_CREAT|os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def validate_webhook(url):
    # Fixed official host: never follow redirects carrying a secret webhook path.
    if not re.fullmatch(r'https://open\.feishu\.cn/open-apis/bot/v2/hook/[A-Za-z0-9-]+', url):
        raise ValueError('Expected a Feishu custom-bot HTTPS webhook on open.feishu.cn')


def load_config(path):
    path = Path(path).expanduser()
    config = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    if os.environ.get('FEISHU_WEBHOOK'):
        config['webhook'] = os.environ['FEISHU_WEBHOOK']
    if os.environ.get('FEISHU_SECRET'):
        config['secret'] = os.environ['FEISHU_SECRET']
    if os.environ.get('SERVERCHAN_SENDKEY'):
        config.update(provider='serverchan',sendkey=os.environ['SERVERCHAN_SENDKEY'])
    config.setdefault('provider','feishu' if config.get('webhook') else 'serverchan')
    if config['provider']=='serverchan':
        if not re.fullmatch(r'SCT[A-Za-z0-9]+',config.get('sendkey','')):
            raise ValueError('A ServerChan Turbo SCT SendKey is required')
    elif config['provider']=='feishu':
        validate_webhook(config.get('webhook', ''))
    else:
        raise ValueError('Unknown notification provider')
    return config


def payload(text, config, timestamp=None):
    result = {'msg_type': 'text', 'content': {'text': text}}
    if config.get('secret'):
        timestamp = str(int(time.time()) if timestamp is None else timestamp)
        key = (timestamp+'\n'+config['secret']).encode('utf-8')
        result.update(timestamp=timestamp, sign=base64.b64encode(hmac.new(key,b'',hashlib.sha256).digest()).decode('ascii'))
    return result


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def notify(text, config):
    """Bounded best effort. Never expose webhook URLs or response bodies in errors."""
    if config.get('provider')=='serverchan':
        if not re.fullmatch(r'SCT[A-Za-z0-9]+',config.get('sendkey','')):
            raise ValueError('Invalid SCT SendKey')
        url='https://sctapi.ftqq.com/'+config['sendkey']+'.send'
        mime='application/x-www-form-urlencoded'
    else:
        validate_webhook(config['webhook'])
        url=config['webhook']
        mime='application/json'
    for attempt in range(3):
        try:
            data=(urllib.parse.urlencode({'title':'AutoDL任务通知','desp':text}).encode('utf-8')
                  if config.get('provider')=='serverchan' else
                  json.dumps(payload(text,config),ensure_ascii=False).encode('utf-8'))
            request = urllib.request.Request(url,data=data,
                        headers={'Content-Type':mime},method='POST')
            with urllib.request.build_opener(NoRedirect()).open(request,timeout=8) as response:
                result = json.loads(response.read(65536))
                if result.get('code',result.get('StatusCode',-1)) != 0:
                    raise RuntimeError('Notification service rejected request')
            return {'delivered': True, 'attempts': attempt+1}
        except Exception as error:
            category = type(error).__name__
            if attempt < 2:
                time.sleep(.5*(attempt+1))
    return {'delivered':False,'attempts':3,'error_type':category}


@contextlib.contextmanager
def instance_lock(root):
    """Only one managed task per state directory; Linux flock survives SSH loss."""
    root.mkdir(parents=True,exist_ok=True)
    stream = (root/'instance.lock').open('a+b')
    try:
        if os.name == 'posix':
            import fcntl
            try:
                fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Another managed task is active; use status') from None
        else:
            import msvcrt
            stream.seek(0);stream.write(b'0');stream.flush();stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            except OSError:
                raise RuntimeError('Another managed task is active; use status') from None
        yield
    finally:
        stream.close()


def stop_child(child):
    if child.poll() is not None:
        return
    if os.name == 'posix':
        try: os.killpg(child.pid,signal.SIGTERM)
        except ProcessLookupError: return
    else:
        child.terminate()
    try:
        child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        if os.name == 'posix':
            try: os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError: pass
        else: child.kill()
        child.wait()


def perform_shutdown():
    # AutoDL documents this instance-local command. No shell interpolation.
    result = subprocess.run([SHUTDOWN],stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,timeout=15,check=False)
    return result.returncode


def run(args):
    root = Path(args.state).expanduser().resolve()
    directory = root/args.id
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'status.json').exists():
        print('Task id already exists; old evidence retained.',file=sys.stderr)
        return 2
    try:
        claim=os.open(directory/'claimed',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        os.close(claim)
    except FileExistsError:
        print('Task id is already claimed; refusing concurrent overwrite.',file=sys.stderr)
        return 2
    state = {'id':args.id,'name':args.name,'state':'starting','started_utc':utc(),
             'wrapper_pid':os.getpid(),'hostname':socket.gethostname(),
             'cwd':str(Path(args.cwd).resolve()),'shutdown_requested':args.shutdown,
             'log':str(directory/'task.log')}
    def save(**updates):
        state.update(updates,updated_utc=utc())
        write_json(directory/'status.json',state)
    try:
        config = load_config(args.config) if not args.no_notify else None
        if args.shutdown and (sys.platform != 'linux' or not os.access(SHUTDOWN,os.X_OK)):
            raise RuntimeError('AutoDL shutdown requires Linux and executable /usr/bin/shutdown')
        if args.shutdown and (os.getenv('SLURM_JOB_ID') or shutil.which('scontrol')):
            raise RuntimeError('Instance shutdown is disabled on detected Slurm/HPC systems')
        with instance_lock(root):
            # Retain the instance lock until the countdown/action has finished.
            save()
            interrupted = [False]
            def handler(signum, frame): interrupted[0] = True
            old = {}
            for sig in (signal.SIGINT,signal.SIGTERM):
                old[sig] = signal.signal(sig,handler)
            started = time.monotonic()
            child = None
            try:
                with (directory/'task.log').open('ab',buffering=0) as logfile:
                    child = subprocess.Popen(args.command,cwd=args.cwd,stdin=subprocess.DEVNULL,
                                stdout=logfile,stderr=subprocess.STDOUT,start_new_session=os.name=='posix')
                    save(state='running',child_pid=child.pid)
                    terminal = None
                    while child.poll() is None:
                        if interrupted[0]: terminal='cancelled';break
                        if args.timeout and time.monotonic()-started >= args.timeout:
                            terminal='timed_out';break
                        time.sleep(.25)
                    if terminal:
                        stop_child(child)
                    exit_code = child.wait()
                terminal = terminal or ('succeeded' if exit_code==0 else 'failed')
                missing = args.success_file and not Path(args.success_file).is_file()
                if terminal=='succeeded' and missing:
                    terminal='failed'
                save(state=terminal,exit_code=exit_code,missing_success_file=bool(missing),
                     elapsed_seconds=round(time.monotonic()-started,3),finished_utc=utc(),child_pid=None)
            except Exception as error:
                if child is not None: stop_child(child)
                save(state='launch_failed',exit_code=127,error_type=type(error).__name__,finished_utc=utc())
            finally:
                for sig,previous in old.items():signal.signal(sig,previous)
            text = (f"AutoDL任务通知\n任务：{args.name}\n任务ID：{args.id}\n"
                    f"状态：{state['state']}\n退出码：{state['exit_code']}\n"
                    f"耗时：{state.get('elapsed_seconds',0):.1f} 秒\n"
                    f"结束时间：{state['finished_utc']}\n"
                    + (f"计划：通知后等待 {args.grace} 秒，再请求实例关机。" if args.shutdown
                       else '计划：保留实例运行。'))
            delivery=notify(text,config) if config else {'delivered':False,'disabled':True}
            save(notification=delivery)
            if args.shutdown:
                save(shutdown={'state':'countdown','grace_seconds':args.grace})
                deadline=time.monotonic()+args.grace
                while time.monotonic()<deadline:
                    if (directory/'cancel_shutdown').exists():
                        save(shutdown={'state':'cancelled'})
                        return 0 if state['state']=='succeeded' else 1
                    time.sleep(min(.25,max(0,deadline-time.monotonic())))
                if (directory/'cancel_shutdown').exists():
                    save(shutdown={'state':'cancelled'})
                    return 0 if state['state']=='succeeded' else 1
                save(shutdown={'state':'attempting','utc':utc()})
                try:
                    code=perform_shutdown()
                    save(shutdown={'state':'request_returned' if code==0 else 'failed',
                                   'exit_code':code,'confirmed_poweroff':False})
                except Exception as error:
                    save(shutdown={'state':'failed','error_type':type(error).__name__})
            return 0 if state['state']=='succeeded' else 1
    except Exception as error:
        # Do not serialize exception text: requests may contain secret URLs.
        save(state='launch_failed',error_type=type(error).__name__,finished_utc=utc())
        print(f'Cannot start task ({type(error).__name__}); check config, lock and shutdown prerequisites.',file=sys.stderr)
        return 2


def start(args):
    # Fail before detaching if configuration is absent or malformed.
    if not args.no_notify: load_config(args.config)
    if args.shutdown and sys.platform!='linux':
        raise ValueError('--shutdown is supported only inside a Linux AutoDL instance')
    root=Path(args.state).expanduser().resolve()
    directory=root/args.id
    directory.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,str(Path(__file__).resolve()),'--config',str(Path(args.config).expanduser().resolve()),
             '--state',str(root),'run','--id',args.id,'--name',args.name,'--cwd',str(Path(args.cwd).resolve()),
             '--grace',str(args.grace),'--timeout',str(args.timeout)]
    if args.shutdown: command.append('--shutdown')
    if args.no_notify: command.append('--no-notify')
    if args.success_file: command.extend(['--success-file',str(Path(args.success_file).resolve())])
    command+=['--']+args.command
    with (directory/'wrapper.log').open('ab') as output:
        flags={'start_new_session':True} if os.name=='posix' else {'creationflags':subprocess.CREATE_NO_WINDOW|subprocess.DETACHED_PROCESS}
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,**flags)
    for _ in range(50):
        if (directory/'status.json').exists():
            state=json.loads((directory/'status.json').read_text(encoding='utf-8'))
            if state['state']=='launch_failed':raise RuntimeError('Task wrapper failed to start; inspect status')
            print(json.dumps({'id':args.id,'wrapper_pid':child.pid,'status':str(directory/'status.json')},ensure_ascii=False))
            return 0
        if child.poll() is not None:raise RuntimeError('Task wrapper exited before writing state')
        time.sleep(.1)
    raise RuntimeError('Task wrapper did not acknowledge startup within 5 seconds; inspect state before retrying')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',action='version',version=VERSION)
    parser.add_argument('--config',default=str(CONFIG))
    parser.add_argument('--state',default=str(STATE))
    commands=parser.add_subparsers(dest='action',required=True)
    init=commands.add_parser('init',help='Store private notification settings; hidden input')
    init.add_argument('--provider',choices=['serverchan','feishu'],default='serverchan')
    commands.add_parser('test-notify',help='Explicitly send one configuration-test notification')
    commands.add_parser('doctor',help='Check configuration and runtime without sending messages or shutting down')
    commands.add_parser('status',help='Show recorded state; not a cloud power-state query')
    cancel=commands.add_parser('cancel-shutdown')
    cancel.add_argument('id')
    for verb in ('start','run'):
        sub=commands.add_parser(verb)
        sub.add_argument('--name',required=True)
        sub.add_argument('--id',default=None)
        sub.add_argument('--cwd',default=os.getcwd())
        sub.add_argument('--shutdown',action='store_true',help='Explicitly request AutoDL instance shutdown after success OR failure')
        sub.add_argument('--no-notify',action='store_true')
        sub.add_argument('--grace',type=int,default=60)
        sub.add_argument('--timeout',type=float,default=0,help='Maximum task duration in seconds; 0 means unlimited')
        sub.add_argument('--success-file',help='Require this output file after zero exit status')
        sub.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args(argv)
    if args.action in ('start','run'):
        if args.command and args.command[0]=='--':args.command=args.command[1:]
        if not args.command:parser.error('Missing training command after --')
        if args.grace<0 or args.timeout<0 or not math.isfinite(args.timeout):parser.error('grace and timeout must be finite and nonnegative')
        args.id=args.id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]
        if not re.fullmatch('[A-Za-z0-9_-]+',args.id):parser.error('Invalid task id')
        args.cwd=str(Path(args.cwd).expanduser().resolve())
        if args.success_file:args.success_file=str((Path(args.cwd)/args.success_file).resolve())
        return start(args) if args.action=='start' else run(args)
    if args.action=='init':
        target=Path(args.config).expanduser()
        if target.exists():raise ValueError('Configuration exists; edit the existing private file instead')
        if args.provider=='serverchan':
            sendkey=getpass.getpass('ServerChan Turbo SendKey, SCT prefix (hidden): ').strip()
            if not re.fullmatch(r'SCT[A-Za-z0-9]+',sendkey):raise ValueError('Invalid SCT SendKey')
            config={'provider':'serverchan','sendkey':sendkey}
        else:
            url=getpass.getpass('Feishu webhook (hidden): ').strip()
            validate_webhook(url)
            secret=getpass.getpass('Signing secret (hidden, Enter to omit): ').strip()
            config={'provider':'feishu','webhook':url,'secret':secret}
        write_json(target,config)
        print('Private configuration saved outside the repository.')
    elif args.action=='test-notify':
        result=notify('AutoDL任务通知：连接测试。未执行关机。',load_config(args.config))
        print(json.dumps(result));return 0 if result['delivered'] else 1
    elif args.action=='doctor':
        load_config(args.config)
        print(json.dumps({'config_valid':True,'python':sys.version.split()[0],
                          'linux':sys.platform=='linux','shutdown_executable':os.access(SHUTDOWN,os.X_OK)}))
    elif args.action=='cancel-shutdown':
        if not re.fullmatch('[A-Za-z0-9_-]+',args.id):raise ValueError('Invalid task id')
        directory=Path(args.state).expanduser()/args.id
        if not (directory/'status.json').exists():raise ValueError('Unknown task')
        (directory/'cancel_shutdown').touch()
        print('Cancellation recorded; effective only before shutdown command is invoked.')
    else:
        for path in sorted(Path(args.state).expanduser().glob('*/status.json')):
            print(path.read_text(encoding='utf-8'))
    return 0


if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,RuntimeError,OSError) as error:
        print(f'Configuration or startup error ({type(error).__name__}); run doctor and inspect local state.',file=sys.stderr)
        sys.exit(2)
