import argparse
import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import guard
import bootstrap


class GuardTests(unittest.TestCase):
    def args(self,directory,command,**changes):
        values=dict(state=str(directory),id='test',name='test task',cwd=str(directory),
                    config=str(directory/'missing.json'),no_notify=True,shutdown=False,
                    grace=0,timeout=0,success_file=None,command=command)
        values.update(changes)
        return argparse.Namespace(**values)

    def state(self,directory):
        return json.loads((directory/'test/status.json').read_text(encoding='utf-8'))

    def test_task_success_preserves_log_and_no_shutdown(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(guard,'perform_shutdown') as shutdown:
            root=Path(tmp)
            result=guard.run(self.args(root,[sys.executable,'-c','print("trained")']))
            self.assertEqual(result,0)
            self.assertEqual(self.state(root)['state'],'succeeded')
            self.assertIn('trained',(root/'test/task.log').read_text())
            shutdown.assert_not_called()

    def test_task_failure_keeps_original_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.assertEqual(guard.run(self.args(root,[sys.executable,'-c','raise SystemExit(7)'])),1)
            self.assertEqual(self.state(root)['exit_code'],7)
            self.assertEqual(self.state(root)['state'],'failed')

    def test_timeout_terminates_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            guard.run(self.args(root,[sys.executable,'-c','import time;time.sleep(20)'],timeout=.1))
            self.assertEqual(self.state(root)['state'],'timed_out')

    def test_missing_output_does_not_report_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            guard.run(self.args(root,[sys.executable,'-c','pass'],success_file=str(root/'metrics.json')))
            self.assertEqual(self.state(root)['state'],'failed')

    def test_existing_task_id_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            args=self.args(root,[sys.executable,'-c','pass'])
            guard.run(args)
            before=(root/'test/status.json').read_bytes()
            self.assertEqual(guard.run(args),2)
            self.assertEqual(before,(root/'test/status.json').read_bytes())

    def test_serverchan_http_payload(self):
        response=Mock()
        response.__enter__=Mock(return_value=response)
        response.__exit__=Mock(return_value=False)
        response.read.return_value=b'{"code":0}'
        opener=Mock();opener.open.return_value=response
        with patch.object(guard.urllib.request,'build_opener',return_value=opener):
            self.assertTrue(guard.notify('task ended',{'provider':'serverchan','sendkey':'SCTtest123'})['delivered'])
        request=opener.open.call_args.args[0]
        self.assertEqual(request.full_url,'https://sctapi.ftqq.com/SCTtest123.send')
        fields=guard.urllib.parse.parse_qs(request.data.decode())
        self.assertEqual(fields['desp'],['task ended'])

    def test_failed_notification_is_bounded_and_redacted(self):
        opener=Mock();opener.open.side_effect=OSError('secret SCTdoNotLog')
        with patch.object(guard.urllib.request,'build_opener',return_value=opener),patch.object(guard.time,'sleep'):
            result=guard.notify('task ended',{'provider':'serverchan','sendkey':'SCTtest123'})
        self.assertFalse(result['delivered']);self.assertEqual(opener.open.call_count,3)
        self.assertNotIn('SCT',json.dumps(result))

    def test_feishu_signature_matches_official_algorithm(self):
        result=guard.payload('test',{'secret':'example'},123)
        expected=base64.b64encode(hmac.new(b'123\nexample',b'',hashlib.sha256).digest()).decode()
        self.assertEqual(result['sign'],expected)

    def test_malformed_sendkey_cannot_change_host(self):
        with self.assertRaises(ValueError):
            guard.notify('x',{'provider':'serverchan','sendkey':'SCT/../../evil'})

    @unittest.skipUnless(os.name=='posix','POSIX file-lock behavior')
    def test_duplicate_task_rejected_by_instance_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with guard.instance_lock(root):
                self.assertEqual(guard.run(self.args(root,[sys.executable,'-c','pass'])),2)
            self.assertEqual(self.state(root)['state'],'launch_failed')

    def test_failed_notification_does_not_prevent_opt_in_shutdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(guard.sys,'platform','linux'),patch.object(guard.os,'access',return_value=True),\
                 patch.object(guard.shutil,'which',return_value=None),patch.dict(os.environ,{},clear=True),\
                 patch.object(guard,'load_config',return_value={'provider':'serverchan','sendkey':'SCTtest123'}),\
                 patch.object(guard,'notify',return_value={'delivered':False}) as notification,\
                 patch.object(guard,'perform_shutdown',return_value=0) as shutdown:
                guard.run(self.args(root,[sys.executable,'-c','raise SystemExit(3)'],shutdown=True,no_notify=False))
            shutdown.assert_called_once()
            notification.assert_called_once()
            self.assertFalse(self.state(root)['notification']['delivered'])
            self.assertEqual(self.state(root)['shutdown']['state'],'request_returned')
            self.assertFalse(self.state(root)['shutdown']['confirmed_poweroff'])

    def test_cancel_marker_prevents_shutdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'test').mkdir();(root/'test/cancel_shutdown').touch()
            with patch.object(guard.sys,'platform','linux'),patch.object(guard.os,'access',return_value=True),\
                 patch.object(guard.shutil,'which',return_value=None),patch.dict(os.environ,{},clear=True),\
                 patch.object(guard,'perform_shutdown') as shutdown:
                guard.run(self.args(root,[sys.executable,'-c','pass'],shutdown=True))
            shutdown.assert_not_called()
            self.assertEqual(self.state(root)['shutdown']['state'],'cancelled')

    def test_slurm_refuses_instance_shutdown(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'SLURM_JOB_ID':'123'}),\
             patch.object(guard.sys,'platform','linux'),patch.object(guard.os,'access',return_value=True),\
             patch.object(guard,'perform_shutdown') as shutdown:
            self.assertEqual(guard.run(self.args(Path(tmp),[sys.executable,'-c','pass'],shutdown=True)),2)
            shutdown.assert_not_called()

    def test_detached_start_runs_to_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            args=self.args(root,[sys.executable,'-c','print("detached")'])
            # start() expects its run directory not to exist yet.
            guard.start(args)
            import time
            for _ in range(100):
                if self.state(root)['state'] in guard.TERMINAL and 'notification' in self.state(root):break
                time.sleep(.05)
            self.assertEqual(self.state(root)['state'],'succeeded',self.state(root))
            self.assertIn('detached',(root/'test/task.log').read_text())


class BootstrapTests(unittest.TestCase):
    HTML='<a href="cu118/">cu118</a><a href="/whl/cu124/">cu124</a><a href="cu128/">cu128</a><a href="cu130/">cu130</a><a href="nightly/cu134/">nightly</a>'

    def test_driver_ceiling_excludes_newer_cuda(self):
        info={'gpu':{'driver_cuda_max':[12,4],'compute_capabilities':['8.0']}}
        self.assertEqual(bootstrap.choose_channels(self.HTML,info,'cuda'),['cu124','cu118'])

    def test_blackwell_requires_128(self):
        info={'gpu':{'driver_cuda_max':[13,0],'compute_capabilities':['12.0']}}
        self.assertEqual(bootstrap.choose_channels(self.HTML,info,'cuda'),['cu130','cu128'])

    def test_volta_excludes_cuda_13(self):
        info={'gpu':{'driver_cuda_max':[13,0],'compute_capabilities':['7.0']}}
        self.assertNotIn('cu130',bootstrap.choose_channels(self.HTML,info,'cuda'))

    def test_missing_gpu_never_silently_installs_cpu(self):
        with self.assertRaises(ValueError):bootstrap.choose_channels(self.HTML,{'gpu':None},'cuda')
        self.assertEqual(bootstrap.choose_channels('',{'gpu':None},'cpu'),['cpu'])

    def test_stable_version_and_numeric_order(self):
        self.assertEqual(bootstrap.stable_version('torch (2.10.0+cu130)\nAvailable versions: 2.10.0'), '2.10.0+cu130')
        self.assertIsNone(bootstrap.stable_version('torch (2.10.0rc1)'))
        self.assertGreater(bootstrap.version_key('2.10.0'),bootstrap.version_key('2.9.1'))

    def test_smoke_code_is_valid_python(self):
        for device in ['cpu','cuda']:
            compile(bootstrap.smoke_source(device,['vision','llm']),'smoke.py','exec')

    def test_planning_does_not_create_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=subprocess.CompletedProcess([],0,'torch (2.5.1+cpu)\n','')
            with patch.object(bootstrap,'system_info',return_value={'gpu':None,'slurm_available':False}),\
                 patch.object(bootstrap,'capture',return_value=result),\
                 patch.object(bootstrap.subprocess,'run') as install:
                self.assertEqual(bootstrap.main(['--fresh','--device','cpu','--destination',tmp]),0)
            install.assert_not_called()
            self.assertEqual(list(Path(tmp).iterdir()),[])

    def test_reusable_environment_skips_network_and_installation(self):
        existing={'status':'reuse','versions':{'torch':'2.5.1'},'missing':[]}
        with patch.object(bootstrap,'system_info',return_value={'gpu':None,'slurm_available':False}),\
             patch.object(bootstrap,'inspect_current',return_value=existing),\
             patch.object(bootstrap.urllib.request,'urlopen') as network,\
             patch.object(bootstrap.subprocess,'run') as install:
            self.assertEqual(bootstrap.main(['--device','cpu']),0)
        network.assert_not_called();install.assert_not_called()

    def test_apply_reuse_writes_record_without_installing(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=argparse.Namespace(apply=True,destination=tmp)
            existing={'status':'reuse','versions':{'torch':'2.5.1'},'missing':[]}
            with patch.object(bootstrap,'capture',return_value=subprocess.CompletedProcess([],0,'torch==2.5.1\n')),\
                 patch.object(bootstrap.subprocess,'run') as install:
                self.assertTrue(bootstrap.reuse_or_supplement(args,existing,'cpu',['vision']))
            install.assert_not_called()
            result=json.loads((Path(tmp)/'latest.json').read_text())
            self.assertEqual(result['mode'],'reuse');self.assertEqual(result['installed_packages'],[])

    def test_missing_packages_plan_does_not_install(self):
        existing={'status':'supplement','missing':['timm'],'versions':{'torch':'2.5.1'}}
        args=argparse.Namespace(apply=False)
        with patch.object(bootstrap,'supplement_existing') as install:
            self.assertTrue(bootstrap.reuse_or_supplement(args,existing,'cuda',['vision']))
        install.assert_not_called()

    def test_failed_overlay_does_not_publish_ready_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=argparse.Namespace(apply=True,destination=tmp)
            existing={'status':'supplement','missing':['timm'],'versions':{'torch':'2.5.1'}}
            with patch.object(bootstrap,'supplement_existing',side_effect=RuntimeError('incompatible')):
                self.assertFalse(bootstrap.reuse_or_supplement(args,existing,'cpu',['vision']))
            self.assertFalse((Path(tmp)/'latest.json').exists())

    def test_core_runtime_failure_is_not_reused(self):
        inventory={'versions':{'torch':'2.5.1'},'missing':['timm'],'source_sites':[]}
        with patch.object(bootstrap,'capture',side_effect=[subprocess.CompletedProcess([],0,json.dumps(inventory)),
                                                           subprocess.CompletedProcess([],1,'CUDA failure')]):
            result=bootstrap.inspect_current('cuda',['timm'],['vision'])
        self.assertEqual(result['status'],'unusable')
        self.assertEqual(result['reason'],'existing_torch_runtime_failed')

    def test_healthy_torch_with_missing_extras_requests_supplement(self):
        inventory={'versions':{'torch':'2.5.1'},'missing':['timm'],'source_sites':[]}
        with patch.object(bootstrap,'capture',side_effect=[subprocess.CompletedProcess([],0,json.dumps(inventory)),
                    subprocess.CompletedProcess([],0,json.dumps({'torch':'2.5.1','cuda_runtime':'12.4'}))]):
            result=bootstrap.inspect_current('cuda',['timm'],['vision'])
        self.assertEqual(result['status'],'supplement')
        self.assertEqual(result['missing'],['timm'])

    def test_fresh_explicitly_bypasses_existing_probe(self):
        with patch.object(bootstrap,'system_info',return_value={'gpu':None,'slurm_available':False}),\
             patch.object(bootstrap,'inspect_current') as existing,\
             patch.object(bootstrap,'capture',return_value=subprocess.CompletedProcess([],0,'torch (2.5.1+cpu)\n')):
            self.assertEqual(bootstrap.main(['--fresh','--device','cpu']),0)
        existing.assert_not_called()

    def test_overlay_installs_only_missing_extra_and_pins_existing_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);site=root/'site';site.mkdir();base=root/'base';base.mkdir()
            existing={'source_sites':[str(base)],'missing':['timm'],
                      'versions':{'torch':'2.5.1','torchvision':'0.20.1','torchaudio':'2.5.1'}}
            responses=[subprocess.CompletedProcess([],0,str(site)),
                       subprocess.CompletedProcess([],0,'No broken requirements found.'),
                       subprocess.CompletedProcess([],0,'{"smoke_passed":true}'),
                       subprocess.CompletedProcess([],0,json.dumps(existing['versions'])),
                       subprocess.CompletedProcess([],0,'torch==2.5.1\ntimm==1.0.0\n')]
            directory=root/'overlay'
            def simulated_run(command,**kwargs):
                if command[1:3]==['-m','venv']:directory.mkdir()
                return subprocess.CompletedProcess(command,0)
            with patch.object(bootstrap,'capture',side_effect=responses),\
                 patch.object(bootstrap.subprocess,'run',side_effect=simulated_run) as commands:
                result=bootstrap.supplement_existing(directory,existing,'cpu',['vision'])
            installs=[call.args[0] for call in commands.call_args_list if 'install' in call.args[0]]
            self.assertEqual(len(installs),2) # pip tooling plus the missing extra
            self.assertEqual(installs[-1][-1],'timm')
            self.assertFalse(any('torch' in command or 'torchvision' in command for command in installs))
            self.assertIn('torch==2.5.1',(directory/'base-torch-constraints.txt').read_text())
            self.assertIn(str(base),(site/'task_guard_base.pth').read_text(encoding='utf-8'))
            self.assertEqual(result['mode'],'supplement')

    def test_dependency_conflict_blocks_ready_even_when_imports_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(bootstrap,'capture',return_value=subprocess.CompletedProcess([],1,
                            'httpcore 1.0.7 requires h11<0.15,>=0.13, but you have h11 0.16.0.')):
                with self.assertRaises(bootstrap.DependencyConflict):
                    bootstrap.validate_dependencies('python',root)
            self.assertIn('httpcore',(root/'pip-check.log').read_text())

    def test_overlay_dependency_conflict_does_not_fall_back_to_torch_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=argparse.Namespace(apply=True,destination=tmp)
            existing={'status':'supplement','missing':['timm'],'versions':{'torch':'2.5.1'}}
            with patch.object(bootstrap,'supplement_existing',side_effect=bootstrap.DependencyConflict('h11 conflict')):
                with self.assertRaises(bootstrap.DependencyConflict):
                    bootstrap.reuse_or_supplement(args,existing,'cpu',['vision'])
            self.assertFalse((Path(tmp)/'latest.json').exists())

    def test_existing_dependency_conflict_stops_apply_without_network(self):
        with patch.object(bootstrap,'system_info',return_value={'slurm_available':False}),\
             patch.object(bootstrap,'inspect_current',return_value={'status':'dependency_conflict'}),\
             patch.object(bootstrap.urllib.request,'urlopen') as network:
            with self.assertRaises(bootstrap.DependencyConflict):
                bootstrap.main(['--apply','--device','cpu'])
        network.assert_not_called()


if __name__=='__main__':unittest.main()
