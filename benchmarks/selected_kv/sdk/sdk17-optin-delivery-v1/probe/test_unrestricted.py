"""CPU-only tests of the new no-audit deployment harness, not hardware oracles."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_probe
import probe_contract


class UnrestrictedTests(unittest.TestCase):
    def test_child_environment_has_no_audit_preload_or_NUMA_override(self):
        args=SimpleNamespace(sdk_site=Path('/sdk'),mf_site=Path('/mf'))
        with patch.dict(os.environ,dict(LD_AUDIT='old.so',LD_PRELOAD='old.so',DSA_MMC_DDR_NUMA='6')):
            result=run_probe.environment(args,Path('/owned'))
        self.assertTrue(all(k not in result for k in ('LD_AUDIT','LD_PRELOAD','DSA_MMC_DDR_NUMA')))
        self.assertEqual(result['ASCEND_RT_VISIBLE_DEVICES'],'8')

    def test_real_config_writer_sets_only_explicit_optin_and_records_unknown_placement(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);config=base/'sdk/memcache_hybrid/config';config.mkdir(parents=True)
            (config/'mmc-meta.conf').write_text('')
            (config/'mmc-local.conf').write_text('ock.mmc.local_service.dram.unrestricted.enabled=false\n')
            args=SimpleNamespace(output=base/'run',sdk_site=base/'sdk',mf_site=base/'sdk',
                unrestricted=True,node=None,combined=True,timeout_seconds=120,deadline_unix=time.time()+120,audit=None)
            with patch.object(run_probe,'supervise',return_value=0),patch.object(run_probe,'ports',return_value=[10001,10002,10003]):
                self.assertEqual(run_probe.run(args),0)
            value=json.loads((args.output/'private-config.json').read_text())
            self.assertEqual(value['mode'],'unrestricted');self.assertIsNone(value['node'])
            self.assertIsNone(value['audit_sha256'])
            text=(args.output/'local.conf').read_text()
            self.assertIn('ock.mmc.local_service.dram.unrestricted.enabled=true',text)
            self.assertEqual(text.count('ock.mmc.local_service.dram.unrestricted.enabled='),1)
            self.assertIn('ock.mmc.local_service.dram.size=10GB',text)
            self.assertIn('ock.mmc.local_service.protocol=device_sdma',text)

    def test_cli_refuses_NUMA_claim_in_unrestricted_mode_before_any_SDK_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            run=subprocess.run([sys.executable,str(HERE/'run_probe.py'),'--sdk-site',str(root),
                '--mf-site',str(root),'--unrestricted','--node','0','--output',str(root/'run'),
                '--deadline-unix',str(time.time()+120)],capture_output=True,text=True)
            self.assertEqual(run.returncode,2)
            self.assertIn('cannot claim audit or a NUMA placement',run.stderr)
            self.assertFalse((root/'run').exists())

    def test_admission_in_unknown_NUMA_mode_still_requires_fresh_actual_worker_identity(self):
        ready=dict(pid=42,startticks=90,boot_id='boot',run_id='r',node=None)
        value=dict(ready,issued_unix=100,physical_devices=[8],prior_model_scope_empty_verified_by_parent=True)
        probe_contract.validate_admission(value,ready,110)
        with self.assertRaises(ValueError):
            probe_contract.validate_admission(dict(value,node=0),ready,110)

    def test_no_global_or_driver_configuration_apis(self):
        text='\n'.join((HERE/n).read_text() for n in ('run_probe.py','probe_worker.py'))
        for name in ('halMemCtl','halMemAdvise','drop_caches','compact_memory','DONTNEED','killpg'):
            self.assertNotIn(name,text)


if __name__=='__main__':unittest.main()
