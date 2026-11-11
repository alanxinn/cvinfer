"""Run two-instance matting parity and cached-image C# prediction after wevisone_validate.py."""
import subprocess,sys,json,os
from pathlib import Path
import numpy as np
repo=Path(__file__).resolve().parents[1];root=repo/'cache/wevisone/validation/extra';root.mkdir(parents=True,exist_ok=True)
image=repo/'cache/wevisone/validation/real/inputs/livingroom.png';model=repo/'cache/wevisone/2B/engines'
cs=repo/'csharp/alanxinn.CvinferDemo/bin/Release/net8.0-windows/alanxinn.CvinferDemo.dll'
base=[sys.executable,str(repo/'scripts/wevisone_infer.py'),'--model',str(model),'--image',str(image),'--box-matting','--box','60','300','195','380','--box','395','335','540','425']
csharp=['dotnet',str(cs),'--type','wevisone','--model',str(model),'--img',str(image)]
cases=[('multi_native',base),('multi_csharp',csharp+['--box-matting','--boxes','60,300,195,380;395,335,540,425']),('cached_csharp',csharp+['--vl-task','detection','--query','sofa','--warmup','1','--iterations','3','--timing'])]
report={'cases':[]}
for name,cmd in cases:
 dest=root/name;dest.mkdir(exist_ok=True)
 run=subprocess.run(cmd+['--output',str(dest)],capture_output=True,timeout=240)
 (dest/'stdout.log').write_bytes(run.stdout);(dest/'stderr.log').write_bytes(run.stderr)
 value={'name':name}
 try:
  assert run.returncode==0,run.stderr.decode('utf-8',errors='replace')
  data=json.loads((dest/'result.json').read_text(encoding='utf-8-sig'))
  assert not data.get('Truncated',data.get('truncated',True))
  if name.startswith('multi'):
   assert len(data.get('Masks',data.get('masks',[])))==2
   alpha=[np.fromfile(dest/f'alpha_{i:03d}.f32',dtype='<f4') for i in range(2)]
   for a in alpha:assert a.size==640*640 and np.isfinite(a).all() and a.max()>.5 and a.min()<.5
   assert np.max(np.abs(alpha[0]-alpha[1]))>.5
   if name=='multi_csharp':
    for i,a in enumerate(alpha):np.testing.assert_allclose(a,np.fromfile(root/'multi_native'/f'alpha_{i:03d}.f32',dtype='<f4'),atol=1e-6,rtol=0)
  else:
   expected=json.loads((repo/'cache/wevisone/validation/real/detection/native/result.json').read_text())
   assert data['Response']==expected['response'] and len(data['LatencySamplesMs'])==3
  value['status']='PASS'
 except Exception as e:value.update(status='FAIL',error=str(e))
 report['cases'].append(value);(root/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(name,value,flush=True)
report['status']='PASS' if all(c['status']=='PASS' for c in report['cases']) else 'FAIL'
(root/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
sys.exit(0 if report['status']=='PASS' else 1)
