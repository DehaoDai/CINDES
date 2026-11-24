#!/usr/bin/env python3
"""Exercise the SCC launcher with fake qsub/qstat; never submit real jobs.

All execution takes place in a temporary copy. These checks validate submission
plumbing, not scientific performance. The report contains no simulation data.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    package = args.package.resolve()
    report = {'check': 'mocked_scc_submission', 'real_scheduler_called': False,
              'scientific_results': False, 'checks': {}}
    with tempfile.TemporaryDirectory(prefix='bounded_small_scc_test_') as temporary:
        base = Path(temporary)
        checkout = base / 'package with spaces'; checkout.mkdir()
        for name in ('study_config.py', 'submit_scc.sh', 'tree_array.qsub', 'summarize.qsub',
                     'scc_env.sh', 'run_experiment.py', 'summarize_results.py', 'make_results_bundle.py'):
            shutil.copy2(package / name, checkout / name)
        bindir = base / 'mock commands'; bindir.mkdir()
        qsub = bindir / 'qsub'
        qsub.write_text('''#!/usr/bin/env python3
import json,os,pathlib,sys
args=sys.argv[1:]
script=next((pathlib.Path(arg) for arg in args if arg.endswith('.qsub')),None)
record={'argv':args,'cwd':os.getcwd(),'script':script.read_text() if script and script.exists() else None}
with open(os.environ['MOCK_CALLS'],'a') as handle: handle.write(json.dumps(record)+'\\n')
summary=script and script.name=='summarize.qsub'
if summary and os.environ.get('MOCK_FAIL_SUMMARY')=='1':
    print('mock summary submission failure',file=sys.stderr);sys.exit(9)
if os.environ.get('MOCK_BAD_RESPONSE')=='1':
    print('unrecognized mock submission response');sys.exit(0)
print('7300002' if summary else '7300001.1-80:1')
''')
        qsub.chmod(0o755)
        qstat = bindir / 'qstat'
        qstat.write_text('''#!/usr/bin/env python3
import os,sys
mode=os.environ.get('MOCK_QSTAT','empty')
if mode=='failed': print('mock scheduler unavailable',file=sys.stderr);sys.exit(3)
if mode=='malformed': print('not XML');sys.exit(0)
print('<job_info><queue_info><job_list><JB_job_number>7300001</JB_job_number></job_list></queue_info></job_info>' if mode=='active' else '<job_info><queue_info/><job_info/></job_info>')
''')
        qstat.chmod(0o755)
        calls_path = base / 'qsub_calls.jsonl'
        env = os.environ.copy()
        for key in ('CINDES_ENV_SCRIPT', 'CINDES_CPU_THREADS', 'MOCK_BAD_RESPONSE', 'MOCK_FAIL_SUMMARY'):
            env.pop(key, None)
        env.update(PATH=str(bindir)+os.pathsep+env.get('PATH',''),
                   MOCK_CALLS=str(calls_path), SCC_PROJECT='environment_project',
                   PYTHON_BIN=sys.executable, USER='mock_scc_user')
        def calls():
            return [json.loads(line) for line in calls_path.read_text().splitlines()] if calls_path.exists() else []
        def execute(root, options=(), overrides=None):
            selected = dict(env); selected.update(overrides or {})
            return subprocess.run(['bash', str(checkout / 'submit_scc.sh'), '--root', str(root), *options],
                                  cwd=base, env=selected, text=True, capture_output=True)
        def assert_ok(done):
            assert done.returncode == 0, (done.returncode, done.stdout, done.stderr)
        def check_array(call, task_range, root):
            argv=call['argv']
            assert '-t' not in argv and '-tc' not in argv
            assert not any(arg.startswith('-t ') or arg.startswith('-tc ') for arg in argv)
            assert re.findall(r'(?m)^#\$ -t ([^\n]+)$', call['script']) == [task_range]
            assert re.findall(r'(?m)^#\$ -tc ([^\n]+)$', call['script']) == ['2']
            assert '#$ -pe omp 4\n' in call['script'] and '#$ -l gpus=1\n' in call['script']
            assert '--require-cuda' in call['script']
            script_index = next(i for i,arg in enumerate(argv) if arg.endswith('.qsub'))
            assert argv[script_index+1:] == [str(root), '-', sys.executable, '4']
            assert call['cwd'] == str(checkout)
            assert (root / 'logs').is_dir()
        def record(root):
            items=list((root/'submissions').glob('submission_*.txt'))
            assert items
            return items[-1].read_text()

        dry_root=base/'dry result with spaces'
        before=sorted(str(path.relative_to(base)) for path in base.rglob('*'))
        done=execute(dry_root,('--dry-run','--project','explicit_project'),
                     {'CINDES_OUTPUT_ROOT':str(base/'unused environment root')})
        assert_ok(done)
        after=sorted(str(path.relative_to(base)) for path in base.rglob('*'))
        assert before==after and not dry_root.exists() and not calls()
        assert str(dry_root) in done.stdout and 'explicit_project' in done.stdout
        report['checks']['dry_run_has_no_files_or_jobs_and_cli_overrides_environment']=True

        full=base/'full results with spaces'
        done=execute(full,('--project','explicit_project')); assert_ok(done)
        current=calls(); assert len(current)==2
        check_array(current[0],'1-80',full)
        assert current[0]['argv'][current[0]['argv'].index('-P')+1]=='explicit_project'
        summary=current[1]['argv']
        assert summary[summary.index('-hold_jid')+1]=='7300001'
        assert 'summarize_results.py --profile full' in current[1]['script']
        assert 'make_results_bundle.py --profile full' in current[1]['script']
        saved=record(full)
        assert 'array_job=7300001\n' in saved and 'summary_job=7300002\n' in saved
        assert 'status=submitted\n' in saved
        assert len(list((full/'submissions').glob('*.stdout')))==2
        assert len(list((full/'submissions').glob('*.stderr')))==2
        report['checks']['full80_two_jobs_with_embedded_range_cap_and_summary_dependency']=True
        report['checks']['space_containing_paths_preserved_as_single_arguments']=True
        report['checks']['commands_responses_job_ids_and_submission_scripts_saved']=True

        count=len(calls()); done=execute(full)
        assert done.returncode!=0 and len(calls())==count and 'already has a submission record' in done.stderr
        report['checks']['duplicate_submission_guard']=True
        for status,needle in [('active','remain queued or running'),('malformed','Cannot parse qstat XML'),('failed','Cannot read scheduler status')]:
            count=len(calls()); done=execute(full,('--resume',),{'MOCK_QSTAT':status})
            assert done.returncode!=0 and len(calls())==count and needle in done.stderr
        report['checks']['resume_blocks_active_or_unreadable_scheduler_state']=True
        count=len(calls()); done=execute(full,('--resume',),{'MOCK_QSTAT':'empty'}); assert_ok(done)
        assert len(calls())==count+2
        report['checks']['resume_proceeds_after_recorded_jobs_leave_scheduler']=True

        for label,options,task_range in [('pilot',('--pilot',),'1-1'),('subset',('--tasks','21-40'),'21-40'),('n2000',('--tasks','61-80'),'61-80')]:
            root=base/(label+' results'); count=len(calls())
            done=execute(root,options); assert_ok(done)
            new=calls()[count:]; assert len(new)==1; check_array(new[0],task_range,root)
            assert 'summary_job=' not in record(root)
        report['checks']['pilot_task1_and_subsets21_40_and61_80_use_one_array_job_each']=True

        locked=base/'locked results'; locked.mkdir()
        with (locked/'.submission.lock').open('w') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            count=len(calls()); done=execute(locked)
            assert done.returncode!=0 and len(calls())==count and 'in progress' in done.stderr
        report['checks']['atomic_submission_lock_blocks_concurrent_submitter']=True

        template=checkout/'tree_array.qsub'; original=template.read_text()
        template.write_text(original.replace('#$ -tc 2\n',''))
        malformed=base/'malformed template results'; count=len(calls()); done=execute(malformed)
        assert done.returncode!=0 and len(calls())==count and not malformed.exists()
        template.write_text(original)
        report['checks']['malformed_array_template_fails_before_any_job_or_output_creation']=True

        failure=base/'summary failure results'; count=len(calls())
        done=execute(failure,overrides={'MOCK_FAIL_SUMMARY':'1'})
        assert done.returncode==9 and len(calls())==count+2
        saved=record(failure)
        assert 'array_job=7300001\n' in saved and 'summary_submission_exit_code=9\n' in saved
        assert 'status=interrupted_or_failed\n' in saved and 'stage=summary\n' in saved
        report['checks']['later_submission_failure_preserves_accepted_array_job_id']=True

        ambiguous=base/'unparsed response results'; count=len(calls())
        done=execute(ambiguous,overrides={'MOCK_BAD_RESPONSE':'1'})
        assert done.returncode!=0 and len(calls())==count+1
        count=len(calls()); done=execute(ambiguous,('--resume',),{'MOCK_QSTAT':'empty'})
        assert done.returncode!=0 and len(calls())==count and 'earlier qsub response was not recognized' in done.stderr
        report['checks']['unparsed_prior_submission_response_blocks_unsafe_resume']=True
    report['passed']=True
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
