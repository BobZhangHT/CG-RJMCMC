import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import revision_tables as t

def test_conditional_level_error_uses_correct_k_only():
    rows=[dict(cohort='primary',cell='D5',method='A3',correct_K=c,trajectory_RMSE=v,test_RMSE=v,Hausdorff=v,segment_level_RMSE=v) for c,v in [(1,1.),(0,100.),(1,3.)]]
    r=t.aggregate_methods(rows)[0]
    assert r['conditional_level_RMSE_n']==2 and r['conditional_level_RMSE_median']==2
    assert r['trajectory_RMSE_median']==3 and r['correct_K']==2/3

def test_runtime_requires_31_actual_repeats():
    rows=[dict(group='High',repeat=i,seconds=i/1000) for i in range(1,32)]
    r=t.timing_summary(rows)[0]
    assert r['median_seconds']==.016
    assert r['q25_seconds']==.0085 and r['q75_seconds']==.0235
    with pytest.raises(ValueError):t.timing_summary(rows[:-1])
    with pytest.raises(ValueError):t.timing_summary(rows[:-1]+[rows[0]])

def test_wilson_and_dataset_mcse():
    lo,hi=t.wilson(0,10)
    assert abs(lo)<1e-15 and 0<hi<1
    assert t.mcse([0,1])==.5

def test_table_files_derived_from_rows(tmp_path):
    t.emit(tmp_path,'table',[dict(method='CG_RJMCMC',estimate=.1234)])
    assert '.1234' in (tmp_path/'table.csv').read_text()
    assert 'CG\\_RJMCMC' in (tmp_path/'table.tex').read_text()

def test_compact_manuscript_panels(tmp_path):
    rows=[dict(cohort='primary',cell='D5',method=m,correct_K=1,trajectory_RMSE=.1,test_RMSE=.2,Hausdorff=0,segment_level_RMSE=.1) for m in ('B0','B1','B2','PELT-BIC','A3','Constrained DP-BIC')]
    agg=t.aggregate_methods(rows)
    small=t.aggregate_methods([dict(rows[-1],cohort='N35',cell='SN0')])
    coverage=[dict(cell='SN0',coverage=.95,coverage_mcse=.01,mean_width=.05,maximum_rhat=1.,minimum_bulk_ess=1000.)]
    app=[dict(group='High',method='CG-RJMCMC',test_RMSE=.15,K=0,cuts=[],runtime_seconds=2.)]
    t.manuscript_tables(tmp_path,agg,small,coverage,app)
    assert '0.10000' in (tmp_path/'table2.tex').read_text()
    assert '95.00' in (tmp_path/'table3.tex').read_text()
    assert '2.000000' in (tmp_path/'table5.tex').read_text()


def test_public_small_csv_metadata_boolean_and_numeric_regression(tmp_path):
    import csv,json
    path=tmp_path/'small_sample_results.csv'
    fields=['stage','cell','rep','data_seed','chain_seeds','method','correct_K','trajectory_RMSE','test_RMSE','Hausdorff','segment_level_RMSE','coverage','interval_width','diagnostic_pass','diagnostics','outcomes_hash']
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for rep,passed in [(1,True),(2,False)]:
            writer.writerow(dict(stage='formal',cell='SN0',rep=rep,data_seed=12345+rep,chain_seeds=json.dumps([1,2,3]),method='CG-RJMCMC',correct_K=1,trajectory_RMSE=.0123,test_RMSE=.4,Hausdorff=0,segment_level_RMSE=.01,coverage=.95,interval_width=.05,diagnostic_pass=passed,diagnostics=json.dumps([dict(split_Rhat=1.001,bulk_ESS=10000)]),outcomes_hash='fixture-'+str(rep)))
    records=t.read_public_small_csv(path)
    methods=[t.normalize_small_method(method,r['cell'],r['replicate']) for r in records for method in r['methods']]
    assert [r['diagnostic_pass'] for r in methods]==[True,False]
    assert sum(r['diagnostic_pass'] is True for r in methods)==1
    assert methods[0]['diagnostics'][0]['split_Rhat']==1.001
    assert methods[0]['data_seed']==12346 and methods[0]['chain_seeds']==[1,2,3]
    assert methods[0]['stage']=='formal' and methods[0]['rep']==1 and methods[0]['cell']=='SN0'
    assert t.aggregate_methods(methods)[0]['trajectory_RMSE_median']==.0123

@pytest.mark.parametrize('metadata', [dict(cell='SN1'),dict(rep=2),dict(replicate=2),dict(cohort='primary'),dict(diagnostic_pass='invalid')])
def test_public_small_csv_rejects_identity_or_boolean_mismatch(metadata):
    with pytest.raises(ValueError):t.normalize_small_method(dict(method='CG-RJMCMC',**metadata),'SN0',1)


def test_package_timing_executes_actual_solver_for_31_repeats(tmp_path,monkeypatch):
    import json
    import numpy as np
    from scripts import revision_tables as tables, application as app, simulation as sim
    person=np.repeat(np.arange(2),64)
    trial=np.tile(np.arange(1,65),2)
    z=(trial-32.5)/np.std(np.arange(1,65))
    c=np.repeat(np.array([-.5,.5]),64)
    y=.08*c+np.random.default_rng(197).normal(0,.1,128)
    train=(person,trial,z,c,y)
    source=tmp_path/"application.json"
    source.write_text(json.dumps({"groups":[{"data":{"first_pc":g,"participants":2}} for g in ("High","Low")]}))
    monkeypatch.setattr(app,"_source_files",lambda raw:[])
    monkeypatch.setattr(app,"prepare_group",lambda raw,g:(train,train,{"first_pc":g,"participants":2}))
    calls=[];actual=sim.pelt_fit
    def tracked(*args):
        calls.append(True)
        return actual(*args)
    monkeypatch.setattr(sim,"pelt_fit",tracked)
    out=tmp_path/"timing";out.mkdir()
    rows=tables.time_application(source,tmp_path,out)
    assert len(calls)==64 and len(rows)==62
    assert all(r["seconds"]>=0 for r in rows)
    assert {r["repeat"] for r in rows}==set(range(1,32))
    manifest=json.loads((out/"timing_manifest.json").read_text())
    assert all(p["num_threads"]==1 for p in manifest["threadpools"])
