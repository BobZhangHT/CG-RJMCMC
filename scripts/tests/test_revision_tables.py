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


@pytest.fixture
def residual_fixture(tmp_path):
    """Public runner schema, with deliberately synthetic scientific values."""
    import json
    exact=tmp_path/'corrected_final_exact';student=tmp_path/'formal_student'
    exact.mkdir();student.mkdir()
    def put(path,value):path.write_text(json.dumps(value),encoding='utf8')
    protocol=tmp_path/'current_protocol.json'
    put(protocol,dict(Student_computation=dict(documented_diagnostic_extension=dict(iterations_per_chain=60000,same_model_data_seeds=True))))
    for group in ('High','Low'):
        for mode,directory in [('exact',exact),('student',student)]:
            put(directory/f'run_manifest_{mode}_{group}.json',dict(mode=mode,group=group,design_protocol_sha256='frozen-identity',source_sha256={'runner':'historical'}))
        for rho in (0.,.2,.4):
            put(exact/f'{group.lower()}_ar{rho:.1f}_summary.json',dict(mean=[1.,2.,3.+rho],q025=[0.,1.,2.+rho],q975=[2.,3.,4.+rho],enumerated_partitions=27931))
        for df,rho in ((4,0.),(8,0.),(4,.2)):
            diagnostics={name:dict(rhat=1.001,ess_bulk=2000.,ess_tail=2000.,constant_chain=False) for name in t.RESIDUAL_MONITORS}
            diagnostics['phase_contrast']['quantile_mcse']=[.001,.001]
            diagnostics['PIP32']=dict(rhat=None,ess_bulk=None,ess_tail=None,constant_chain=True)
            put(student/f'{group.lower()}_t{df}_rho{rho:.1f}_summary.json',dict(df=df,rho=rho,iterations=15000,warmup=2000,retained_per_chain=13000,needs_escalation=False,diagnostics=diagnostics,phase_contrast=dict(mean=.1,q025=-.2,q975=.3)))
    return tmp_path,protocol


def test_residual_panel_all_conditions_and_provenance(residual_fixture):
    root,protocol=residual_fixture
    rows,provenance=t.residual_panel(root,protocol)
    assert len(rows)==12 and len({(r['model'],r['group']) for r in rows})==12
    assert rows[0]['mean']==3.
    assert len(provenance['selected'])==12
    assert all(r['design_protocol_sha256']=='frozen-identity' for r in provenance['historical_runs'])
    assert all('PIP32' in r['constant_monitors'] for r in provenance['selected'] if r['iterations'])
    assert '0.1000 [-0.2000, 0.3000]' in t.residual_table_tex(rows)
    split,_=t.residual_panel([root/'corrected_final_exact',root/'formal_student'],protocol)
    assert split==rows


def test_residual_selects_extension_alone_and_never_falls_back(residual_fixture):
    root,protocol=residual_fixture
    source=root/'formal_student/high_t4_rho0.0_summary.json'
    summary=t.read(source);summary.update(iterations=60000,retained_per_chain=58000)
    summary['phase_contrast']['mean']=.25
    extension=source.with_name('high_t4_rho0.0_extended60000_summary.json')
    t.save(extension,summary)
    rows,provenance=t.residual_panel(root,protocol)
    assert rows[2]['mean']==.25
    assert provenance['selected'][2]['iterations']==60000
    summary['diagnostics']['PIP12']['ess_bulk']=1499.
    t.save(extension,summary)
    with pytest.raises(ValueError,match='Unaccepted residual diagnostics'):t.residual_panel(root,protocol)


def test_residual_documented_layout_accepts_group_both_manifests(residual_fixture):
    root,protocol=residual_fixture
    expected,_=t.residual_panel(root,protocol)
    for old,new,mode in [('corrected_final_exact','exact','exact'),('formal_student','student','student')]:
        directory=root/old
        manifest=t.read(directory/f'run_manifest_{mode}_High.json')
        manifest['group']='both'
        t.save(directory/f'run_manifest_{mode}_both.json',manifest)
        for group in ('High','Low'):(directory/f'run_manifest_{mode}_{group}.json').unlink()
        directory.rename(root/new)
    rows,provenance=t.residual_panel(root,protocol)
    assert rows==expected and len(rows)==12
    assert len(provenance['historical_runs'])==2
    assert all(r['group']=='both' and r['applies_to_groups']==['High','Low'] for r in provenance['historical_runs'])
    assert all(r['design_protocol_sha256']=='frozen-identity' for r in provenance['historical_runs'])
    manifest_path=root/'exact/run_manifest_exact_both.json'
    manifest=t.read(manifest_path);manifest['group']='High';t.save(manifest_path,manifest)
    with pytest.raises(ValueError,match='manifest identity differs'):t.residual_panel(root,protocol)


@pytest.mark.parametrize('fault',['missing','nonfinite','duplicate','diagnostics','mcse','identity','extension'])
def test_residual_rejects_incomplete_or_unaccepted_inputs(residual_fixture,fault):
    root,protocol=residual_fixture
    path=root/'formal_student/high_t4_rho0.0_summary.json';summary=t.read(path)
    if fault=='missing':path.unlink()
    elif fault=='duplicate':t.save(path.with_name('high_t4_rho0.0_duplicate_summary.json'),summary)
    else:
        if fault=='nonfinite':summary['phase_contrast']['mean']=None
        elif fault=='diagnostics':summary['diagnostics']['delta16']['rhat']=None
        elif fault=='mcse':summary['diagnostics']['phase_contrast']['quantile_mcse']=[.003,.001]
        elif fault=='identity':summary['df']=8
        elif fault=='extension':
            summary.update(iterations=60000,retained_per_chain=58000)
            t.save(protocol,{})
        t.save(path,summary)
    with pytest.raises((ValueError,TypeError)):t.residual_panel(root,protocol)


def test_combined_table5_preserves_panel_a(tmp_path,residual_fixture):
    root,protocol=residual_fixture;residual,_=t.residual_panel(root,protocol)
    rows=[dict(cohort='primary',cell='D5',method=m,correct_K=1,trajectory_RMSE=.1,test_RMSE=.2,Hausdorff=0,segment_level_RMSE=.1) for m in ('B0','B1','B2','PELT-BIC','A3','Constrained DP-BIC')]
    agg=t.aggregate_methods(rows)
    app=[dict(group='High',method='CG-RJMCMC',test_RMSE=.15,K=0,cuts=[],runtime_seconds=2.)]
    t.manuscript_tables(tmp_path,agg,[],[],app)
    original=(tmp_path/'table5.tex').read_text()
    t.manuscript_tables(tmp_path,agg,[],[],app,residual)
    assert (tmp_path/'table5_panel_a.tex').read_text()==original
    assert original in (tmp_path/'table5.tex').read_text()
    assert 'Panel B' in (tmp_path/'table5.tex').read_text()
