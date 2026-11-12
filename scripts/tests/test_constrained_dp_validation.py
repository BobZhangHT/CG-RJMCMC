import numpy as np
import pytest
from scripts import constrained_dp_validation as dp

def test_exact_and_capped_against_centered_bruteforce():
    rng=np.random.default_rng(73)
    for n in (8,11,14):
        for _ in range(5):
            y=rng.normal(size=n);w=rng.uniform(.2,3,size=n)
            for k in (0,1,3):
                actual=dp.constrained_dp(y,w,2,k);expected=dp.brute_force(y,w,2,k)
                assert dp.centered_objective(y,w,actual,2)==pytest.approx(dp.centered_objective(y,w,expected,2),abs=1e-10)
            assert dp.partition_objective(y,w,dp.exact_dp(y,w,2),2)==pytest.approx(dp.partition_objective(y,w,dp.sim.pelt_bic(y,w,2),2),abs=1e-10)

def test_constant_tie_and_invalid_inputs():
    assert dp.constrained_dp(np.ones(12),np.ones(12),2,4)==()
    assert dp.exact_dp(np.ones(12),np.ones(12),2)==()
    for y,w in (([1,float('nan')],[1,1]),([1,2],[0,1]),([1,2],[1])):
        with pytest.raises(ValueError):dp.constrained_dp(y,w,1,1)
    with pytest.raises(ValueError):dp.partition_objective(np.zeros(8),np.ones(8),(3,),4)

def test_smoke(tmp_path):
    assert dp.main(['--smoke','--output-root',str(tmp_path)])==0
    assert (tmp_path/'manifest.json').exists()


def test_count_cap_is_distinct_from_solver_change():
    y=np.repeat([0.,10.,0.,10.,0.,10.,0.],3);w=np.ones(21)
    free=dp.exact_dp(y,w,3)
    result=dp.solve(y,w,free,3)
    assert len(free)==6
    assert not result['solver_partition_changed']
    assert result['cap_partition_changed'] and result['cap_objective_increased']
    assert len(result['constrained_dp_cuts'])<=4


def test_unified_source_discovery_and_namespace(tmp_path):
    unified=tmp_path/'unified';small=tmp_path/'small'
    paths=['replicates/main/D0/rep_000001/summary.json','replicates/main/D5/rep_000001/summary.json','replicates/sensitivity/prior_D1/rep_000001/summary.json','replicates/kmax/cap_D5/rep_000001/summary.json']
    for rel in paths:
        p=unified/rel;p.parent.mkdir(parents=True);p.write_text('{}')
    p=small/'formal/SN0/R000001.json';p.parent.mkdir(parents=True);p.write_text('{}')
    found=dp.collect_reconstruction_tasks(simulation_root=unified,small_sample_root=small)
    assert len(found)==5 and len({p for p,_ in found})==5
    cfg={'namespace':'R61-FORMAL-v1','namespace_by_dgp':{'D5':'R61-D5-BALANCED-v1'}}
    assert dp._case_namespace(cfg,{'dgp':'D5'})=='R61-D5-BALANCED-v1'
    assert dp._case_namespace(cfg,{'dgp':'D1'})=='R61-FORMAL-v1'

def test_exact_bridge_metadata_and_resume(tmp_path):
    old=dict(replicate=1,data_hash='d',holdout_hash='h',training_mask_hash='t',data_seed=73,n_train=8,n_test=2)
    record={'source_sha256':'source'}
    for _ in range(2):dp.export_exact_bridge(tmp_path,old,record,np.eye(2),np.zeros(2),1.,'code','R61-D5-BALANCED-v1')
    with np.load(tmp_path/'statistics/D5/rep_000001.npz') as a:
        assert a['n']==8 and str(a['source_summary_hash'])=='source' and str(a['code_hash'])=='code'
    metadata=dp.read(tmp_path/'replicates/D5/rep_000001.json')
    assert metadata['namespace']=='R61-D5-BALANCED-v1' and metadata['replicate']==1
    with pytest.raises(ValueError):dp.export_exact_bridge(tmp_path,old,record,np.eye(2),np.zeros(2),1.,'changed','R61-D5-BALANCED-v1')


def test_large_offset_prefix_sse_translation_invariance():
    base=np.array([0.,1.,-1.,2.,-2.,0.,1.,-1.]);weights=np.ones(8)
    shifted=1e8+base
    assert dp.partition_objective(shifted,weights,(),2)==pytest.approx(12.+np.log(8),abs=1e-10)
    for kmax in (0,1,3):
        reference=dp.brute_force(shifted,weights,2,kmax)
        cuts=dp.constrained_dp(shifted,weights,2,kmax)
        assert dp.centered_objective(shifted,weights,cuts,2)==pytest.approx(dp.centered_objective(shifted,weights,reference,2),abs=1e-10)
        assert dp.constrained_dp(base,weights,2,kmax)==cuts
    assert dp.exact_dp(base,weights,2)==dp.exact_dp(shifted,weights,2)
    assert dp.sim.pelt_bic(base,weights,2)==dp.sim.pelt_bic(shifted,weights,2)


def test_curve_metrics_keep_conditional_level_fields(monkeypatch):
    monkeypatch.setattr(dp.sim,'test_rmse',lambda *args: .25)
    curve=np.r_[np.full(32,.07),np.full(32,.16)]
    truth=np.r_[np.full(32,.04),np.full(32,.14)]
    row=dp.curve_metrics(curve,(32,),truth,[32],[.04,.14],None,None,35)
    assert row['segment_level_RMSE']==pytest.approx(np.sqrt((.03**2+.02**2)/2))
    assert row['segment_level_bias']==pytest.approx([.03,.02])
    wrong=dp.curve_metrics(curve,(),truth,[32],[.04,.14],None,None,35)
    assert wrong['segment_level_RMSE'] is None and wrong['segment_level_bias'] is None


def test_reconstruction_checkpoint_rejects_dependency_change(tmp_path,monkeypatch):
    source=tmp_path/'source.json';dp.sim.atomic_json(source,dict(case={'stage':'main','cell':'D1'},replicate=1))
    out=tmp_path/'output';cache=out/'reconstructed_inputs/primary/D1/R000001.npz';cache.parent.mkdir(parents=True)
    np.savez_compressed(cache,statistic=np.zeros(8))
    identity={'source_sha256':{'simulation':'old'},'input_sha256':{'config':'old'}}
    monkeypatch.setattr(dp,'reconstruction_identity',lambda *args:identity)
    record=dict(reconstruction_dependency_fingerprint=dp.identity_digest(identity),reconstruction_code_sha256=dp.sim.sha256_file(dp.Path(dp.__file__)),source_sha256=dp.sim.sha256_file(source),cache_sha256=dp.sim.sha256_file(cache))
    dp.sim.atomic_json(cache.with_suffix('.json'),record)
    assert dp.reconstruct_source(source,tmp_path,out)==cache.with_suffix('.json')
    for change in ('simulation','config','design','application'):
        identity={'source_sha256':{'simulation':'old'},'input_sha256':{'config':'old',change:'new'}}
        with pytest.raises(ValueError,match='fingerprint mismatch'):dp.reconstruct_source(source,tmp_path,out)


def test_run_resume_binds_scientific_dependencies(tmp_path,monkeypatch):
    assert dp.main(['--smoke','--output-root',str(tmp_path)])==0
    original=dp.scientific_identity
    monkeypatch.setattr(dp,'scientific_identity',lambda:dict(original(),changed='simulation'))
    with pytest.raises(ValueError,match='scientific dependency'):dp.main(['--smoke','--output-root',str(tmp_path)])


def test_application_resume_identity_includes_statistics(tmp_path):
    source=tmp_path/'source.json';dp.sim.atomic_json(source,{})
    cache=tmp_path/'cache';cache.mkdir()
    groups=[{'group':group,'audit':{'archived_cuts':[]}} for group in ('Low','High')]
    dp.sim.atomic_json(cache/'application.json',dict(source=str(source),source_sha256=dp.sim.sha256_file(source),groups=groups))
    for group in ('Low','High'):np.savez_compressed(cache/(group+'_application_statistics.npz'),statistic=np.zeros(8),weights=np.ones(8))
    args=['--smoke','--cache-root',str(cache),'--output-root',str(tmp_path/'out')]
    assert dp.main(args)==0
    np.savez_compressed(cache/'Low_application_statistics.npz',statistic=np.arange(8.),weights=np.ones(8))
    with pytest.raises(ValueError,match='input fingerprint mismatch'):dp.main(args)


def test_checkpoint_restores_missing_exact_bridge(tmp_path,monkeypatch):
    case={'stage':'main','cell':'D5','dgp':'D5'}
    old=dict(case=case,replicate=1,data_hash='d',holdout_hash='h',training_mask_hash='t',data_seed=73,n_train=8,n_test=2)
    source=tmp_path/'source.json';dp.sim.atomic_json(source,old)
    dp.sim.atomic_json(tmp_path/'resolved_config.json',dict(config={'namespace':'R61-D5-BALANCED-v1'}))
    out=tmp_path/'output';cache=out/'reconstructed_inputs/primary/D5/R000001.npz';cache.parent.mkdir(parents=True)
    np.savez_compressed(cache,H=np.eye(2),h=np.zeros(2),yqy=1.)
    identity={'dependency':'fixed'};monkeypatch.setattr(dp,'reconstruction_identity',lambda *args:identity)
    record=dict(reconstruction_dependency_fingerprint=dp.identity_digest(identity),reconstruction_code_sha256=dp.sim.sha256_file(dp.Path(dp.__file__)),source_sha256=dp.sim.sha256_file(source),cache_sha256=dp.sim.sha256_file(cache))
    dp.sim.atomic_json(cache.with_suffix('.json'),record)
    for _ in range(2):
        assert dp.reconstruct_source(source,tmp_path,out)==cache.with_suffix('.json')
        for p in (out/'statistics/D5/rep_000001.npz',out/'replicates/D5/rep_000001.json'):
            assert p.exists();p.unlink()


def test_cached_exact_bridge_preserves_provenance_and_restores_aliases(tmp_path):
    run=tmp_path/'run';source=run/'replicates/main/D5/rep_000001/summary.json';source.parent.mkdir(parents=True)
    namespace='R61-D5-BALANCED-v1';case={'stage':'main','cell':'D5','dgp':'D5'}
    original=dict(case=case,replicate=1,data_hash='d',holdout_hash='h',training_mask_hash='t',data_seed=dp.sim.deterministic_seed(namespace,'DATA|main|D5|R000001'),n_train=8,n_test=2)
    dp.sim.atomic_json(source,original);dp.sim.atomic_json(run/'resolved_config.json',dict(config={'namespace':namespace}))
    source_hash=dp.sim.sha256_file(source);cache=tmp_path/'old.npz'
    np.savez_compressed(cache,H=np.eye(64),h=np.zeros(64),yqy=1.,code_hash='historical',source_summary_hash=source_hash,data_hash='d',holdout_hash='h',training_mask_hash='t')
    cache_hash=dp.sim.sha256_file(cache);record_path=tmp_path/'record.json'
    dp.sim.atomic_json(record_path,dict(source=str(source),cache=str(cache),source_sha256=source_hash,cache_sha256=cache_hash))
    audited=dict(cohort='primary',cell='D5',rep=1,source_sha256=source_hash,cache_sha256=cache_hash)
    out=tmp_path/'out'
    for _ in range(2):
        staged=dp.stage_cached_exact_input(record_path,tmp_path,out,audited)
        assert staged['original_statistics_sha256']==cache_hash
        metadata=dp.read(out/'replicates/D5/rep_000001.json')
        assert metadata['code_hash']=='historical' and metadata['original_statistics_sha256']==cache_hash
        assert metadata['staging_scope']=='validated archived input staging; no new posterior calculation'
        (out/'statistics/D5/rep_000001.npz').unlink();(out/'replicates/D5/rep_000001.json').unlink()
    cache.write_bytes(cache.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='fingerprint mismatch'):dp.stage_cached_exact_input(record_path,tmp_path,out,audited)
