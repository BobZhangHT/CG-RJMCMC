import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import exact_d5_500 as e
from scipy.special import stdtrit

def test_one_component_matches_student_t():
    ex=dict(weights=np.array([1.]),locations=np.full((1,64),.4),scales=np.full((1,64),.2),df=20.)
    q,c=e.quantiles(ex)
    np.testing.assert_allclose(q[:,0],.4+.2*stdtrit(20,[.025,.975]),atol=1e-11)
    assert c['endpoint_location_certified'] and c['minimum_lower_cdf_margin']>1e-13

def test_invalid_weights_rejected():
    with pytest.raises(ValueError):
        e.quantiles(dict(weights=np.array([.2]),locations=np.zeros((1,64)),scales=np.ones((1,64)),df=10.))

def test_low_density_gap_does_not_pass_residual_only():
    ex=dict(weights=np.array([.025,.975]),locations=np.array([[-10.]*64,[10.]*64]),scales=np.full((2,64),1e-8),df=100.)
    with pytest.raises((e.EndpointCertificationError,ValueError,AssertionError)):
        e.quantiles(ex)

def test_paired_mcse_clusters_by_dataset():
    original=np.array([[[0,0],[1,1]],[[0,0],[1,1]]],float)
    exact=original.copy();exact[0,0,0]=.6
    result=e.summarize_arrays(exact,original,np.array([.5,.5]),np.array([True,False]))
    row=next(r for r in result['paired'] if r['contrast']=='exact_minus_original' and r['metric']=='coverage' and r['stratum']=='all')
    assert row['mean_difference']==-.25
    assert row['paired_mcse']==.25

def test_complete_partition_support():
    states=e.precision.legal_partitions()
    assert len(states)==27931
    assert np.bincount(list(map(len,states))).tolist()==[1,49,861,6545,20475]


def certificate_fixture():
    identity=dict(numerical=dict(endpoint_location_tolerance=1e-8,required_cdf_sign_margin=1e-13,full_cdf_residual_max=1e-8))
    cert=dict(full_support_count=27931,endpoint_location_certified=True,endpoint_location_tolerance=1e-8,required_cdf_sign_margin=1e-13,max_full_cdf_residual=1e-12,minimum_lower_cdf_margin=2e-10,minimum_upper_cdf_margin=2e-10)
    return cert,identity

@pytest.mark.parametrize('field,value', [('endpoint_location_tolerance',1e-7),('required_cdf_sign_margin',1e-14),('minimum_lower_cdf_margin',1e-13),('max_full_cdf_residual',float('nan'))])
def test_archived_certificate_rejects_loose_or_nonfinite_tolerances(field,value):
    cert,identity=certificate_fixture();cert[field]=value
    with pytest.raises(ValueError):e.validate_certificate(cert,identity)

def test_archived_certificate_respects_stricter_recorded_protocol():
    cert,identity=certificate_fixture();e.validate_certificate(cert,identity)
    identity['numerical']['endpoint_location_tolerance']=1e-9
    with pytest.raises(ValueError):e.validate_certificate(cert,identity)
    cert['endpoint_location_tolerance']=1e-9;e.validate_certificate(cert,identity)


def test_unified_public_reconstruction_bounded_preflight(tmp_path,monkeypatch):
    import json,csv
    source=tmp_path/'unified';original=source/'replicates/main/D5/rep_000001';original.mkdir(parents=True)
    cache=tmp_path/'benchmark/reconstructed_inputs/primary/D5';cache.mkdir(parents=True)
    cfg=dict(namespace='R61-FORMAL-v1',namespace_by_dgp={'D5':'R61-D5-BALANCED-v1'},nominal_model=dict(kmax=4,min_segment=8,prior_factorial_power=1.),dgps={'D5':dict(cuts=[13,25,39,51],levels=[.13125,.03125,.13125,.03125,.13125])})
    (source/'resolved_config.json').write_text(json.dumps(dict(config=cfg)))
    case=dict(stage='main',cell='D5',dgp='D5',kmax=4,min_segment=8,prior_factorial_power=1.,participants=500,signal_multiplier=1.,training_mcar=0.)
    summary=dict(case=case,replicate=1,data_seed=e.core.f.deterministic_seed('R61-D5-BALANCED-v1','DATA|main|D5|R000001'),truth=cfg['dgps']['D5'],n_train=25500,n_test=6500,run_fingerprint='synthetic-fixture',data_hash='fixture-data',holdout_hash='fixture-holdout',training_mask_hash='fixture-training',A3=dict(bma_q025=[-1.]*64,bma_q975=[1.]*64))
    summary_path=original/'summary.json';summary_path.write_text(json.dumps(summary))
    with (original/'a3_bma_curve.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['trial','estimate','q025','q975']);writer.writeheader();writer.writerows(dict(trial=t,estimate=0.,q025=-1.,q975=1.) for t in range(1,65))
    npz=cache/'R000001.npz';np.savez(npz,H=np.eye(64),h=np.zeros(64),yqy=100.,source_sha256=e.core.sha(summary_path))
    record=dict(cohort='primary',cell='D5',rep=1,source_sha256=e.core.sha(summary_path),cache_sha256=e.core.sha(npz),reconstruction_code_sha256='fixture-reconstructor')
    (cache/'R000001.json').write_text(json.dumps(record))
    monkeypatch.setattr(e.core,'SOURCE',source);monkeypatch.setattr(e,'ORIGINAL',source/'replicates/main/D5');monkeypatch.setattr(e,'CACHE',cache);monkeypatch.setattr(e,'CACHE_RECORDS',cache)
    identity=e.make_identity([1],'bounded_synthetic_fresh_input_preflight')
    assert identity['inputs']['1']['namespace']=='R61-D5-BALANCED-v1'
    assert identity['inputs']['1']['input_schema']=='public_reconstructed'
    inputs,statistics,quantiles,_=e.case_inputs(1)
    assert statistics['n']==25500 and quantiles.shape==(2,64)
    record['cache_sha256']='wrong';(cache/'R000001.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='fingerprint'):e.case_inputs(1)
