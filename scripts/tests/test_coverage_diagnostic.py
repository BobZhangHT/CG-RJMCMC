"""Independent scalar algebra and dataset-level uncertainty checks."""
import math
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import coverage_diagnostic as cd


def scalar(case,cuts,lam):
    edges=(0,)+cuts+(64,);p=len(edges)-1
    z=np.zeros((64,p))
    for i,(a,b) in enumerate(zip(edges[:-1],edges[1:])):z[a:b,i]=1
    a=z.T@case['H']@z+lam*np.eye(p);g=z.T@case['h']
    mu=np.linalg.solve(a,g);diag=np.diag(np.linalg.inv(a))
    shape=3+case['n']/2;rate=2+.5*case['yqy']-.5*g@mu
    lw=(-math.lgamma(len(cuts)+1)-cd.exact.precision._log_partition_count(len(cuts))
        +p/2*math.log(lam)-np.linalg.slogdet(a)[1]/2-shape*math.log(rate))
    return lw,z@mu,z@np.sqrt(rate/shape*diag)


@pytest.mark.parametrize('lam',[1.,4.,16.])
def test_generalized_scalar_equivalence_and_dimensional_normalization(lam):
    rng=np.random.default_rng(92);x=rng.normal(size=(96,64));y=rng.normal(size=96)
    case=dict(H=x.T@x,h=x.T@y,yqy=y@y,n=96)
    states=[(),(8,),(13,25),(13,25,39,51)]
    actual=cd.batched_parameters(case,states,(lam,))[lam]
    reference=[scalar(case,s,lam) for s in states]
    lw=np.array([r[0] for r in reference]);weights=np.exp(lw-cd.logsumexp(lw))
    np.testing.assert_allclose(actual['weights'],weights,atol=1e-12,rtol=1e-11)
    np.testing.assert_allclose(actual['locations'],[r[1] for r in reference],atol=1e-12)
    np.testing.assert_allclose(actual['scales'],[r[2] for r in reference],atol=1e-12)
    if lam==4:
        for i,s in enumerate(states):
            old=cd.exact.precision.partition_posterior_parameters(case['H'],case['h'],case['yqy'],case['n'],s)
            assert old[0]==pytest.approx(reference[i][0],abs=1e-10)


def test_coverage_aggregation_uses_independent_datasets_and_paired_mcse():
    # Dataset-wide common errors make treating coordinates as independent wrong.
    truth=cd.TRUTH
    means={m:truth+np.array([-.02,0.,.04])[:,None]+np.zeros((3,64)) for m in cd.METHODS}
    q={m:np.stack([means[m]-.01,means[m]+.01],axis=1) for m in cd.METHODS}
    rows,paired,_=cd.summarize_data(q,means)
    row=next(r for r in rows if r['method']==cd.METHODS[0] and r['stratum']=='all' and r['metric']=='coverage')
    assert row['estimate']==pytest.approx(1/3)
    assert row['mcse']==pytest.approx(np.std([0,1,0],ddof=1)/np.sqrt(3))
    assert all(r['paired_mcse']==pytest.approx(0) for r in paired if r['left']!=r['right'])
    rmse=next(r for r in rows if r['method']==cd.METHODS[0] and r['stratum']=='all' and r['metric']=='RMSE')
    assert rmse['estimate']==pytest.approx(np.sqrt((.02**2+.04**2)/3))


def test_identity_and_certificate_fail_closed(tmp_path):
    identity={'identity_sha256':'a','inputs':{'1':{'x':1}}}
    cd.exact.bind_identity(tmp_path,identity)
    with pytest.raises(RuntimeError):cd.exact.bind_identity(tmp_path,dict(identity,identity_sha256='b'))
    cert=dict(full_support_count=1,endpoint_location_certified=True,endpoint_location_tolerance=1e-8,
        required_cdf_sign_margin=1e-13,max_full_cdf_residual=1e-14,minimum_lower_cdf_margin=2e-13,minimum_upper_cdf_margin=2e-13)
    cd.validate_certificate(cert,1)
    with pytest.raises(ValueError):cd.validate_certificate(dict(cert,minimum_lower_cdf_margin=1e-14),1)


def test_oracle_true_partition_and_boundary_rule():
    assert tuple(cd.exact.CUTS)==(13,25,39,51)
    assert cd.NEAR.sum()==20 and (~cd.NEAR).sum()==44
    assert cd.TRUTH[12]==.13125 and cd.TRUTH[13]==.03125


def test_large_logweights_keep_unit_sum_and_mixed_dimension_prior():
    lw=np.array([-180000.3,-180000.5,-180003.7])
    weights=cd.normalize_logweights(lw)
    assert weights.sum()==pytest.approx(1.,abs=2e-16)
    np.testing.assert_allclose(weights,cd.normalize_logweights(lw+100000.),atol=1e-12)


def test_ambiguous_coverage_requires_cdf_separation(monkeypatch):
    mixture=dict(weights=np.ones(1),locations=np.zeros((1,64)),scales=np.ones((1,64)),df=20.)
    q=np.array([cd.TRUTH,cd.TRUTH+1.])
    monkeypatch.setattr(cd,'stdtr',lambda df,x:np.full_like(x,.025+5e-14))
    with pytest.raises(cd.exact.EndpointCertificationError):cd.coverage_decisions(q,mixture)
    monkeypatch.setattr(cd,'stdtr',lambda df,x:np.full_like(x,.5))
    below,above,ambiguous=cd.coverage_decisions(q,mixture)
    assert ambiguous.all() and not below.any() and not above.any()


def test_proposal_certifies_full_support_and_exact_duplicate_mapping():
    ex=dict(weights=np.array([.3,.7]),locations=np.array([[0.]*64,[1.]*64]),scales=np.full((2,64),.15),df=20.)
    ex['locations'][0,10]=.01
    actual,cert=cd.accelerated_quantiles(ex)
    expected,_=cd.exact.quantiles(ex)
    np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=0)
    assert cert['unique_coordinates']==2
    assert cert['full_support_count']==2 and cert['newly_evaluated_endpoints']==4
    assert cert['minimum_lower_cdf_margin']>1e-13 and cert['minimum_upper_cdf_margin']>1e-13


def test_low_density_proposal_falls_back_to_full_cdf():
    ex=dict(weights=np.array([8e-13,1-8e-13]),locations=np.array([[-100000.]*64,[0.]*64]),
        scales=np.array([[1.]*64,[1000.]*64]),df=20.)
    actual,cert=cd.accelerated_quantiles(ex)
    expected,_=cd.exact.quantiles(ex)
    np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=0)
    assert cert['full_CDF_fallback_endpoints']>0
    assert cert['auxiliary_omitted_mass']<=1e-12


def test_uncertifiable_endpoint_fails_closed():
    ex=dict(weights=np.ones(1),locations=np.zeros((1,64)),scales=np.full((1,64),1e8),df=20.)
    with pytest.raises(cd.exact.EndpointCertificationError):cd.accelerated_quantiles(ex)


def test_candidate_solver_failure_uses_full_cdf_fallback(monkeypatch):
    ex=dict(weights=np.ones(1),locations=np.zeros((1,64)),scales=np.ones((1,64)),df=20.)
    real=cd.brentq;calls=[0]
    def fail_once(*args,**kwargs):
        calls[0]+=1
        if calls[0]==1:raise ValueError('Deliberate failed proposal bracket')
        return real(*args,**kwargs)
    monkeypatch.setattr(cd,'brentq',fail_once)
    actual,cert=cd.accelerated_quantiles(ex)
    expected,_=cd.exact.quantiles(ex)
    np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=0)
    assert cert['full_CDF_fallback_endpoints']==1




def test_protocol_json_round_trip_supports_identity_resume(tmp_path):
    # Run identities are serialized to JSON between invocations.
    import json
    identity = {"protocol": cd.protocol(), "identity_sha256": "same"}
    assert json.loads(json.dumps(identity)) == identity
    cd.exact.bind_identity(tmp_path, identity)
    cd.exact.bind_identity(tmp_path, identity)
