"""Same-data D5 phase-prior x partition-knowledge diagnostic; never simulates data.

All roots are explicit. Pilot and formal500 identities cannot be interchanged.
The original scalar posterior and certified full-CDF solver remain unchanged.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import itertools
import math
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.special import stdtr
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits
if __package__:
    from . import exact_d5_500 as exact
else:
    import exact_d5_500 as exact

ROOT=exact.core.ROOT
PRIORS={'original':4.,'twice_sd':1.,'half_sd':16.}
METHODS=tuple(f'{prior}__{knowledge}' for prior in PRIORS for knowledge in ('unknown','oracle'))
TRUTH=exact.TRUTH
NEAR=exact.NEAR


def accelerated_quantiles(ex,auxiliary_mass=1e-12):
    """Active-CDF proposals, accepted only by full-support crossing certificates.

    Omitted mass controls proposal speed alone. Every accepted endpoint has
    the unchanged full-mixture residual and q +/- 1e-8 strict crossing checks.
    If a proposal fails, a full-support Brent solve is certified by the same
    checks. Duplicate coordinates are reused only after exact array equality.
    """
    w=np.asarray(ex['weights']);loc=np.asarray(ex['locations']);scale=np.asarray(ex['scales']);df=ex['df']
    if w.ndim!=1 or loc.shape!=(len(w),64) or scale.shape!=loc.shape:raise ValueError('Invalid mixture dimensions')
    if not np.isfinite(w).all() or np.any(w<0) or abs(w.sum()-1)>1e-12 or not np.isfinite(loc).all() or not np.isfinite(scale).all() or np.any(scale<=0) or not np.isfinite(df) or df<=0:raise ValueError('Invalid Student-t mixture')
    if not 0<auxiliary_mass<=1e-12:raise ValueError('Invalid proposal omitted-mass bound')
    loc_by_coordinate=np.ascontiguousarray(loc.T)
    scale_by_coordinate=np.ascontiguousarray(scale.T)
    order=np.argsort(w);omitted_count=np.searchsorted(np.cumsum(w[order]),auxiliary_mass)
    active=order[omitted_count:];omitted=float(w[order[:omitted_count]].sum())
    if omitted>auxiliary_mass:raise ValueError('Proposal omission exceeds bound')
    coarse_count=np.searchsorted(np.cumsum(w[order]),1e-4)
    coarse=order[coarse_count:];coarse_omitted=float(w[order[:coarse_count]].sum())
    roots=np.empty((2,64));residuals=np.empty_like(roots);representatives=[];mapping=[]
    for t in range(64):
        same=next((r for r in representatives if np.array_equal(loc[:,t],loc[:,r]) and np.array_equal(scale[:,t],scale[:,r])),None)
        if same is None:representatives.append(t);mapping.append(t)
        else:mapping.append(same)
    full_calls=partial_calls=fallback_count=0;minimum_lower=minimum_upper=float('inf')
    for t in representatives:
        lo=float(loc[:,t].min()-16*scale[:,t].max());hi=float(loc[:,t].max()+16*scale[:,t].max())
        cache={}
        lt=loc_by_coordinate[t];st=scale_by_coordinate[t]
        wa=w[active];la=loc[active,t];sa=scale[active,t]
        wc=w[coarse];lc=loc[coarse,t];sc=scale[coarse,t]
        def full(x):
            if x not in cache:cache[x]=float(np.sum(w*stdtr(df,(x-lt)/st)))
            return cache[x]
        def partial(x):return float(np.sum(wa*stdtr(df,(x-la)/sa)))
        def coarse_cdf(x):return float(np.sum(wc*stdtr(df,(x-lc)/sc)))
        for j,p in enumerate((.025,.975)):
            # Coarse roots only bracket the tiny-omission proposal. Their
            # omitted mass cannot enter an accepted endpoint certificate.
            def certificate(x):return abs(full(x)-p),p-full(x-1e-8),full(x+1e-8)-p
            try:
                qa,ia=brentq(lambda x:coarse_cdf(x)-(p-coarse_omitted),lo,hi,xtol=1e-12,rtol=1e-14,full_output=True)
                qb,ib=brentq(lambda x:coarse_cdf(x)-p,lo,hi,xtol=1e-12,rtol=1e-14,full_output=True)
                if not ia.converged or not ib.converged:raise RuntimeError('Coarse bracket solver failed')
                qa-=1e-10;qb+=1e-10
                partial_calls+=ia.function_calls+ib.function_calls
                if not partial(qa)<=p<=partial(qb):qa,qb=lo,hi
                # Unnormalized active CDF, deliberately not renormalized.
                q,info=brentq(lambda x:partial(x)-p,qa,qb,xtol=1e-12,rtol=1e-14,full_output=True)
                partial_calls+=info.function_calls
                if not info.converged:raise RuntimeError('Proposal solver failed')
                residual,lower,upper=certificate(q)
            except (ValueError,RuntimeError,FloatingPointError,OverflowError):
                residual,lower,upper=float('inf'),float('-inf'),float('-inf')
            if not (residual<1e-8 and lower>1e-13 and upper>1e-13):
                fallback_count+=1
                if not full(lo)<p<full(hi):raise ValueError('Full-CDF fallback bracket failed')
                q,info=brentq(lambda x:full(x)-p,lo,hi,xtol=1e-12,rtol=1e-14,full_output=True)
                if not info.converged:raise RuntimeError('Full-CDF fallback solver failed')
                residual,lower,upper=certificate(q)
            if not (residual<1e-8 and lower>1e-13 and upper>1e-13):
                raise exact.EndpointCertificationError('Full-mixture endpoint location could not be certified',dict(
                    bin=t+1,probability=p,endpoint=float(q),lower_margin=lower,upper_margin=upper,cdf_residual=residual,
                    full_support_count=len(w),location_tolerance=1e-8,required_cdf_sign_margin=1e-13))
            roots[j,t]=q;residuals[j,t]=residual;minimum_lower=min(minimum_lower,lower);minimum_upper=min(minimum_upper,upper)
        full_calls+=len(cache)
    for t,r in enumerate(mapping):
        if t!=r:roots[:,t]=roots[:,r];residuals[:,t]=residuals[:,r]
    if not np.isfinite(roots).all() or np.any(roots[0]>roots[1]):raise ValueError('Invalid endpoints')
    return roots,dict(full_support_count=len(w),auxiliary_support_count=len(active),auxiliary_omitted_mass=omitted,
        max_full_cdf_residual=float(residuals.max()),full_cdf_calls=full_calls,partial_cdf_calls=partial_calls,
        final_solver='Active-CDF proposal followed by full-support residual and crossing certificate; full-CDF Brent fallback',
        xtol=1e-12,rtol=1e-14,endpoint_location_certified=True,endpoint_location_tolerance=1e-8,
        required_cdf_sign_margin=1e-13,minimum_lower_cdf_margin=minimum_lower,minimum_upper_cdf_margin=minimum_upper,
        unique_coordinates=len(representatives),coordinate_representatives=mapping,full_CDF_fallback_endpoints=fallback_count,
        newly_evaluated_endpoints=2*len(representatives),cached_baseline=False,coarse_bracket_support_count=len(coarse),coarse_bracket_omitted_mass=coarse_omitted)


def logsumexp(values):
    """Stable NumPy-only normalization, avoiding optional array-backend imports."""
    maximum=np.max(values)
    return maximum+np.log(np.exp(values-maximum).sum())


def normalize_logweights(values):
    weights=np.exp(values-np.max(values))
    return weights/weights.sum()


def scalar_original_parameters(case,states):
    locations=[];scales=[];logweights=[]
    for cuts in states:
        lw,mu,diag,rate,shape=exact.precision.partition_posterior_parameters(case['H'],case['h'],case['yqy'],case['n'],cuts)
        phase=np.searchsorted(cuts,np.arange(1,65),side='left')
        locations.append(mu[phase]);scales.append(np.sqrt(rate/shape*diag)[phase]);logweights.append(lw)
    lw=np.array(logweights)
    return dict(locations=np.array(locations),scales=np.array(scales),weights=normalize_logweights(lw))


def batched_parameters(case, states, precisions=(4.,1.,16.)):
    """Prefix sums produce Z'HZ and Z'h; one eigensystem serves each lambda.

    For p phases, log prior normalization is p/2 log(lambda). Keeping this
    term is essential when comparing partitions of different dimensions.
    """
    count=len(states)
    output={lam:dict(locations=np.empty((count,64)),scales=np.empty((count,64)),
                    logweights=np.empty(count),df=6.+case['n']) for lam in precisions}
    hp=np.r_[0.,np.cumsum(case['h'])]
    prefix=np.pad(np.cumsum(np.cumsum(case['H'],axis=0),axis=1),((1,0),(1,0)))
    shape=3.+case['n']/2.
    sizes=np.array([len(s) for s in states])
    for k in np.unique(sizes):
        ids=np.flatnonzero(sizes==k)
        edges=np.array([(0,)+states[i]+(64,) for i in ids])
        a,b=edges[:,:-1],edges[:,1:]
        matrices=(prefix[b[:,:,None],b[:,None,:]]-prefix[a[:,:,None],b[:,None,:]]
                  -prefix[b[:,:,None],a[:,None,:]]+prefix[a[:,:,None],a[:,None,:]])
        # H is validated symmetric; averaging suppresses prefix cancellation.
        eigen,vectors=np.linalg.eigh((matrices+matrices.transpose(0,2,1))/2)
        g=hp[b]-hp[a]
        projected=np.einsum('bij,bi->bj',vectors,g)
        phase=np.sum(np.arange(64)[None,:,None]>=edges[:,None,1:-1],axis=2)
        for lam in precisions:
            if not np.isfinite(lam) or lam<=0: raise ValueError('Invalid prior precision')
            denom=eigen+lam
            if np.any(denom<=0): raise ValueError('Nonpositive posterior precision')
            mu=np.einsum('bij,bj->bi',vectors,projected/denom)
            diag=np.einsum('bij,bj->bi',vectors*vectors,1./denom)
            rate=2.+.5*case['yqy']-.5*np.einsum('bi,bi->b',g,mu)
            if np.any(rate<=0): raise ValueError('Nonpositive posterior rate')
            target=output[lam]
            target['logweights'][ids]=(-math.lgamma(int(k)+1)-exact.precision._log_partition_count(int(k))
                +.5*(k+1)*np.log(lam)-.5*np.log(denom).sum(1)-shape*np.log(rate))
            target['locations'][ids]=np.take_along_axis(mu,phase,axis=1)
            target['scales'][ids]=np.take_along_axis(np.sqrt(rate[:,None]/shape*diag),phase,axis=1)
    for target in output.values():
        lw=target.pop('logweights');target['weights']=normalize_logweights(lw)
    return output


def protocol():
    return dict(schema=1,design='same-data 3 prior scales x 2 partition knowledge',
        prior_precision=PRIORS,prior_sd_sigma_units={'original':.5,'twice_sd':1.,'half_sd':.25},
        unknown_support=27931,oracle_cuts=exact.CUTS.tolist(),count_prior='factorial power 1; uniform legal partitions conditional on K',
        variance_prior=dict(shape=3.,scale=2.),nuisance_model='unchanged archived sufficient statistics',
        quantiles=[.025,.975],endpoint_location_tolerance=1e-8,required_cdf_sign_margin=1e-13,
        near_rule='distance to cut <=2 on original 1..64 coordinates',near_positions=exact.GRID[NEAR].tolist(),
        far_positions=exact.GRID[~NEAR].tolist(),replication_unit='independent dataset; paired within dataset',
        truth=TRUTH.tolist(),methods=list(METHODS),new_data=False,new_MCMC=False,
        metrics=['coverage','signed_bias','RMSE','width','miscoverage_below','miscoverage_above'],
        interpretation='Unknown-partition prior contrasts combine phase shrinkage and marginal partition-weight changes; oracle contrasts isolate phase-prior contribution conditional on true cuts. Interactions are descriptive, not an additive causal decomposition.',
        strata=['all','near','far','low_phase','high_phase','phase1','phase2','phase3','phase4','phase5'],
        phase_positions={f'phase{i+1}':exact.GRID[np.searchsorted(exact.CUTS,exact.GRID,side='left')==i].tolist() for i in range(5)},
        rmse_mcse='delta method on dataset MSE; paired contrasts use difference of influence values')


def configure(identity):
    exact.core.SOURCE=Path(identity['roots']['simulation'])
    exact.ORIGINAL=exact.core.SOURCE/'replicates/main/D5'
    exact.CACHE=Path(identity['roots']['statistics'])
    exact.CACHE_RECORDS=Path(identity['roots']['records'])
    threadpool_limits(limits=1)


def make_identity(args,reps):
    roots=dict(simulation=str(args.simulation_results.resolve()),statistics=str(args.statistics_root.resolve()),
               records=str(args.records_root.resolve()),baseline=str(args.baseline_exact_root.resolve()))
    configure(dict(roots=roots))
    # The public preflight checks namespace, truth, model and all data identities.
    original=exact.make_identity(reps,'bounded_input_preflight')
    sources=original['source_sha256']|{str(Path(__file__).resolve()):exact.core.sha(Path(__file__))}
    archive=exact.read_json(args.baseline_exact_root/'run_identity.json')
    if exact.digest({k:v for k,v in archive.items() if k!='identity_sha256'})!=archive['identity_sha256']:
        raise ValueError('Baseline archive identity digest invalid')
    identity=dict(schema=1,mode='pilot' if args.pilot else 'formal500',replications=reps,roots=roots,
        inputs=original['inputs'],source_sha256=sources,versions=original['versions'],protocol=protocol(),
        baseline_identity_sha256=archive['identity_sha256'],baseline_sha256={
            str(p.resolve()):exact.core.sha(p) for rep in reps for p in exact.result_paths(args.baseline_exact_root,rep)})
    identity['identity_sha256']=exact.digest(identity)
    return identity


def paths(output,rep):
    p=Path(output)/'replicates'/f'rep_{rep:06d}'
    return p.with_suffix('.json'),p.with_suffix('.npz')


def validate_certificate(cert,support):
    # Reuse original strict validator, with only support cardinality adapted
    # for the oracle's single conditional Student-t component.
    if cert['full_support_count']!=support: raise ValueError('Support mismatch')
    adapted=dict(cert,full_support_count=27931)
    exact.validate_certificate(adapted,dict(numerical=dict(endpoint_location_tolerance=1e-8,required_cdf_sign_margin=1e-13)))


def coverage_decisions(q,mixture):
    ambiguous=np.any(abs(q-TRUTH)<1e-8,axis=0)
    below=q[1]<TRUTH;above=q[0]>TRUTH
    if ambiguous.any():
        cdf=np.sum(mixture['weights'][:,None]*stdtr(mixture['df'],(TRUTH[ambiguous]-mixture['locations'][:,ambiguous])/mixture['scales'][:,ambiguous]),axis=0)
        unresolved=(abs(cdf-.025)<=1e-13)|(abs(cdf-.975)<=1e-13)
        if unresolved.any():
            raise exact.EndpointCertificationError('Truth-CDF coverage decision lacks numerical separation',dict(
                bins=exact.GRID[ambiguous][unresolved].tolist(),cdf_at_truth=cdf[unresolved].tolist(),required_margin=1e-13))
        below[ambiguous]=cdf>.975;above[ambiguous]=cdf<.025
    return below,above,ambiguous


def validate_result(output,rep,identity):
    jp,npz=paths(output,rep)
    if not jp.exists(): return None
    item=exact.read_json(jp)
    if item.get('status')!='complete' or item['identity_sha256']!=identity['identity_sha256'] or item['inputs']!=identity['inputs'][str(rep)]:
        raise ValueError('Resume identity/status mismatch')
    if not npz.is_file() or exact.core.sha(npz)!=item['npz_sha256']:raise ValueError('Saved arrays missing/corrupt')
    for method in METHODS: validate_certificate(item['certificates'][method],27931 if method.endswith('unknown') else 1)
    with np.load(npz,allow_pickle=False) as z:
        if z['identity_sha256'].item()!=identity['identity_sha256'] or not np.array_equal(z['truth'],TRUTH):raise ValueError('NPZ identity/truth mismatch')
        for method in METHODS:
            q=z[method+'__quantiles'];mean=z[method+'__mean'];pk=z[method+'__p_k']
            if q.shape!=(2,64) or mean.shape!=(64,) or not np.isfinite(q).all() or not np.isfinite(mean).all() or np.any(q[0]>q[1]):raise ValueError('Invalid arrays')
            w=z[method+'__weights'];support=27931 if method.endswith('unknown') else 1
            if w.shape!=(support,) or not np.isfinite(w).all() or np.any(w<0) or abs(w.sum()-1)>1e-12:raise ValueError('Invalid weights')
            if pk.shape!=(5,) or not np.isfinite(pk).all() or np.any(pk<0) or abs(pk.sum()-1)>1e-12:raise ValueError('Invalid count probabilities')
            for name in ('coverage','miscoverage_below','miscoverage_above'):
                if z[method+'__'+name].shape!=(64,) or z[method+'__'+name].dtype!=bool:raise ValueError('Invalid coverage decision array')
            if not np.array_equal(z[method+'__coverage'],~(z[method+'__miscoverage_below']|z[method+'__miscoverage_above'])):raise ValueError('Inconsistent coverage decisions')
    return item


def validate_baseline(rep,identity,q,w):
    root=Path(identity['roots']['baseline']);archive=exact.read_json(root/'run_identity.json')
    if archive['identity_sha256']!=identity['baseline_identity_sha256']:raise ValueError('Baseline archive changed')
    saved=exact.validate_result(root,rep,archive)
    if saved is None:raise ValueError('Missing baseline')
    archived=saved['inputs'];current=identity['inputs'][str(rep)]
    for key in ('replication','data_seed','data_hash','holdout_hash','training_mask_hash','n_train','n_test','run_fingerprint','cache_generation_code_hash'):
        if archived[key]!=current[key]:raise ValueError(f'Baseline data identity mismatch: {key}')
    normalized=lambda hashes:{str((ROOT/Path(p)).resolve()):v for p,v in hashes.items()}
    if normalized(archived['input_sha256'])!=normalized(current['input_sha256']):raise ValueError('Baseline input hashes differ')
    exact.verify_hashes(saved['inputs']['input_sha256'])
    for p in exact.result_paths(root,rep):
        if exact.core.sha(p)!=identity['baseline_sha256'][str(p.resolve())]:raise ValueError('Baseline file changed since preflight')
    with np.load(exact.result_paths(root,rep)[1],allow_pickle=False) as z:
        qdiff=float(np.max(abs(q-np.array([z['exact_q025'],z['exact_q975']]))))
        wdiff=float(np.max(abs(w-z['partition_weights'])))
    if qdiff>1e-8 or wdiff>1e-10:raise ValueError(f'Baseline mismatch: endpoints {qdiff}, weights {wdiff}')
    return dict(max_endpoint_difference=qdiff,max_weight_difference=wdiff,archived_record_sha256=exact.core.sha(exact.result_paths(root,rep)[0]))


def compute_case(rep,output,identity):
    configure(identity);start=time.perf_counter()
    exact.verify_hashes(identity['source_sha256'])
    inputs,case,_,_=exact.case_inputs(rep)
    if inputs!=identity['inputs'][str(rep)]:raise ValueError('Input changed after preflight')
    states=exact.precision.legal_partitions();counts=np.array(list(map(len,states)))
    if len(states)!=27931 or np.bincount(counts).tolist()!=[1,49,861,6545,20475]:raise ValueError('Support changed')
    parameter_start=time.perf_counter()
    unknown=batched_parameters(case,states)
    oracle=batched_parameters(case,[tuple(exact.CUTS)])
    parameter_seconds=time.perf_counter()-parameter_start
    scalar_check=None
    if identity['mode']=='pilot':
        scalar_start=time.perf_counter();reference=scalar_original_parameters(case,states)
        scalar_check={key:float(np.max(abs(reference[key]-unknown[4.][key]))) for key in ('weights','locations','scales')}
        scalar_check['seconds']=time.perf_counter()-scalar_start
        if scalar_check['weights']>1e-10 or max(scalar_check['locations'],scalar_check['scales'])>1e-8:raise ValueError('Vectorized/scalar posterior mismatch')
    arrays=dict(truth=TRUTH,near=NEAR,identity_sha256=np.array(identity['identity_sha256']))
    certs={};timings={};baseline=None;posterior={};ambiguities={}
    for prior,lam in PRIORS.items():
        for knowledge,mixture in (('unknown',unknown[lam]),('oracle',oracle[lam])):
            method=prior+'__'+knowledge;t=time.perf_counter()
            try:
                if method=='original__unknown':
                    baseline_root=Path(identity['roots']['baseline'])
                    archive=exact.read_json(baseline_root/'run_identity.json')
                    saved=exact.validate_result(baseline_root,rep,archive)
                    if saved is None:raise ValueError('Missing archived baseline')
                    with np.load(exact.result_paths(baseline_root,rep)[1],allow_pickle=False) as z:q=np.array([z['exact_q025'],z['exact_q975']])
                    baseline=validate_baseline(rep,identity,q,mixture['weights'])
                    cert=dict(saved['checks'],cached_baseline=True,historical_full_cdf_calls=saved['checks']['full_cdf_calls'],full_cdf_calls=0,
                        newly_evaluated_endpoints=0,baseline_record_sha256=baseline['archived_record_sha256'])
                else:q,cert=accelerated_quantiles(mixture)
                below,above,ambiguous=coverage_decisions(q,mixture)
            except exact.EndpointCertificationError as exc:
                exact.atomic_json(Path(output)/'unresolved'/f'rep_{rep:06d}_{method}.json',dict(
                    status='unresolved_endpoint_location',replication=rep,method=method,identity_sha256=identity['identity_sha256'],
                    certificate_failure=exc.details,excluded_from_planned_set=False))
                raise
            validate_certificate(cert,len(mixture['weights']));certs[method]=cert;timings[method]=time.perf_counter()-t
            arrays[method+'__quantiles']=q;arrays[method+'__mean']=mixture['weights']@mixture['locations']
            arrays[method+'__weights']=mixture['weights']
            arrays[method+'__p_k']=np.bincount(counts,weights=mixture['weights'],minlength=5) if knowledge=='unknown' else np.array([0.,0.,0.,0.,1.])
            arrays[method+'__miscoverage_below']=below;arrays[method+'__miscoverage_above']=above
            arrays[method+'__coverage']=~(below|above)
            ambiguities[method]=int(ambiguous.sum())
            posterior[method]=dict(mean_K=float(np.arange(5)@arrays[method+'__p_k']),probability_K4=float(arrays[method+'__p_k'][4]),
                probability_true_partition=float(mixture['weights'][states.index(tuple(exact.CUTS))]) if knowledge=='unknown' else 1.)
            if method=='original__unknown':baseline=validate_baseline(rep,identity,q,mixture['weights'])
    exact.verify_hashes(inputs['input_sha256'])
    jp,npz=paths(output,rep);exact.atomic_npz(npz,**arrays)
    item=dict(status='complete',replication=rep,identity_sha256=identity['identity_sha256'],inputs=inputs,
        npz_sha256=exact.core.sha(npz),certificates=certs,baseline_validation=baseline,scalar_validation=scalar_check,
        posterior=posterior,truth_within_endpoint_tolerance=ambiguities,ambiguous_coverage_rule='full-mixture CDF at truth',
        seconds=dict(parameters=parameter_seconds,quantiles=timings,total=time.perf_counter()-start),
        peak_working_set_bytes=exact.core.f.process_peak_working_set(),new_MCMC=0)
    exact.atomic_json(jp,item);return item


def metrics(quantiles,means,mask,decisions=None):
    errors=means[:,mask]-TRUTH[mask]
    result=dict(coverage=((quantiles[:,0,mask]<=TRUTH[mask])&(TRUTH[mask]<=quantiles[:,1,mask])).mean(1),
        signed_bias=errors.mean(1),RMSE=np.square(errors).mean(1),width=(quantiles[:,1,mask]-quantiles[:,0,mask]).mean(1))
    result['miscoverage_below']=(quantiles[:,1,mask]<TRUTH[mask]).mean(1)
    result['miscoverage_above']=(quantiles[:,0,mask]>TRUTH[mask]).mean(1)
    if decisions is not None:
        for name in ('coverage','miscoverage_below','miscoverage_above'):result[name]=decisions[name][:,mask].mean(1)
    return result


def estimate(values,metric):
    value=float(values.mean())
    if metric=='RMSE':
        root=math.sqrt(value)
        influence=(values-value)/(2*root) if root else np.zeros_like(values)
        return root,influence
    return value,values-value


def summarize_data(q,means,decisions=None):
    n=len(q[METHODS[0]])
    masks=dict(all=np.ones(64,bool),near=NEAR,far=~NEAR,low_phase=TRUTH==TRUTH.min(),high_phase=TRUTH==TRUTH.max())
    masks.update({f'phase{i+1}':np.searchsorted(exact.CUTS,exact.GRID,side='left')==i for i in range(5)})
    stats={};rows=[];paired=[];per_dataset=[]
    for method in METHODS:
        for stratum,mask in masks.items():
            data=metrics(q[method],means[method],mask,None if decisions is None else decisions[method])
            for metric,values in data.items():
                value,influence=estimate(values,metric);stats[method,stratum,metric]=(value,influence)
                rows.append(dict(method=method,stratum=stratum,metric=metric,estimate=value,mcse=exact.mcse(influence),replications=n,positions=int(mask.sum())))
                per_dataset.extend(dict(replication_index=i,method=method,stratum=stratum,metric='MSE' if metric=='RMSE' else metric,value=float(v)) for i,v in enumerate(values))
    for left,right in itertools.combinations(METHODS,2):
        for stratum in masks:
            for metric in protocol()['metrics']:
                a,ia=stats[left,stratum,metric];b,ib=stats[right,stratum,metric]
                paired.append(dict(left=left,right=right,stratum=stratum,metric=metric,difference=a-b,paired_mcse=exact.mcse(ia-ib),pairs=n))
    for method in METHODS:
        for left,right in (('near','far'),('low_phase','high_phase')):
            for metric in protocol()['metrics']:
                a,ia=stats[method,left,metric];b,ib=stats[method,right,metric]
                paired.append(dict(left=method,right=method,stratum=left+'_minus_'+right,metric=metric,difference=a-b,paired_mcse=exact.mcse(ia-ib),pairs=n))
    for prior in ('twice_sd','half_sd'):
        for stratum in masks:
            for metric in protocol()['metrics']:
                terms=[stats[prior+'__oracle',stratum,metric],stats['original__oracle',stratum,metric],
                       stats[prior+'__unknown',stratum,metric],stats['original__unknown',stratum,metric]]
                value=terms[0][0]-terms[1][0]-terms[2][0]+terms[3][0]
                influence=terms[0][1]-terms[1][1]-terms[2][1]+terms[3][1]
                paired.append(dict(left=prior+'_minus_original_oracle',right=prior+'_minus_original_unknown',
                    stratum=stratum,metric=metric,difference=value,paired_mcse=exact.mcse(influence),pairs=n))
    return rows,paired,per_dataset


def aggregate(output,identity):
    reps=identity['replications'];q={m:[] for m in METHODS};means={m:[] for m in METHODS};records=[]
    decisions={m:{name:[] for name in ('coverage','miscoverage_below','miscoverage_above')} for m in METHODS}
    if identity['mode']=='formal500' and reps!=list(range(1,501)):raise ValueError('Formal summary requires 500 datasets')
    for rep in reps:
        item=validate_result(output,rep,identity)
        if item is None:raise ValueError(f'Incomplete diagnostic: replication {rep}')
        records.append(item)
        with np.load(paths(output,rep)[1],allow_pickle=False) as z:
            for m in METHODS:
                q[m].append(z[m+'__quantiles']);means[m].append(z[m+'__mean'])
                for name in decisions[m]:decisions[m][name].append(z[m+'__'+name])
    rows,paired,per_dataset=summarize_data({m:np.array(v) for m,v in q.items()},{m:np.array(v) for m,v in means.items()},
        {m:{name:np.array(v) for name,v in d.items()} for m,d in decisions.items()})
    for row in per_dataset:row['replication']=reps[row.pop('replication_index')]
    for name,data in (('strata',rows),('paired_contrasts',paired),('per_dataset',per_dataset)):exact.write_csv(Path(output)/(name+'.csv'),data)
    result=dict(identity_sha256=identity['identity_sha256'],mode=identity['mode'],replications=len(reps),
        strata=rows,paired_contrasts=paired,mean_case_seconds=float(np.mean([r['seconds']['total'] for r in records])),
        max_peak_working_set_bytes=max(r['peak_working_set_bytes'] or 0 for r in records),new_MCMC=0,
        all_endpoint_certificates_passed=True,truth_within_endpoint_tolerance=sum(sum(r['truth_within_endpoint_tolerance'].values()) for r in records),
        posterior=[dict(replication=r['replication'],methods=r['posterior']) for r in records],
        baseline_max_endpoint_difference=max(r['baseline_validation']['max_endpoint_difference'] for r in records),
        baseline_max_weight_difference=max(r['baseline_validation']['max_weight_difference'] for r in records))
    result['posterior_summary']=[dict(method=m,metric=metric,estimate=float(np.mean(values)),mcse=exact.mcse(values),replications=len(reps))
        for m in METHODS for metric in ('mean_K','probability_K4','probability_true_partition')
        for values in [np.array([r['posterior'][m][metric] for r in records])]]
    exact.write_csv(Path(output)/'posterior_summary.csv',result['posterior_summary'])
    result['numerical_work']=dict(cached_original_unknown_datasets=len(records),
        new_full_CDF_evaluations=sum(c.get('full_cdf_calls',0) for r in records for c in r['certificates'].values()),
        new_endpoint_solves=sum(c.get('newly_evaluated_endpoints',0) for r in records for c in r['certificates'].values()),
        cached_baseline_certificate_source=identity['roots']['baseline'],
        cached_baseline_weights_recomputed_and_compared=True)
    exact.atomic_json(Path(output)/'summary.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    for name in ('pilot','full500','summarize'):action.add_argument('--'+name,action='store_true')
    parser.add_argument('--replications',default='1,3',help='Pilot only; full500 is fixed 1..500')
    for name in ('simulation-results','statistics-root','records-root','baseline-exact-root','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    if not 1<=args.workers<=12:parser.error('workers must be 1..12; one numerical thread per worker')
    output=args.output.resolve()
    for root in (args.simulation_results,args.statistics_root,args.records_root,args.baseline_exact_root):
        if output==root.resolve() or output.is_relative_to(root.resolve()):parser.error('Output must be outside immutable inputs')
    if args.summarize:
        identity=exact.read_json(output/'run_identity.json')
        if exact.digest({k:v for k,v in identity.items() if k!='identity_sha256'})!=identity['identity_sha256']:raise ValueError('Invalid run identity')
        configure(identity);exact.verify_hashes(identity['source_sha256']);exact.verify_hashes(identity['baseline_sha256'])
        for inputs in identity['inputs'].values():exact.verify_hashes(inputs['input_sha256'])
        aggregate(output,identity);return
    reps=sorted({int(v) for v in args.replications.split(',')}) if args.pilot else list(range(1,501))
    if not reps or any(r not in range(1,501) for r in reps):parser.error('Pilot replications must lie in 1..500')
    identity=make_identity(args,reps);output.mkdir(parents=True,exist_ok=True)
    exact.bind_identity(output,identity)
    frozen=output/'frozen_protocol.json'
    if frozen.exists() and exact.read_json(frozen)!=identity['protocol']:raise ValueError('Frozen protocol differs')
    exact.atomic_json(frozen,identity['protocol'])
    states=exact.precision.legal_partitions();cuts=np.zeros((len(states),4),dtype=np.int16)
    for i,s in enumerate(states):cuts[i,:len(s)]=s
    order=output/'partition_order.npz'
    if order.exists():
        with np.load(order,allow_pickle=False) as z:
            if not np.array_equal(z['cuts'],cuts) or not np.array_equal(z['k'],list(map(len,states))):raise ValueError('Partition order differs')
    else:exact.atomic_npz(order,cuts=cuts,k=np.array(list(map(len,states))))
    todo=[r for r in reps if validate_result(output,r,identity) is None]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(compute_case,r,str(output),identity) for r in todo]
        for future in futures:
            item=future.result();print(exact.precision.native(dict(replication=item['replication'],seconds=item['seconds'],baseline=item['baseline_validation'])),flush=True)
    exact.verify_hashes(identity['source_sha256'])
    exact.verify_hashes(identity['baseline_sha256'])
    for inputs in identity['inputs'].values():exact.verify_hashes(inputs['input_sha256'])
    aggregate(output,identity)


if __name__=='__main__':main()
