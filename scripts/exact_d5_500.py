"""Resumable exact intervals for the fixed, unchanged D5 replications 1--500.

--preflight validates every original/cache identity without solving intervals.
--full computes the full fixed set. --summarize requires all 500 valid outputs.
--smoke computes only fixed cases 1 and 3 in a separately labelled directory.
No simulation, model fitting, MCMC, native DLL, or original-result writes occur.
Without --simulation-results, --smoke uses a tiny synthetic mixture only.
Full input roots are explicit, read-only archives; no private analysis imports.
"""
from __future__ import annotations
import os
THREAD_ENV = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')
for _key in THREAD_ENV: os.environ[_key]='1'
os.environ['OMP_DYNAMIC']='FALSE'
import argparse, csv, hashlib, json, multiprocessing, platform, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import scipy
from threadpoolctl import threadpool_limits, threadpool_info
if __package__:
    from . import precision
else:
    import precision
from scipy.special import stdtr, logsumexp
from scipy.optimize import brentq
from types import SimpleNamespace

def parameters(case, states):
    """Reuse the reviewed public scalar posterior; never truncate support."""
    loc = np.empty((len(states),64)); scales = np.empty_like(loc); logw = np.empty(len(states))
    for j,cuts in enumerate(states):
        lw,mu,diag,rate,shape = precision.partition_posterior_parameters(case['H'],case['h'],case['yqy'],case['n'],cuts)
        if rate <= 0 or not np.isfinite(lw): raise ValueError('Invalid conditional posterior')
        logw[j]=lw; sc=precision.student_t_scale(rate,shape,diag)
        edges=(0,)+cuts+(64,)
        for s,(a,b) in enumerate(zip(edges[:-1],edges[1:])):
            loc[j,a:b]=mu[s]; scales[j,a:b]=sc[s]
    weights=np.exp(logw-logsumexp(logw))
    return dict(weights=weights,locations=loc,scales=scales,df=2*shape)

core=SimpleNamespace(ROOT=Path(__file__).resolve().parents[1],p=precision,f=precision.formal,
    SOURCE=Path(__file__).resolve().parents[1]/'results/reproduction/simulation',
    sha=precision.sha256_file,parameters=parameters)
LOCATION_TOLERANCE=1e-8
CDF_SIGN_MARGIN=1e-13

class EndpointCertificationError(RuntimeError):
    def __init__(self,message,details):
        self.details=details
        super().__init__(message,details)

def quantiles(ex, auxiliary_mass=1e-4):
    w=ex['weights']; df=ex['df']
    loc=np.ascontiguousarray(ex['locations'].T); scale=np.ascontiguousarray(ex['scales'].T)
    if w.ndim!=1 or loc.shape!=(64,len(w)) or scale.shape!=loc.shape: raise ValueError('Invalid mixture dimensions')
    if not np.isfinite(w).all() or np.any(w<0) or not np.isclose(w.sum(),1.,rtol=0,atol=1e-12): raise ValueError('Invalid weights')
    if not np.isfinite(loc).all() or not np.isfinite(scale).all() or np.any(scale<=0) or not np.isfinite(df) or df<=0: raise ValueError('Invalid Student-t parameters')
    if not 0<auxiliary_mass<.025: raise ValueError('Auxiliary mass must lie in (0,.025)')
    order=np.argsort(w); omitted_count=np.searchsorted(np.cumsum(w[order]),auxiliary_mass)
    active=order[omitted_count:]; omitted=w[order[:omitted_count]].sum()
    wa=w[active]; la=loc[:,active]; sa=scale[:,active]
    roots=np.empty((2,64)); residuals=np.empty_like(roots)
    full_calls=partial_calls=0; max_bracket_width=0.
    minimum_lower_margin=minimum_upper_margin=float('inf')
    for t in range(64):
        def partial(x): return np.sum(wa*stdtr(df,(x-la[t])/sa[t]))
        cache={}
        def full(x):
            if x not in cache: cache[x]=np.sum(w*stdtr(df,(x-loc[t])/scale[t]))
            return cache[x]
        lo=loc[t].min()-16*scale[t].max(); hi=loc[t].max()+16*scale[t].max()
        if not partial(lo)+omitted<.025 or not partial(hi)>.975:raise ValueError('Initial mixture bracket does not enclose both target probabilities')
        for j,p in enumerate((.025,.975)):
            qa,ia=brentq(lambda x:partial(x)-(p-omitted),lo,hi,xtol=1e-12,rtol=1e-14,full_output=True)
            qb,ib=brentq(lambda x:partial(x)-p,lo,hi,xtol=1e-12,rtol=1e-14,full_output=True)
            # Covers auxiliary root rounding; signs checked on full sums below.
            qa-=1e-10; qb+=1e-10
            if not full(qa)<=p<=full(qb):raise ValueError('Full-CDF auxiliary bracket failed')
            q,info=brentq(lambda x:full(x)-p,qa,qb,xtol=1e-12,rtol=1e-14,full_output=True)
            if not info.converged:raise RuntimeError('Full-CDF solver did not converge')
            roots[j,t]=q; residuals[j,t]=abs(full(q)-p)
            lower_margin=p-full(q-LOCATION_TOLERANCE)
            upper_margin=full(q+LOCATION_TOLERANCE)-p
            if not lower_margin>CDF_SIGN_MARGIN or not upper_margin>CDF_SIGN_MARGIN:
                raise EndpointCertificationError('Full-mixture endpoint location could not be certified',dict(
                    bin=t+1,probability=p,endpoint=float(q),lower_cdf=float(full(q-LOCATION_TOLERANCE)),upper_cdf=float(full(q+LOCATION_TOLERANCE)),
                    lower_margin=float(lower_margin),upper_margin=float(upper_margin),location_tolerance=LOCATION_TOLERANCE,
                    required_cdf_sign_margin=CDF_SIGN_MARGIN,full_support_count=len(w),cdf_residual=float(residuals[j,t])))
            minimum_lower_margin=min(minimum_lower_margin,lower_margin)
            minimum_upper_margin=min(minimum_upper_margin,upper_margin)
            partial_calls+=ia.function_calls+ib.function_calls
            max_bracket_width=max(max_bracket_width,qb-qa)
        full_calls+=len(cache)
    if not np.isfinite(roots).all() or np.any(roots[0]>roots[1]) or residuals.max()>=1e-8: raise RuntimeError('Invalid full-mixture roots')
    return roots,dict(full_support_count=len(w),auxiliary_support_count=len(active),auxiliary_omitted_mass=float(omitted),
        max_auxiliary_bracket_width=max_bracket_width,max_full_cdf_residual=float(residuals.max()),full_cdf_calls=full_calls,
        partial_cdf_calls=partial_calls,final_solver='Brent full sum over every partition',xtol=1e-12,rtol=1e-14,
        endpoint_location_certified=True,endpoint_location_tolerance=LOCATION_TOLERANCE,required_cdf_sign_margin=CDF_SIGN_MARGIN,
        minimum_lower_cdf_margin=float(minimum_lower_margin),minimum_upper_cdf_margin=float(minimum_upper_margin))

ROOT, BASE = core.ROOT, core.ROOT/'results/reproduction/exact_d5'
CACHE=None
CACHE_RECORDS=None
ORIGINAL=core.SOURCE/'replicates/main/D5'
CUTS=np.array([13,25,39,51]); GRID=np.arange(1,65)
TRUTH=np.array([.13125,.03125,.13125,.03125,.13125])[np.searchsorted(CUTS,GRID,side='left')]
NEAR=np.min(abs(GRID[:,None]-CUTS[None,:]),axis=1)<=2
assert NEAR.sum()==20 and (~NEAR).sum()==44
SCHEMA=1

def read_json(path): return json.loads(Path(path).read_text(encoding='utf8'))
def digest(obj): return hashlib.sha256(json.dumps(core.p.native(obj),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def relative(path): return str(Path(path).resolve())
def utc(): return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(core.p.native(obj),indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf8')
    os.replace(temp,path)
def atomic_npz(path,**arrays):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('wb') as stream: np.savez_compressed(stream,**arrays)
    os.replace(temp,path)
def write_csv(path,rows):
    path=Path(path)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('w',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    os.replace(temp,path)

def source_hashes():
    paths=[Path(__file__),ROOT/'scripts/precision.py',ROOT/'scripts/simulation.py',
           core.SOURCE/'resolved_config.json']
    return {str(p.resolve()):core.sha(p) for p in paths}


def verify_hashes(hashes):
    for rel,expected in hashes.items():
        if core.sha(ROOT/rel)!=expected: raise RuntimeError(f'Input/source changed: {rel}')

def d5_namespace(cfg,case):
    """Use the public per-DGP namespace while accepting historical archives."""
    helper=getattr(core.f,'case_namespace',None)
    return helper(cfg,case) if helper is not None else cfg.get('namespace_by_dgp',{}).get(case['dgp'],cfg['namespace'])


def case_inputs(rep):
    if rep not in range(1,501):raise ValueError('replication must be in 1..500')
    stem=f'rep_{rep:06d}'
    original=ORIGINAL/stem/'summary.json';curve=original.with_name('a3_bma_curve.csv')
    legacy=(CACHE/(stem+'.npz'),CACHE_RECORDS/(stem+'.json'))
    reconstructed=(CACHE/f'R{rep:06d}.npz',CACHE_RECORDS/f'R{rep:06d}.json')
    if all(p.is_file() for p in legacy):cache,record_path=legacy;schema='legacy_or_public_bridge'
    elif all(p.is_file() for p in reconstructed):cache,record_path=reconstructed;schema='public_reconstructed'
    else:raise FileNotFoundError(f'Missing hash-matched sufficient statistics for {rep}; supply legacy/public bridge rep_NNNNNN NPZ+JSON roots or public reconstructed_inputs/primary/D5 RNNNNNN NPZ+JSON roots. Original summary and draw-quantile CSV are also required; MCMC will not be rerun.')
    paths=[original,curve,cache,record_path]
    if not all(p.is_file() for p in paths):raise FileNotFoundError(f'Missing original summary or saved draw quantiles for {rep}: {paths}')
    hashes={relative(p):core.sha(p) for p in paths}
    old=read_json(original);record=read_json(record_path);case=old['case']
    cfg=read_json(core.SOURCE/'resolved_config.json')['config'];namespace=d5_namespace(cfg,case)
    if old['replicate']!=rep:raise ValueError('Original replication identity differs')
    if old['data_seed']!=core.f.deterministic_seed(namespace,f'DATA|main|D5|R{rep:06d}'):raise ValueError('Original data seed differs from fixed replication')
    expected={'dgp':'D5','stage':'main','cell':'D5','kmax':4,'min_segment':8,'prior_factorial_power':1.,'participants':500,'signal_multiplier':1.,'training_mcar':0.}
    for key,value in expected.items():
        if case.get(key)!=value:raise ValueError(f'Original case configuration mismatch {rep}: {key}')
    if old['truth']!={'cuts':CUTS.tolist(),'levels':[.13125,.03125,.13125,.03125,.13125]}:raise ValueError('Original truth differs')
    if schema=='legacy_or_public_bridge':
        if record['replicate']!=rep or record['dgp']!='D5' or record['namespace']!=namespace:raise ValueError('Cache dataset/namespace differs')
        if record['source_summary_hash']!=hashes[relative(original)]:raise ValueError('Cache record original summary hash differs')
        code_hash=record['code_hash']
    else:
        if (record['cohort'],record['cell'],record['rep'])!=('primary','D5',rep):raise ValueError('Reconstructed dataset identity differs')
        if record['source_sha256']!=hashes[relative(original)] or record['cache_sha256']!=hashes[relative(cache)]:raise ValueError('Reconstructed source/statistics fingerprint differs')
        code_hash=record['reconstruction_code_sha256']
    with np.load(cache,allow_pickle=False) as z:
        if schema=='legacy_or_public_bridge':
            for key in ('data_hash','holdout_hash','training_mask_hash'):
                if z[key].item()!=old[key] or record[key]!=old[key]:raise ValueError(f'Cache identity mismatch {rep}: {key}')
            if z['source_summary_hash'].item()!=hashes[relative(original)] or z['code_hash'].item()!=code_hash:raise ValueError('Cache provenance differs')
        elif z['source_sha256'].item()!=hashes[relative(original)]:raise ValueError('Reconstructed NPZ original summary fingerprint differs')
        if 'n' in z and int(z['n'])!=old['n_train']:raise ValueError('Cached training count differs')
        H=z['H'].copy();h=z['h'].copy();yqy=float(z['yqy'])
    if H.shape!=(64,64) or h.shape!=(64,) or not np.isfinite(H).all() or not np.isfinite(h).all() or not np.isfinite(yqy):raise ValueError('Invalid sufficient statistics')
    if not np.allclose(H,H.T,rtol=1e-10,atol=1e-8):raise ValueError('Sufficient-statistic H is asymmetric')
    if old['n_train']!=25500 or old['n_test']!=6500:raise ValueError('Training/test count mismatch')
    for key in ('n_train','n_test'):
        if key in record and record[key]!=old[key]:raise ValueError('Cache observation count differs')
    with curve.open(newline='',encoding='utf-8-sig') as stream:rows=list(csv.DictReader(stream))
    if [int(r['trial']) for r in rows]!=GRID.tolist():raise ValueError('Original curve position indexing differs')
    original_q=np.array([[float(r[key]) for r in rows] for key in ('q025','q975')])
    original_mean=np.array([float(r['estimate']) for r in rows])
    if not np.isfinite(original_q).all() or not np.isfinite(original_mean).all() or np.any(original_q[0]>original_q[1]):raise ValueError('Invalid original intervals')
    np.testing.assert_allclose(original_q,np.array([old['A3']['bma_q025'],old['A3']['bma_q975']]),rtol=0,atol=1e-14)
    inputs=dict(replication=rep,input_schema=schema,input_sha256=hashes,data_seed=old['data_seed'],namespace=namespace,
        **{key:old[key] for key in ('data_hash','holdout_hash','training_mask_hash','n_train','n_test','run_fingerprint')},cache_generation_code_hash=code_hash)
    return inputs,dict(H=H,h=h,yqy=yqy,n=old['n_train']),original_q,original_mean


def make_identity(reps,mode):
    cfg=read_json(core.SOURCE/'resolved_config.json')['config']
    if d5_namespace(cfg,{'dgp':'D5'})!='R61-D5-BALANCED-v1' or cfg['nominal_model']!={'kmax':4,'min_segment':8,'prior_factorial_power':1.}:raise ValueError('Original D5 seed namespace/model protocol differs')
    if cfg['dgps']['D5']!={'cuts':CUTS.tolist(),'levels':[.13125,.03125,.13125,.03125,.13125]}:raise ValueError('Original D5 data-generating process differs')
    all_inputs={str(rep):case_inputs(rep)[0] for rep in reps}
    identity=dict(roots=dict(original=str(ORIGINAL),statistics=str(CACHE),records=str(CACHE_RECORDS)),schema_version=SCHEMA,mode=mode,replications=list(reps),source_sha256=source_hashes(),inputs=all_inputs,
        model=dict(Kmax=4,minseg=8,prior_factorial_power=1.,delta_prior_precision=4.,ig_shape=3.,ig_scale=2.),
        numerical=dict(quantiles=[.025,.975],xtol=1e-12,rtol=1e-14,full_cdf_residual_max=1e-8,auxiliary_bracket_mass=1e-4,final_support=27931,
            endpoint_location_tolerance=LOCATION_TOLERANCE,required_cdf_sign_margin=CDF_SIGN_MARGIN),
        truth=TRUTH.tolist(),near_positions=GRID[NEAR].tolist(),far_positions=GRID[~NEAR].tolist(),
        versions=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__))
    identity['identity_sha256']=digest(identity)
    return identity

def bind_identity(output,identity):
    path=output/'run_identity.json'
    if path.exists():
        if read_json(path)!=identity: raise RuntimeError('Resume refused: source/input/protocol/software identity differs. Use a new output directory.')
    else: atomic_json(path,identity)

def result_paths(output,rep):
    return output/'replicates'/f'rep_{rep:06d}.json',output/'replicates'/f'rep_{rep:06d}.npz'

def validate_certificate(cert,identity):
    """Validate provenance-bound location and CDF tolerances, without new CDF work."""
    protocol=identity['numerical']
    location=float(protocol['endpoint_location_tolerance']);margin=float(protocol['required_cdf_sign_margin'])
    residual_limit=float(protocol.get('full_cdf_residual_max',1e-8))
    values=[location,margin,residual_limit]+[float(cert[k]) for k in ('endpoint_location_tolerance','required_cdf_sign_margin','max_full_cdf_residual','minimum_lower_cdf_margin','minimum_upper_cdf_margin')]
    if not np.isfinite(values).all():raise ValueError('Nonfinite exact endpoint certificate')
    if not 0<location<=LOCATION_TOLERANCE or margin<CDF_SIGN_MARGIN or not 0<residual_limit<=1e-8:raise ValueError('Archived numerical protocol has inadequate endpoint tolerances')
    if cert.get('full_support_count')!=27931 or cert.get('endpoint_location_certified') is not True:raise ValueError('Invalid full-support endpoint certificate')
    if not 0<float(cert['endpoint_location_tolerance'])<=location or float(cert['required_cdf_sign_margin'])<margin:raise ValueError('Endpoint certificate does not satisfy recorded numerical protocol')
    if not 0<=float(cert['max_full_cdf_residual'])<residual_limit:raise ValueError('Invalid full-mixture root residual')
    if min(float(cert['minimum_lower_cdf_margin']),float(cert['minimum_upper_cdf_margin']))<=float(cert['required_cdf_sign_margin']):raise ValueError('Invalid strict full-CDF crossing margins')


def validate_result(output,rep,identity):
    path,npz=result_paths(output,rep)
    if not path.exists(): return None
    item=read_json(path)
    if item['identity_sha256']!=identity['identity_sha256'] or item['inputs']!=identity['inputs'][str(rep)]: raise RuntimeError(f'Resume refused: case {rep} identity differs')
    if item.get('status')!='complete' or not npz.is_file() or core.sha(npz)!=item['npz_sha256']: raise RuntimeError(f'Resume refused: case {rep} output is missing/corrupt')
    validate_certificate(item['checks'],identity)
    with np.load(npz,allow_pickle=False) as z:
        for key in ('exact_q025','exact_q975','original_q025','original_q975','exact_mean','original_mean','pips','truth'):
            if z[key].shape!=(64,) or not np.isfinite(z[key]).all(): raise RuntimeError(f'Invalid saved array {key}')
        if z['partition_weights'].shape!=(27931,) or np.any(z['partition_weights']<0) or not np.isclose(z['partition_weights'].sum(),1.,atol=1e-12,rtol=0): raise RuntimeError('Invalid saved partition weights')
        if not np.array_equal(z['truth'],TRUTH) or np.any(z['exact_q025']>z['exact_q975']): raise RuntimeError('Saved truth/interval inconsistency')
        if str(z['identity_sha256'].item())!=identity['identity_sha256']: raise RuntimeError('NPZ identity mismatch')
    return item

_IDENTITY=_STATES=_COUNTS=_CUT_ARRAY=_THREAD_LIMITER=None
def worker_init(identity):
    global ORIGINAL, CACHE, CACHE_RECORDS
    ORIGINAL=Path(identity['roots']['original'])
    CACHE=Path(identity['roots']['statistics'])
    CACHE_RECORDS=Path(identity['roots']['records'])
    global _IDENTITY,_STATES,_COUNTS,_CUT_ARRAY,_THREAD_LIMITER
    _THREAD_LIMITER=threadpool_limits(limits=1)
    if any(pool['num_threads']!=1 for pool in threadpool_info()): raise RuntimeError('A numerical thread pool is not limited to one thread')
    verify_hashes(identity['source_sha256'])
    _IDENTITY=identity; _STATES=core.p.legal_partitions(); _COUNTS=np.array(list(map(len,_STATES)))
    if len(_STATES)!=27931 or len(set(_STATES))!=27931 or np.bincount(_COUNTS,minlength=5).tolist()!=[1,49,861,6545,20475]: raise RuntimeError('Partition enumeration changed')
    _CUT_ARRAY=np.zeros((len(_STATES),4),dtype=np.int16)
    for i,state in enumerate(_STATES): _CUT_ARRAY[i,:len(state)]=state

def compute_case(rep,output):
    output=Path(output); started=time.perf_counter()
    inputs,case,oq,om=case_inputs(rep)
    if inputs!=_IDENTITY['inputs'][str(rep)]: raise RuntimeError(f'Input changed since preflight for {rep}')
    ex=core.parameters(case,_STATES); parameter_seconds=time.perf_counter()-started
    root_start=time.perf_counter()
    try: q,checks=quantiles(ex)
    except EndpointCertificationError as exc:
        atomic_json(output/'unresolved'/f'rep_{rep:06d}.json',dict(status='unresolved_endpoint_location',replication=rep,
            identity_sha256=_IDENTITY['identity_sha256'],inputs=inputs,certificate_failure=exc.details,recorded_utc=utc(),
            excluded_from_planned_set=False,required_action='Resolve with stable-tail arithmetic or higher-precision CDF; this case must not be excluded from the 500-case result'))
        raise
    root_seconds=time.perf_counter()-root_start
    w=ex['weights']; pk=np.bincount(_COUNTS,weights=w,minlength=5)
    pips=np.array([w[np.any(_CUT_ARRAY==t,axis=1)].sum() for t in GRID])
    if abs(pk.sum()-1)>1e-12 or abs(pips.sum()-np.arange(5)@pk)>1e-10: raise RuntimeError('Probability identities failed')
    verify_hashes(inputs['input_sha256'])
    path,npz=result_paths(output,rep)
    atomic_npz(npz,exact_q025=q[0],exact_q975=q[1],exact_mean=w@ex['locations'],original_q025=oq[0],original_q975=oq[1],original_mean=om,
        truth=TRUTH,near=NEAR,p_k=pk,pips=pips,partition_weights=w,identity_sha256=np.array(_IDENTITY['identity_sha256']))
    item=dict(status='complete',replication=rep,identity_sha256=_IDENTITY['identity_sha256'],inputs=inputs,npz_sha256=core.sha(npz),
        checks=checks,df=ex['df'],support_counts_by_k=np.bincount(_COUNTS,minlength=5),
        numerically_nonzero_weights=int((w>0).sum()),p_k=pk,pips=pips,exact_minus_original_max_endpoint_difference=float(abs(q-oq).max()),
        seconds=dict(cache_load_and_parameters=parameter_seconds,root_inversion=root_seconds,total=time.perf_counter()-started),
        peak_working_set_bytes=core.f.process_peak_working_set(),worker_pid=os.getpid(),thread_environment={k:os.environ[k] for k in THREAD_ENV},
        threadpools=threadpool_info(),finished_utc=utc(),new_chains=0)
    atomic_json(path,item)
    return item

def mcse(values):
    values=np.asarray(values,float)
    return float(values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else None

def summarize_arrays(exact,original,truth,near):
    """Each pair is (replication, endpoint, position); MCSE clusters by dataset."""
    exact=np.asarray(exact); original=np.asarray(original); truth=np.asarray(truth); near=np.asarray(near,bool)
    if exact.shape!=original.shape or exact.ndim!=3 or exact.shape[1]!=2 or exact.shape[2]!=len(truth): raise ValueError('Interval array shapes differ')
    if not np.isfinite(exact).all() or not np.isfinite(original).all() or np.any(exact[:,0]>exact[:,1]) or np.any(original[:,0]>original[:,1]): raise ValueError('Invalid interval values')
    n=len(exact); strata=[]; paired=[]; pointwise=[]; per_dataset=[]; method_arrays={}
    masks={'all':np.ones(len(truth),bool),'near_le_2':near,'far_gt_2':~near}
    if not near.any() or near.all(): raise ValueError('Both strata must be nonempty')
    for method,arr in (('exact_mixture',exact),('original_draw_quantile',original)):
        cover=(arr[:,0]<=truth)&(truth<=arr[:,1]); width=arr[:,1]-arr[:,0]; method_arrays[method]=(cover,width)
        for label,mask in masks.items():
            c=cover[:,mask].mean(1); w=width[:,mask].mean(1)
            strata.append(dict(method=method,stratum=label,positions=int(mask.sum()),replications=n,coverage=float(c.mean()),coverage_mcse=mcse(c),mean_width=float(w.mean()),width_mcse=mcse(w)))
            per_dataset.extend(dict(replication_index=i+1,method=method,stratum=label,coverage=float(c[i]),width=float(w[i])) for i in range(n))
        for metric,values in (('coverage',cover),('width',width)):
            diff=values[:,near].mean(1)-values[:,~near].mean(1)
            paired.append(dict(contrast=f'{method}: near_minus_far',stratum='near_minus_far',metric=metric,pairs=n,mean_difference=float(diff.mean()),paired_mcse=mcse(diff)))
        pointwise.extend(dict(method=method,bin=t+1,near_boundary=bool(near[t]),replications=n,coverage=float(cover[:,t].mean()),coverage_mcse=mcse(cover[:,t]),mean_width=float(width[:,t].mean()),width_mcse=mcse(width[:,t])) for t in range(len(truth)))
        simultaneous=cover.all(1)
        strata.append(dict(method=method,stratum='simultaneous_all64',positions=len(truth),replications=n,coverage=float(simultaneous.mean()),coverage_mcse=mcse(simultaneous),mean_width=None,width_mcse=None))
    ec,ew=method_arrays['exact_mixture']; oc,ow=method_arrays['original_draw_quantile']
    for label,mask in masks.items():
        for metric,delta in (('coverage',ec.astype(float)-oc),('width',ew-ow)):
            diff=delta[:,mask].mean(1)
            paired.append(dict(contrast='exact_minus_original',stratum=label,metric=metric,pairs=n,mean_difference=float(diff.mean()),paired_mcse=mcse(diff)))
    d=ec.all(1).astype(float)-oc.all(1)
    paired.append(dict(contrast='exact_minus_original',stratum='simultaneous_all64',metric='coverage',pairs=n,mean_difference=float(d.mean()),paired_mcse=mcse(d)))
    return dict(strata=strata,paired=paired,pointwise=pointwise,per_dataset=per_dataset,
        exact_coverage=ec,original_coverage=oc,exact_width=ew,original_width=ow)

def aggregate(output,identity):
    reps=identity['replications']; exact=[]; original=[]; records=[]
    if identity['mode']=='formal500' and reps!=list(range(1,501)): raise RuntimeError('Formal aggregation requires exactly replications 1..500')
    for rep in reps:
        item=validate_result(output,rep,identity)
        if item is None: raise RuntimeError(f'Cannot aggregate incomplete set, missing {rep}')
        records.append(item)
        with np.load(result_paths(output,rep)[1],allow_pickle=False) as z:
            exact.append(np.array([z['exact_q025'],z['exact_q975']]))
            original.append(np.array([z['original_q025'],z['original_q975']]))
    result=summarize_arrays(exact,original,TRUTH,NEAR)
    for row in result['per_dataset']: row['replication']=reps[row.pop('replication_index')-1]
    for name in ('strata','paired','pointwise','per_dataset'): write_csv(output/f'{name}.csv',result[name])
    atomic_npz(output/'coverage_and_width_arrays.npz',replications=np.array(reps),truth=TRUTH,near=NEAR,
        **{name:result[name] for name in ('exact_coverage','original_coverage','exact_width','original_width')})
    summary=dict(identity_sha256=identity['identity_sha256'],scope=identity['mode'],complete_replications=len(reps),strata=result['strata'],paired=result['paired'],
        max_full_cdf_residual=max(r['checks']['max_full_cdf_residual'] for r in records),all_partition_support=27931,
        endpoint_location_tolerance=LOCATION_TOLERANCE,required_cdf_sign_margin=CDF_SIGN_MARGIN,
        minimum_endpoint_cdf_margin=min(min(r['checks']['minimum_lower_cdf_margin'],r['checks']['minimum_upper_cdf_margin']) for r in records),
        max_peak_working_set_bytes=max(r['peak_working_set_bytes'] or 0 for r in records),
        maximum_exact_minus_original_endpoint_difference=max(r['exact_minus_original_max_endpoint_difference'] for r in records),
        mean_case_seconds=float(np.mean([r['seconds']['total'] for r in records])),new_chains=0,
        minimum_pointwise_coverage={m:float(min(r['coverage'] for r in result['pointwise'] if r['method']==m)) for m in ('exact_mixture','original_draw_quantile')},
        mcse_definition='sample SD across dataset-level averages/differences divided by sqrt(number of datasets); positions are not independent replications')
    discrepancies=np.array([r['exact_minus_original_max_endpoint_difference'] for r in records])
    write_csv(output/'endpoint_discrepancies.csv',[dict(replication=rep,max_endpoint_absolute_difference=float(d)) for rep,d in zip(reps,discrepancies)])
    summary['datasets_max_endpoint_difference_gt_0p002']=int((discrepancies>.002).sum())
    summary['endpoint_difference_quantiles']=dict(median=float(np.median(discrepancies)),q95=float(np.quantile(discrepancies,.95)),maximum=float(discrepancies.max()))
    atomic_json(output/'summary.json',summary)
    return summary

def reaggregate_cache(archive,output):
    """Verify saved certificates/arrays/input hashes; does NOT evaluate roots anew."""
    archive=archive.resolve();output=output.resolve()
    if output==archive or output.is_relative_to(archive):raise ValueError('Output must be outside immutable exact archive')
    identity=read_json(archive/'run_identity.json')
    if digest({k:v for k,v in identity.items() if k!='identity_sha256'})!=identity['identity_sha256']:raise ValueError('Archived run identity fingerprint is invalid')
    reps=identity['replications']
    if reps!=list(range(1,501)):raise ValueError('Cached formal result must contain exactly datasets 1..500')
    exact=[];original=[];records=[];hashes={}
    for rep in reps:
        stem=f'rep_{rep:06d}';jp=archive/'replicates'/(stem+'.json');npz=jp.with_suffix('.npz')
        item=read_json(jp)
        if item['identity_sha256']!=identity['identity_sha256'] or item['inputs']!=identity['inputs'][str(rep)]:raise ValueError('Cached dataset identity mismatch')
        if item['status']!='complete' or core.sha(npz)!=item['npz_sha256']:raise ValueError('Corrupt cached dataset')
        verify_hashes(item['inputs']['input_sha256'])
        checks=item['checks']
        validate_certificate(checks,identity)
        with np.load(npz,allow_pickle=False) as z:
            eq=np.array([z['exact_q025'],z['exact_q975']]);oq=np.array([z['original_q025'],z['original_q975']])
            if str(z['identity_sha256'].item())!=identity['identity_sha256'] or not np.array_equal(z['truth'],TRUTH) or not np.array_equal(z['near'],NEAR):raise ValueError('Cached NPZ identity/truth mismatch')
            w=z['partition_weights']
            if w.shape!=(27931,) or not np.isfinite(w).all() or np.any(w<0) or abs(w.sum()-1)>1e-12:raise ValueError('Cached posterior weights invalid')
        # Compare directly with unchanged saved main-fit endpoints, not a reconstruction.
        curve_paths=[ROOT/path for path in item['inputs']['input_sha256'] if Path(path).name=='a3_bma_curve.csv']
        if len(curve_paths)!=1:raise ValueError('Cached input record lacks original curve')
        with curve_paths[0].open(encoding='utf-8-sig',newline='') as f:curve=list(csv.DictReader(f))
        reference=np.array([[float(r[k]) for r in curve] for k in ('q025','q975')])
        if not np.array_equal(oq,reference):raise ValueError('Cached original quantiles differ from original CSV')
        exact.append(eq);original.append(oq);records.append(item)
        hashes[str(jp)]=core.sha(jp);hashes[str(npz)]=core.sha(npz)
    result=summarize_arrays(exact,original,TRUTH,NEAR)
    for row in result['per_dataset']:row['replication']=row.pop('replication_index')
    output.mkdir(parents=True,exist_ok=True)
    fingerprint=digest(dict(inputs=hashes,algorithm=core.sha(Path(__file__))))
    manifest=output/'cache_reaggregation_manifest.json'
    if manifest.exists() and read_json(manifest)['fingerprint']!=fingerprint:raise ValueError('Cache output identity differs; use a new output directory')
    for name in ('strata','paired','pointwise','per_dataset'):write_csv(output/(name+'.csv'),result[name])
    differences=np.max(np.abs(np.array(exact)-np.array(original)),axis=(1,2))
    write_csv(output/'endpoint_discrepancies.csv',[dict(replication=rep,max_endpoint_absolute_difference=float(d)) for rep,d in zip(reps,differences)])
    summary=dict(scope='verified_archived_cache_reaggregation',root_evaluations_performed=False,new_MCMC_runs=0,complete_replications=500,all_partition_support=27931,strata=result['strata'],paired=result['paired'],datasets_max_endpoint_difference_gt_0p002=int((differences>.002).sum()),endpoint_difference_quantiles=dict(median=float(np.median(differences)),q95=float(np.quantile(differences,.95)),maximum=float(differences.max())),max_full_cdf_residual=max(r['checks']['max_full_cdf_residual'] for r in records),minimum_endpoint_cdf_margin=min(min(r['checks']['minimum_lower_cdf_margin'],r['checks']['minimum_upper_cdf_margin']) for r in records))
    atomic_json(output/'summary.json',summary)
    atomic_json(manifest,dict(fingerprint=fingerprint,archive=str(archive),source_sha256=hashes,historical_algorithm_identity=identity['identity_sha256'],root_evaluations_performed=False,input_hashes_verified=True,original_quantiles_verified=True))
    print(json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    for name in ('preflight','full','summarize','smoke','reaggregate-cache'): action.add_argument('--'+name,action='store_true')
    parser.add_argument('--cached-exact-root',type=Path,help='Archived all500 root for --reaggregate-cache; verifies saved certificates but does not re-evaluate CDFs')
    parser.add_argument('--simulation-results',type=Path,help='Original D5 run root; resolved_config.json and replicates/main/D5 required')
    parser.add_argument('--statistics-root',type=Path,help='Immutable NPZ directory: statistics/D5 rep_NNNNNN.npz or reconstructed_inputs/primary/D5 RNNNNNN.npz')
    parser.add_argument('--records-root',type=Path,help='Matching immutable statistic provenance JSON directory')
    parser.add_argument('--preflight-replications',help='Comma-separated fixed replication IDs for bounded --preflight only; never changes the formal 1..500 computation set')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--max-wall-minutes',type=float,default=120.,help='Stop scheduling after this budget; completed cases remain resumable')
    args=parser.parse_args()
    if not __debug__: parser.error('Do not run with Python -O: prototype verification assertions must remain enabled')
    if not 1<=args.workers<=12: parser.error('workers must be between 1 and 12')
    global ORIGINAL,CACHE,CACHE_RECORDS
    if args.reaggregate_cache:
        if args.cached_exact_root is None or args.output is None:parser.error('--reaggregate-cache requires --cached-exact-root and --output')
        reaggregate_cache(args.cached_exact_root,args.output);return
    if args.smoke and args.simulation_results is None:
        output=(args.output or BASE/'smoke_fixture').resolve()
        output.mkdir(parents=True,exist_ok=True)
        mixture=dict(weights=np.array([.3,.7]),locations=np.array([[0.]*64,[1.]*64]),scales=np.full((2,64),.15),df=20.)
        q,checks=quantiles(mixture)
        atomic_npz(output/'fixture_endpoints.npz',q025=q[0],q975=q[1])
        atomic_json(output/'summary.json',dict(scope='synthetic_fixture_only',checks=checks,new_chains=0,manuscript_result=False))
        print('PASS synthetic fixture full-CDF inversion and crossing certificate'); return
    if any(x is None for x in (args.simulation_results,args.statistics_root,args.records_root)):
        parser.error('Supply --simulation-results, --statistics-root and --records-root. Original saved draw quantiles and hash-matched sufficient statistics are required; this entry never reruns MCMC or fabricates original draws.')
    core.SOURCE=args.simulation_results.resolve()
    ORIGINAL=core.SOURCE/'replicates/main/D5'; CACHE=args.statistics_root.resolve(); CACHE_RECORDS=args.records_root.resolve()
    mode='smoke_fixed_1_3' if args.smoke else 'formal500'; reps=[1,3] if args.smoke else list(range(1,501))
    if args.preflight_replications:
        if not args.preflight:parser.error('--preflight-replications is permitted only with --preflight')
        try:reps=sorted({int(value) for value in args.preflight_replications.split(',')})
        except ValueError:parser.error('--preflight-replications requires comma-separated integers in 1..500')
        if not reps or any(rep not in range(1,501) for rep in reps):parser.error('--preflight-replications requires IDs in 1..500')
        mode='bounded_input_preflight'
    output=(args.output or BASE/('smoke_runner' if args.smoke else 'all500')).resolve()
    for input_root in (core.SOURCE,CACHE,CACHE_RECORDS):
        if output==input_root or output.is_relative_to(input_root): parser.error('Output must be outside immutable input roots')
    output.mkdir(parents=True,exist_ok=True)
    identity=make_identity(reps,mode)
    if args.preflight:
        atomic_json(output/'preflight.json',dict(status='validated_no_intervals_computed',created_utc=utc(),identity=identity,
            requested_workers=args.workers,logical_processors=os.cpu_count(),measured_physical_cores=None,
            historical_eight_worker_peak_bytes=8*311103488,historical_memory_note='Historical archived workload estimate, not a measurement of this host or current scalar implementation',thread_environment={k:os.environ[k] for k in THREAD_ENV}))
        print(json.dumps(dict(status='preflight_pass',replications=len(reps),identity_sha256=identity['identity_sha256'],output=str(output))),flush=True); return
    bind_identity(output,identity)
    if args.summarize:
        verify_hashes(identity['source_sha256']); aggregate(output,identity); return
    existing=[rep for rep in reps if validate_result(output,rep,identity) is not None]
    todo=[rep for rep in reps if rep not in existing]
    started=time.perf_counter(); manifest=dict(status='running',identity_sha256=identity['identity_sha256'],started_utc=utc(),target_replications=reps,
        workers=args.workers,resumed_cases=existing,new_cases_completed=[],max_wall_minutes=args.max_wall_minutes,thread_environment={k:os.environ[k] for k in THREAD_ENV},
        historical_eight_worker_peak_bytes=8*311103488,historical_memory_note='Historical archived workload estimate, not a measurement of this host or current scalar implementation',new_chains=0,original_inputs_written=False)
    atomic_json(output/'run_manifest.json',manifest)
    # Save one deterministic state ordering shared by every per-case weight array.
    states=core.p.legal_partitions(); cuts=np.zeros((27931,4),dtype=np.int16); counts=np.array(list(map(len,states)),dtype=np.int8)
    for i,s in enumerate(states): cuts[i,:len(s)]=s
    partitions=output/'partition_order.npz'
    if not partitions.exists(): atomic_npz(partitions,cuts=cuts,k=counts)
    else:
        with np.load(partitions,allow_pickle=False) as z:
            if not np.array_equal(z['cuts'],cuts) or not np.array_equal(z['k'],counts): raise RuntimeError('Saved partition order differs')
    try:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=worker_init,initargs=(identity,)) as pool:
            pending={}; cursor=0
            while cursor<len(todo) or pending:
                elapsed=time.perf_counter()-started
                within_budget=elapsed<args.max_wall_minutes*60
                while cursor<len(todo) and len(pending)<args.workers*2 and within_budget:
                    rep=todo[cursor]; cursor+=1; pending[pool.submit(compute_case,rep,str(output))]=rep
                if not pending: break
                done,_=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
                for future in done:
                    rep=pending.pop(future); item=future.result(); manifest['new_cases_completed'].append(rep)
                    print(json.dumps(dict(replication=rep,complete=len(existing)+len(manifest['new_cases_completed']),target=len(reps),seconds=item['seconds']['total'],max_cdf_residual=item['checks']['max_full_cdf_residual'])),flush=True)
                manifest['elapsed_seconds']=time.perf_counter()-started
                atomic_json(output/'run_manifest.json',manifest)
        verify_hashes(identity['source_sha256'])
        # Detect any original/cache mutation during the whole run, not only each task.
        for inputs in identity['inputs'].values(): verify_hashes(inputs['input_sha256'])
        complete=len(existing)+len(manifest['new_cases_completed'])
        if complete==len(reps):
            aggregate(output,identity); manifest['status']='complete'
        else: manifest['status']='budget_limited_resumable'
        manifest.update(finished_utc=utc(),elapsed_seconds=time.perf_counter()-started,complete_cases=complete)
        atomic_json(output/'run_manifest.json',manifest)
    except BaseException:
        manifest.update(status='failed_resumable',elapsed_seconds=time.perf_counter()-started,traceback=traceback.format_exc(),finished_utc=utc())
        atomic_json(output/'run_manifest.json',manifest); raise

if __name__=='__main__': main()
