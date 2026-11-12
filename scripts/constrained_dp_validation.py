"""Independent weighted-SSE DP references and read-only cached solver audit.

The objective includes log(T) for every segment, including the first.
Cache audit inputs are explicit; no private Python module is imported.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.metadata, json, math, platform, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
if __package__:
    from . import simulation as sim
else:
    import simulation as sim
ROOT = Path(__file__).resolve().parents[1]
METHOD = "Constrained DP-BIC (Kmax=4)"
_LAYOUT_CACHE = {}
def _inputs(statistic, weights, minseg, kmax):
    y, w = np.asarray(statistic, float), np.asarray(weights, float)
    if y.ndim != 1 or w.shape != y.shape or not len(y):
        raise ValueError("statistic and weights must be nonempty equal-length vectors")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("finite statistics and strictly positive finite weights required")
    if not isinstance(minseg, (int, np.integer)) or minseg < 1 or minseg > len(y):
        raise ValueError("minseg must be an integer between 1 and T")
    if not isinstance(kmax, (int, np.integer)) or kmax < 0:
        raise ValueError("kmax must be a nonnegative integer")
    return y, w

def partition_objective(statistic, weights, cuts=(), minseg=8):
    """Original additive SSE + log(T) * number of segments objective."""
    y, w = _inputs(statistic, weights, minseg, 0)
    cuts = tuple(cuts)
    edges = (0,) + cuts + (len(y),)
    if any(not isinstance(c, (int, np.integer)) for c in cuts) or any(
        b - a < minseg for a, b in zip(edges, edges[1:])
    ):
        raise ValueError("invalid cuts or segment shorter than minseg")
    cost = sim.WeightedL2(minseg).fit(np.c_[y, w])
    return float(sum(cost.error(a, b) + math.log(len(y)) for a, b in zip(edges, edges[1:])))

def constrained_dp(statistic, weights, minseg=8, kmax=4):
    """Globally minimize weighted SSE + log(T) per segment for K <= kmax.

    Exact segment-count DP, O((kmax+1) T^2) time and O(T^2 + kmax*T)
    memory.  Ties use the first predecessor and then the fewest segments.
    The returned tuple has at most kmax cuts and every segment has >= minseg
    positions.  The calculation deliberately matches WeightedL2's prefix sums.
    """
    y, w = _inputs(statistic, weights, minseg, kmax)
    n = len(y)
    # Weighted SSE is translation invariant; remove the common offset before
    # prefix products to avoid catastrophic cancellation, without clipping SSE.
    centered = y - y[0]
    sw, swy, swy2 = (np.r_[0.0, np.cumsum(v)] for v in (w, w*centered, w*centered*centered))
    cost = np.full((n+1, n+1), np.inf)
    for end in range(minseg, n+1):
        starts = np.arange(end-minseg+1)
        cost[starts, end] = (swy2[end]-swy2[starts]
                            - (swy[end]-swy[starts])**2/(sw[end]-sw[starts]))
    max_segments = min(kmax+1, n//minseg)
    dp = np.full((max_segments+1, n+1), np.inf)
    previous = np.full(dp.shape, -1, dtype=int)
    dp[0, 0] = 0.0
    penalty = math.log(n)
    for count in range(1, max_segments+1):
        for end in range(count*minseg, n+1):
            starts = np.arange((count-1)*minseg, end-minseg+1)
            values = dp[count-1, starts] + cost[starts, end] + penalty
            j = int(np.argmin(values))
            dp[count, end], previous[count, end] = values[j], starts[j]
    count = int(np.argmin(dp[1:, n]))+1
    end, cuts = n, []
    while count:
        start = int(previous[count, end])
        if start:
            cuts.append(start)
        end, count = start, count-1
    return tuple(reversed(cuts))

def weighted_statistic(train, participants, H, h, yqy):
    """Training-only preprocessing identical to simulation.pelt_fit."""
    person, trial, z, c, y = train
    global_curve = sim.fixed_curve(H, h, yqy, len(y), ())
    global_delta = float(global_curve[0])
    nuisance = sim.recover_nuisance(train, global_curve, participants)
    residual = y-(nuisance[0]+nuisance[1]*z+nuisance[person+2]+c*global_delta)
    sigma2 = float(residual@residual/max(1, len(y)-(participants+3)))
    diff, variance = np.empty(sim.T), np.empty(sim.T)
    for t in range(1, sim.T+1):
        index = trial == t
        rt, ct = residual[index], c[index]
        plus, minus = rt[ct > 0], rt[ct < 0]
        if not len(plus) or not len(minus):
            raise RuntimeError(f"training trial {t} lacks a congruency arm")
        diff[t-1] = global_delta+plus.mean()-minus.mean()
        variance[t-1] = sigma2*(1/len(plus)+1/len(minus))
    return diff, 1/variance

def constrained_fit(train, participants, H, h, yqy, minseg=8, kmax=4):
    """Use original preprocessing and observation-level refit around exact DP."""
    statistic, weights = weighted_statistic(train, participants, H, h, yqy)
    cuts = constrained_dp(statistic, weights, minseg, kmax)
    return cuts, sim.fixed_curve(H, h, yqy, len(train[-1]), cuts)

def exact_dp(statistic, weights, minseg=8):
    """Unpruned recurrence, independent of segment-count DP."""
    y,w=_inputs(statistic,weights,minseg,0)
    cost=sim.WeightedL2(minseg).fit(np.c_[y,w]);f={0:0.};paths={0:()}
    for end in range(minseg,len(y)+1):
        value,start=min((f[s]+cost.error(s,end)+math.log(len(y)),s) for s in [0]+list(range(minseg,end-minseg+1)))
        f[end]=value;paths[end]=paths[start]+((start,) if start else ())
    return paths[len(y)]


def centered_objective(statistic,weights,cuts=(),minseg=8):
    y,w=_inputs(statistic,weights,minseg,0)
    partition_objective(y,w,cuts,minseg)
    edges=(0,)+tuple(cuts)+(len(y),)
    return float(sum(np.dot(w[a:b],(y[a:b]-np.average(y[a:b],weights=w[a:b]))**2)+math.log(len(y)) for a,b in zip(edges,edges[1:])))


def brute_force(statistic,weights,minseg=8,kmax=4):
    from itertools import combinations
    y,w=_inputs(statistic,weights,minseg,kmax);choices=[]
    for k in range(min(kmax,len(y)//minseg-1)+1):
        for cuts in combinations(range(minseg,len(y)-minseg+1),k):
            if min(np.diff((0,)+cuts+(len(y),)))>=minseg:
                choices.append((centered_objective(y,w,cuts,minseg),k,tuple(reversed(cuts)),cuts))
    return min(choices)[-1]


def solve(y,w,oldcuts=(),minseg=8):
    corrected=sim.pelt_bic(y,w,minseg);free=exact_dp(y,w,minseg);cap=constrained_dp(y,w,minseg,4)
    oo,fo,co,do=[partition_objective(y,w,c,minseg) for c in (oldcuts,free,corrected,cap)]
    tol=1e-9*max(1.,abs(fo))
    if abs(co-fo)>tol or do<fo-tol or (len(corrected)<=4 and abs(do-fo)>tol):raise ArithmeticError('PELT/DP objective disagreement')
    if abs(centered_objective(y,w,free,minseg)-fo)>tol:raise ArithmeticError('prefix/centered SSE disagreement')
    return dict(archived_cuts=list(oldcuts),corrected_cuts=list(corrected),exact_unrestricted_dp_cuts=list(free),constrained_dp_cuts=list(cap),archived_objective=oo,corrected_objective=co,unrestricted_objective=fo,constrained_objective=do,objective_tolerance=tol,solver_partition_changed=tuple(corrected)!=tuple(oldcuts),solver_objective_corrected=oo>fo+tol,cap_partition_changed=cap!=corrected,cap_objective_increased=do>fo+tol,corrected_K_gt4=len(corrected)>4)


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)) or ['status']);writer.writeheader();writer.writerows(rows)


def _audit_record(task):
    path,input_root=task;old=read(path);source=input_root/old['source'];cache=input_root/old['cache']
    for p,key in ((source,'source_sha256'),(cache,'cache_sha256')):
        if sim.sha256_file(p)!=old[key]:raise ValueError(f'fingerprint mismatch: {p}')
    original=read(source)
    cg=next(r for r in original['methods'] if r['method'] in ('A3','CG-RJMCMC'))
    if cg!=old['original_cg']:raise ValueError('CG record mismatch')
    with np.load(cache,allow_pickle=False) as a:
        y,w=a['statistic'].copy(),a['weights'].copy()
        key='source_summary_hash' if 'source_summary_hash' in a else 'source_sha256'
        if str(a[key])!=old['source_sha256']:raise ValueError('cache source mismatch')
        for key in ('data_hash','holdout_hash','training_mask_hash','outcomes_hash'):
            if key in a and str(a[key])!=original[key]:raise ValueError('cache identity '+key)
    audit=solve(y,w,tuple(old['archived_pelt']['estimated_cuts']))
    if 'audit' in old:
        baseline=old['audit'];keys=('archived_objective','corrected_objective','unrestricted_objective','constrained_objective')
        audit['previous_objective_differences']={key:audit[key]-baseline[key] for key in keys}
        audit['previous_partition_changes']={key:audit[key]!=baseline[key] for key in ('corrected_cuts','exact_unrestricted_dp_cuts','constrained_dp_cuts')}
        if any(abs(value)>audit['objective_tolerance'] for value in audit['previous_objective_differences'].values()):raise ArithmeticError('centered objective differs from previous audited objective beyond tolerance')
    if audit['corrected_cuts']!=old['corrected_pelt']['estimated_cuts']:raise ValueError('corrected metric partition mismatch')
    if old.get('constrained_dp') and audit['constrained_dp_cuts']!=old['constrained_dp']['estimated_cuts']:raise ValueError('constrained metric partition mismatch')
    if not old.get('constrained_dp'):
        if audit['constrained_dp_cuts']==audit['corrected_cuts']:
            old['constrained_dp']=dict(old['corrected_pelt'],method=METHOD)
        else:
            # Reconstruct only the cases needing a different observation-level refit.
            case=original['case'];source_root=source.parents[4]
            resolved=read(source_root/'resolved_config.json');cfg=resolved['config']
            geometry=ROOT/'scripts/config/r61_congruency_geometry.csv'
            base,_=sim.load_frozen_geometry(geometry)
            if sim.sha256_array(base)!=resolved['geometry_hash']:raise ValueError('geometry mismatch')
            label=f"{case['stage']}|{case['cell']}|R{original['replicate']:06d}"
            seed=sim.deterministic_seed(_case_namespace(cfg,case),'DATA|'+label)
            if seed!=original['data_seed']:raise ValueError('seed mismatch')
            design=sim.design_for_n(base,case['participants']);outcome,truth,levels=sim.generate_outcome(cfg,case,design,seed)
            testmask=sim.outcome_blind_holdout(*design[:2],cfg['design']['holdout_trials_per_participant'],_case_namespace(cfg,case)+'|'+label)
            trainmask=sim.mcar_training_mask(~testmask,_case_namespace(cfg,case),label,case['training_mcar'])
            for key,value in (('data_hash',sim.sha256_array(outcome)),('holdout_hash',sim.sha256_array(np.packbits(testmask))),('training_mask_hash',sim.sha256_array(np.packbits(trainmask)))):
                if original[key]!=value:raise ValueError('reconstruction mismatch '+key)
            train,test=sim.pack(design,outcome,trainmask),sim.pack(design,outcome,testmask)
            with np.load(cache) as a:H,h,yqy=a['H'],a['h'],float(a['yqy'])
            yy,ww=weighted_statistic(train,case['participants'],H,h,yqy)
            if not np.allclose(yy,y,rtol=0,atol=1e-14) or not np.allclose(ww,w,rtol=1e-14,atol=0):raise ValueError('reconstructed statistic mismatch')
            cuts=tuple(audit['constrained_dp_cuts']);curve=sim.fixed_curve(H,h,yqy,len(train[-1]),cuts)
            old['constrained_dp']=dict(method=METHOD,**curve_metrics(curve,cuts,truth,original['truth']['cuts'],original['truth']['levels'],train,test,case['participants']))
    config_path=source.parents[4]/'resolved_config.json'
    return dict(source_config_sha256=sim.sha256_file(config_path) if config_path.exists() else None,cohort=old.get('cohort','sensitivity'),cell=old.get('cell',old.get('case',{}).get('cell')),rep=old.get('rep',original.get('replicate',original.get('rep'))),source_sha256=old['source_sha256'],cache_sha256=old['cache_sha256'],audit=audit,corrected_pelt=old['corrected_pelt'],constrained_dp=old.get('constrained_dp'),original_cg=cg)


def aggregate(records,output):
    summary=[];paired=[];support=[]
    for cohort,cell in sorted(set((r['cohort'],r['cell']) for r in records)):
        rr=[r for r in records if (r['cohort'],r['cell'])==(cohort,cell)]
        support.append(dict(cohort=cohort,cell=cell,n=len(rr),**{k:sum(r['audit'][k] for r in rr) for k in ('solver_partition_changed','cap_partition_changed','solver_objective_corrected','cap_objective_increased')}))
        for method in ('corrected_pelt','constrained_dp','original_cg'):
            for metric in ('correct_K','Hausdorff','trajectory_RMSE','test_RMSE','segment_level_RMSE'):
                values=[r[method][metric] for r in rr if r.get(method) is not None and r[method].get(metric) is not None]
                if not values:continue
                summary.append(dict(cohort=cohort,cell=cell,method=method,metric=metric,n=len(values),mean=float(np.mean(values)),MCSE=float(np.std(values,ddof=1)/np.sqrt(len(values))) if len(values)>1 else None))
                if method!='original_cg':
                    v=[r['original_cg'][metric]-r[method][metric] for r in rr if r.get(method) is not None and r[method].get(metric) is not None and r['original_cg'].get(metric) is not None]
                    if not v:continue
                    paired.append(dict(cohort=cohort,cell=cell,contrast='CG minus '+method,metric=metric,n=len(v),mean_difference=float(np.mean(v)),MCSE=float(np.std(v,ddof=1)/np.sqrt(len(v))) if len(v)>1 else None))
    write_csv(output/'per_dataset_objectives.csv',[dict(cohort=r['cohort'],cell=r['cell'],rep=r['rep'],**r['audit']) for r in records]);write_csv(output/'summary.csv',summary);write_csv(output/'paired_differences.csv',paired);write_csv(output/'support_and_optimality.csv',support)
    return support


def _case_namespace(cfg,case):
    """Resolve per-DGP namespaces while retaining legacy configuration support."""
    resolver=getattr(sim,'case_namespace',None)
    return resolver(cfg,case) if resolver else cfg.get('namespace_by_dgp',{}).get(case['dgp'],cfg['namespace'])


def collect_reconstruction_tasks(simulation_root=None,primary_root=None,balanced_d5_root=None,small_sample_root=None):
    """Find exact source scope from one current root or split historical roots."""
    tasks=[]
    if simulation_root is not None:
        for dgp in ('D0','D1','D2','D3','D4','D5'):
            tasks.extend((p,simulation_root) for p in sorted((simulation_root/'replicates/main'/dgp).glob('rep_*/summary.json')))
        for stage in ('sensitivity','kmax'):
            tasks.extend((p,simulation_root) for p in sorted((simulation_root/'replicates'/stage).glob('*/rep_*/summary.json')))
    else:
        for dgp in ('D0','D1','D2','D3','D4'):
            tasks.extend((p,primary_root) for p in sorted((primary_root/'replicates/main'/dgp).glob('rep_*/summary.json')))
        source=balanced_d5_root/'main';tasks.extend((p,source) for p in sorted((source/'replicates/main/D5').glob('rep_*/summary.json')))
        for stage in ('sensitivity','kmax'):
            tasks.extend((p,primary_root) for p in sorted((primary_root/'replicates'/stage).glob('*D1*/rep_*/summary.json')))
            source=balanced_d5_root/stage;tasks.extend((p,source) for p in sorted((source/'replicates'/stage).glob('*/rep_*/summary.json')))
    tasks.extend((p,small_sample_root) for p in sorted((small_sample_root/'formal').glob('SN*/R*.json')))
    return tasks


def export_exact_bridge(output,original,record,H,h,yqy,code_hash,namespace):
    """Write explicit compatible aliases for exact-D5 sufficient-statistic input."""
    rep=original['replicate'];stem=f'rep_{rep:06d}'
    metadata=dict(reconstruction_dependency_fingerprint=record.get('reconstruction_dependency_fingerprint',''),staging_scope=record.get('staging_scope','fresh reconstructed input'),original_statistics_sha256=record.get('original_statistics_sha256',''),original_audit_record_sha256=record.get('original_audit_record_sha256',''),replicate=rep,dgp='D5',namespace=namespace,source_summary_hash=record['source_sha256'],code_hash=code_hash,
        **{key:original[key] for key in ('data_hash','holdout_hash','training_mask_hash','data_seed','n_train','n_test')})
    statistics=output/'statistics/D5';records=output/'replicates/D5'
    statistics.mkdir(parents=True,exist_ok=True);records.mkdir(parents=True,exist_ok=True)
    cache=statistics/f'{stem}.npz'
    if cache.exists():
        with np.load(cache,allow_pickle=False) as a:
            for key in ('source_summary_hash','code_hash','data_hash','holdout_hash','training_mask_hash'):
                if str(a[key])!=str(metadata[key]):raise ValueError('exact bridge checkpoint fingerprint mismatch')
    else:np.savez_compressed(cache,H=H,h=h,yqy=yqy,n=original['n_train'],**metadata)
    target=records/f'{stem}.json'
    if target.exists() and read(target)!=metadata:raise ValueError('exact bridge record fingerprint mismatch')
    sim.atomic_json(target,metadata)


def stage_cached_exact_input(path,input_root,output,audited):
    """Stage already-audited legacy D5 statistics; no new posterior calculation."""
    archived=read(path)
    if (audited['cohort'],audited['cell'])!=('primary','D5'):return None
    source=input_root/archived['source'];cache=input_root/archived['cache']
    source_hash,cache_hash=sim.sha256_file(source),sim.sha256_file(cache)
    if source_hash!=archived['source_sha256'] or cache_hash!=archived['cache_sha256'] or source_hash!=audited['source_sha256'] or cache_hash!=audited['cache_sha256']:raise ValueError('cached D5 staging fingerprint mismatch')
    original=read(source);case=original['case'];rep=original['replicate']
    if (case['dgp'],case['stage'],case['cell'],rep)!=('D5','main','D5',audited['rep']):raise ValueError('cached D5 staging identity mismatch')
    cfg=read(source.parents[4]/'resolved_config.json')['config'];namespace=_case_namespace(cfg,case)
    if original['data_seed']!=sim.deterministic_seed(namespace,f'DATA|main|D5|R{rep:06d}'):raise ValueError('cached D5 staging seed mismatch')
    with np.load(cache,allow_pickle=False) as a:
        source_key='source_summary_hash' if 'source_summary_hash' in a else 'source_sha256'
        if str(a[source_key])!=source_hash:raise ValueError('cached D5 staging source mismatch')
        for key in ('data_hash','holdout_hash','training_mask_hash'):
            if str(a[key])!=original[key]:raise ValueError('cached D5 staging identity '+key)
        H,h,yqy=a['H'].copy(),a['h'].copy(),float(a['yqy'])
        code_hash=str(a['code_hash']) if 'code_hash' in a else archived.get('reconstruction_code_sha256')
    if not code_hash:raise ValueError('cached D5 staging lacks original generation code hash')
    if H.shape!=(64,64) or h.shape!=(64,) or not np.all(np.isfinite(H)) or not np.all(np.isfinite(h)) or not np.isfinite(yqy):raise ValueError('cached D5 staging invalid sufficient statistics')
    identity=dict(scientific_dependencies=scientific_identity(),source_summary_sha256=source_hash,original_statistics_sha256=cache_hash,original_audit_record_sha256=sim.sha256_file(path))
    provenance=dict(source_sha256=source_hash,reconstruction_dependency_fingerprint=identity_digest(identity),staging_scope='validated archived input staging; no new posterior calculation',original_statistics_sha256=cache_hash,original_audit_record_sha256=identity['original_audit_record_sha256'])
    export_exact_bridge(output,original,provenance,H,h,yqy,code_hash,namespace)
    return dict(replicate=rep,original_summary=str(source.resolve()),original_statistics=str(cache.resolve()),original_audit_record=str(path.resolve()),**identity,staged_statistics_sha256=sim.sha256_file(output/'statistics/D5'/f'rep_{rep:06d}.npz'),staged_record_sha256=sim.sha256_file(output/'replicates/D5'/f'rep_{rep:06d}.json'))


def scientific_identity():
    """Current executable scientific dependencies, distinct from old provenance."""
    paths=[Path(__file__),ROOT/'scripts/simulation.py',ROOT/'scripts/small_sample_validation.py',ROOT/'scripts/application.py',ROOT/'scripts/kernel.c',ROOT/'scripts/config/r61_congruency_geometry.csv']
    return dict(source_sha256={str(p.resolve()):sim.sha256_file(p) for p in paths},
                versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','pandas','ruptures')})


def reconstruction_identity(source_root,design_input=None,application_input=None):
    identity=scientific_identity();inputs={}
    resolved=source_root/'resolved_config.json'
    if resolved.exists():inputs[str(resolved.resolve())]=sim.sha256_file(resolved)
    if design_input is not None:inputs[str(design_input.resolve())]=sim.sha256_file(design_input)
    if application_input is not None:
        if __package__:
            from . import application
        else:
            import application
        inputs.update({str(p.resolve()):sim.sha256_file(p) for p in application._source_files(application_input)})
    identity['input_sha256']=inputs
    return identity


def identity_digest(identity):
    return hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def curve_metrics(curve,cuts,truth,truthcuts,truthlevels,train,test,participants):
    """Same correct-K-conditional ordered-level definition as primary runs."""
    level_rmse,level_bias=sim.ordered_segment_error(curve,cuts,truthlevels,len(truthcuts))
    return dict(estimated_cuts=list(cuts),estimated_K=len(cuts),correct_K=int(len(cuts)==len(truthcuts)),
        Hausdorff=sim.hausdorff(cuts,tuple(truthcuts)),trajectory_RMSE=float(np.sqrt(np.mean((curve-truth)**2))),
        test_RMSE=sim.test_rmse(train,test,curve,participants),segment_level_RMSE=level_rmse,segment_level_bias=level_bias)


def reconstruct_source(source, source_root, output, design_input=None, application_input=None, dependency_identity=None):
    """Recover an additive solver input from hash-checked outcomes, never MCMC."""
    original=read(source);is_small='cell' in original and original['cell'] in ('SN0','SN1')
    source_hash=sim.sha256_file(source);code_hash=sim.sha256_file(Path(__file__))
    dependencies=dependency_identity if dependency_identity is not None else reconstruction_identity(source_root,design_input,application_input)
    dependency_fingerprint=identity_digest(dependencies)
    case=original.get('case',{})
    initial_cell=original['cell'] if is_small else case['cell']
    initial_rep=original['rep'] if is_small else original['replicate']
    initial_cohort='N35' if is_small else ('primary' if case['stage']=='main' else 'sensitivity')
    staged_cache=output/'reconstructed_inputs'/initial_cohort/initial_cell/f'R{initial_rep:06d}.npz'
    staged_record=staged_cache.with_suffix('.json')
    if staged_record.exists():
        old=read(staged_record)
        if old.get('reconstruction_dependency_fingerprint')!=dependency_fingerprint or old.get('reconstruction_code_sha256')!=code_hash or old['source_sha256']!=source_hash or sim.sha256_file(staged_cache)!=old['cache_sha256']:raise ValueError('reconstruction checkpoint fingerprint mismatch; use a new output root')
        if initial_cohort=='primary' and initial_cell=='D5':
            cfg=read(source_root/'resolved_config.json')['config']
            with np.load(staged_cache,allow_pickle=False) as a:export_exact_bridge(output,original,old,a['H'],a['h'],float(a['yqy']),code_hash,_case_namespace(cfg,case))
        return staged_record
    if is_small:
        if __package__:
            from . import small_sample_validation as small
        else:
            import small_sample_validation as small
        layout_key=(str(design_input),str(application_input),dependency_fingerprint)
        if layout_key not in _LAYOUT_CACHE:_LAYOUT_CACHE[layout_key]=small.load_design(design_input,application_input)
        train_design,test_design=_LAYOUT_CACHE[layout_key]
        small.STATE['protocol']={'cells':small.CELLS}
        cell,rep=original['cell'],original['rep'];N=35;cohort='N35'
        if original['data_seed']!=small.seed('formal',cell,rep,'DATA'):raise ValueError('small seed mismatch')
        train,test,truth=small.simulate(cell,original['data_seed'],train_design,test_design)
        if sim.sha256_array(np.r_[train[-1],test[-1]])!=original['outcomes_hash']:raise ValueError('small outcome mismatch')
        truthcuts=small.CELLS[cell]['cuts'];truthlevels=small.CELLS[cell]['levels']
    else:
        case=original['case'];resolved=read(source_root/'resolved_config.json');cfg=resolved['config']
        cell,rep,N=case['cell'],original['replicate'],case['participants'];cohort='primary' if case['stage']=='main' else 'sensitivity'
        base,_=sim.load_frozen_geometry(ROOT/'scripts/config/r61_congruency_geometry.csv')
        if sim.sha256_array(base)!=resolved['geometry_hash']:raise ValueError('geometry mismatch')
        label=f"{case['stage']}|{cell}|R{rep:06d}";seed=sim.deterministic_seed(_case_namespace(cfg,case),'DATA|'+label)
        if seed!=original['data_seed']:raise ValueError('seed mismatch')
        design=sim.design_for_n(base,N);y,truth,_=sim.generate_outcome(cfg,case,design,seed)
        testmask=sim.outcome_blind_holdout(*design[:2],cfg['design']['holdout_trials_per_participant'],_case_namespace(cfg,case)+'|'+label)
        trainmask=sim.mcar_training_mask(~testmask,_case_namespace(cfg,case),label,case['training_mcar'])
        for key,value in (('data_hash',sim.sha256_array(y)),('holdout_hash',sim.sha256_array(np.packbits(testmask))),('training_mask_hash',sim.sha256_array(np.packbits(trainmask)))):
            if original[key]!=value:raise ValueError('reconstruction mismatch '+key)
        train,test=sim.pack(design,y,trainmask),sim.pack(design,y,testmask);truthcuts=original['truth']['cuts'];truthlevels=original['truth']['levels']
    H,h,yqy=sim.collapsed(*train,N);values,weights=weighted_statistic(train,N,H,h,yqy)
    archived=next(r for r in original['methods'] if r['method']=='PELT-BIC')
    def metric(cuts):
        curve=sim.fixed_curve(H,h,yqy,len(train[-1]),cuts)
        return curve_metrics(curve,cuts,truth,truthcuts,truthlevels,train,test,N)
    audit=solve(values,weights,tuple(archived['estimated_cuts']))
    record=dict(reconstruction_dependency_fingerprint=dependency_fingerprint,reconstruction_dependencies=dependencies,reconstruction_code_sha256=code_hash,cohort=cohort,cell=cell,rep=rep,source=str(source.resolve()),source_sha256=sim.sha256_file(source),archived_pelt=archived,corrected_pelt=metric(tuple(audit['corrected_cuts'])),constrained_dp=metric(tuple(audit['constrained_dp_cuts'])),original_cg=next(r for r in original['methods'] if r['method'] in ('A3','CG-RJMCMC')))
    cache=output/'reconstructed_inputs'/cohort/cell/f'R{rep:06d}.npz';cache.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache,statistic=values,weights=weights,H=H,h=h,yqy=yqy,source_sha256=record['source_sha256'],reconstruction_dependency_fingerprint=dependency_fingerprint)
    record.update(cache=str(cache.resolve()),cache_sha256=sim.sha256_file(cache))
    path=cache.with_suffix('.json');sim.atomic_json(path,record)
    if cohort=='primary' and cell=='D5':export_exact_bridge(output,original,record,H,h,yqy,code_hash,_case_namespace(cfg,case))
    return path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke',action='store_true');mode.add_argument('--full',action='store_true')
    parser.add_argument('--cache-root',type=Path,help='Explicit cache audit directory containing records and additional_records')
    parser.add_argument('--input-root',type=Path,default=ROOT,help='Root resolving recorded input paths')
    parser.add_argument('--output-root',type=Path,default=ROOT/'results/reproduction/constrained_dp_validation');parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--simulation-root',type=Path,help='Unified public simulation output root with all main/sensitivity/kmax cases')
    parser.add_argument('--primary-root',type=Path,help='Historical formal root, D0-D4 and D1 sensitivity')
    parser.add_argument('--balanced-d5-root',type=Path,help='Current balanced D5 root with main/sensitivity/kmax subdirectories')
    parser.add_argument('--small-sample-root',type=Path,help='Historical N35 root with formal/SN*/R*.json')
    parser.add_argument('--design-input',type=Path)
    parser.add_argument('--application-input',type=Path)
    args=parser.parse_args(argv)
    if args.workers<1:parser.error('workers must be positive')
    out=args.output_root.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.cache_root and (out==args.cache_root.resolve() or args.cache_root.resolve() in out.parents):parser.error('output must be outside read-only cache root')
    existing=read(out/'manifest.json') if (out/'manifest.json').exists() else None
    if existing and (existing['source_sha256']!=sim.sha256_file(Path(__file__)) or existing['mode']!=('full' if args.full else 'smoke')):raise ValueError('resume source/mode fingerprint mismatch; use a new output root')
    current_science=scientific_identity()
    if existing and existing.get('scientific_dependencies')!=current_science:raise ValueError('resume scientific dependency fingerprint mismatch; use a new output root')
    records=[];applications=[];staging_tasks=[]
    if args.cache_root:
        paths=sorted((args.cache_root/'records').glob('*/*/R*.json'))+sorted((args.cache_root/'additional_records/sensitivity').glob('*/R*.json'))
        if args.full and len(paths)!=5600:raise ValueError(f'full audit requires 5600 inputs, found {len(paths)}')
        if args.smoke:paths=list(dict.fromkeys(paths[:2]+[p for p in paths if (p.parent.name,p.stem) in (('D1','R000066'),('SN0','R000352'))]))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:records=list(pool.map(_audit_record,[(p,args.input_root) for p in paths]))
        staging_tasks=[(path,record) for path,record in zip(paths,records) if (record['cohort'],record['cell'])==('primary','D5')]
        app=args.cache_root/'application.json'
        if app.exists():
            old=read(app)
            if sim.sha256_file(args.input_root/old['source'])!=old['source_sha256']:raise ValueError('application source mismatch')
            for group in old['groups']:
                cache=args.cache_root/(group['group']+'_application_statistics.npz')
                with np.load(cache) as a:audit=solve(a['statistic'],a['weights'],tuple(group['audit']['archived_cuts']))
                applications.append(dict(group=group['group'],cache_sha256=sim.sha256_file(cache),source_sha256=old['source_sha256'],application_record_sha256=sim.sha256_file(app),audit=audit))
        if args.full and len(applications)!=2:raise ValueError('full audit requires two application groups')
    elif args.full:
        if not args.small_sample_root or not args.application_input or (args.simulation_root is None and not all((args.primary_root,args.balanced_d5_root))):parser.error('--full reconstruction requires --simulation-root (or both legacy roots), --small-sample-root and --application-input')
        tasks=collect_reconstruction_tasks(args.simulation_root,args.primary_root,args.balanced_d5_root,args.small_sample_root)
        if len(tasks)!=5600:raise ValueError(f'full reconstruction requires 5600 source records, found {len(tasks)}')
        reconstruction_dependencies={root:reconstruction_identity(root,args.design_input,args.application_input) for root in set(root for _,root in tasks)}
        for i,(source,root) in enumerate(tasks,1):
            staged=reconstruct_source(source,root,out,args.design_input,args.application_input,reconstruction_dependencies[root])
            records.append(_audit_record((staged,args.input_root)))
            if i%100==0:print(f'reconstructed {i}/5600',flush=True)
        for root,identity in reconstruction_dependencies.items():
            if identity!=reconstruction_identity(root,args.design_input,args.application_input):raise ValueError('scientific input changed during reconstruction; use a new output root')
        if __package__:
            from . import application
        else:
            import application
        for group in ('Low','High'):
            train,test,metadata=application.prepare_group(args.application_input,group);H,h,yqy=sim.collapsed(*train,metadata['participants']);values,weights=weighted_statistic(train,metadata['participants'],H,h,yqy)
            import ruptures as rpt
            oldcuts=tuple(rpt.Pelt(custom_cost=sim.WeightedL2(8),min_size=8,jump=1).fit(np.c_[values,weights]).predict(pen=math.log(len(values)))[:-1])
            applications.append(dict(group=group,audit=solve(values,weights,oldcuts),input_hashes={str(p):sim.sha256_file(p) for p in application._source_files(args.application_input)}))
    else:
        rng=np.random.default_rng(20261112)
        for rep in range(3):
            y=rng.normal(size=16);w=rng.uniform(.1,4,size=16);cap=constrained_dp(y,w,3,4)
            if not np.isclose(centered_objective(y,w,cap,3),centered_objective(y,w,brute_force(y,w,3,4),3)):raise ArithmeticError('brute force mismatch')
            records.append(dict(cohort='synthetic_smoke',cell='tiny',rep=rep,audit=solve(y,w,exact_dp(y,w,3),3)))
    if current_science!=scientific_identity():raise ValueError('scientific dependency changed during audit; use a new output root')
    run_identity=dict(scientific_dependencies=current_science,datasets=[(r.get('source_sha256'),r.get('cache_sha256'),r.get('source_config_sha256')) for r in records],applications=applications,configuration=dict(minseg=8,kmax=4,objective='weighted SSE + log(T) per segment'))
    fingerprint=identity_digest(run_identity)
    if existing and existing['input_fingerprint']!=fingerprint:raise ValueError('resume input fingerprint mismatch; use a new output root')
    if args.full:
        from collections import Counter
        counts=Counter(r['cohort'] for r in records)
        if counts!={'primary':3000,'N35':1000,'sensitivity':1600} or len({(r['cohort'],r['cell'],r['rep']) for r in records})!=5600:raise ValueError('full cohort identities/counts mismatch')
    if args.cache_root:
        staged=[stage_cached_exact_input(path,args.input_root,out,record) for path,record in staging_tasks]
        sim.atomic_json(out/'exact_input_staging.json',dict(scope='validated archived input staging; no new posterior calculation',replications=staged))
    support=aggregate(records,out)
    manifest=dict(mode='full' if args.full else 'smoke',formal_results=bool(args.full),datasets=len(records),application_groups=len(applications),objective='weighted SSE + log(T) per segment',minseg=8,kmax=4,new_sampler_runs=0,python=sys.version,numpy=np.__version__,source_sha256=sim.sha256_file(Path(__file__)),scientific_dependencies=current_science,input_fingerprint=fingerprint,support=support,applications=applications,metric_provenance='Archived observation-level metrics reused only for identical validated partitions; changed sensitivity cap partitions receive hash-verified seeded reconstruction/refit. Solver objectives independently recomputed.',anomaly_records=[r for r in records if (r['cell'],r['rep']) in (('D1',66),('SN0',352))])
    sim.atomic_json(out/'manifest.json',manifest);sim.atomic_json(out/'solver_audit.json',dict(records=records,applications=applications))
    print(json.dumps(dict(datasets=len(records),application_groups=len(applications),output=str(out))))
    return 0

if __name__=='__main__':raise SystemExit(main())
