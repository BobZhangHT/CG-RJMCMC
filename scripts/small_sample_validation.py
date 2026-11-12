"""Public N=35 synthetic validation using cyclic Low-first observation layouts.

Full mode uses 500 independent datasets per cell and the primary 3-chain budget.
Smoke mode is a tiny engineering check and must not be used as formal evidence.
Only layout arrays are reused: all synthetic intercepts/errors are independent.
"""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, math, os, platform, sys, time, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
if __package__:
    from . import simulation as formal, application
    from . import constrained_dp_validation as comparator
else:
    import simulation as formal, application
    import constrained_dp_validation as comparator
ROOT=Path(__file__).resolve().parents[1]
N=35
NAMESPACE="PLOS-ONE-R1-SMALL-N35-LOW-FIRST-v1"
METHODS=("CG-RJMCMC","PELT-BIC","Constrained DP-BIC (Kmax=4)")
CELLS={"SN0":{"cuts":[],"levels":[.08]},"SN1":{"cuts":[32],"levels":[.04,.14]}}
MAPPING=[i%21 for i in range(N)]
STATE={}
OUT=ROOT/'results/reproduction/small_sample'
PROTOCOL=OUT/'small_sample_manifest.json'
EXECUTION_SOURCE_HASHES={"runner":formal.sha256_file(Path(__file__)),"comparator":formal.sha256_file(Path(comparator.__file__)),"simulation":formal.sha256_file(Path(formal.__file__)),"application":formal.sha256_file(Path(application.__file__)),"kernel":formal.sha256_file(ROOT/"scripts/kernel.c")}
def seed(stage, cell, rep, stream):
    return formal.deterministic_seed(NAMESPACE, f"{stage}|{cell}|R{rep:06d}|{stream}")

def design_hashes(train, test):
    return {f"{label}_{field}": formal.sha256_array(arr)
            for label, packed in (("train", train), ("test", test))
            for field, arr in zip(("person", "bin", "z", "contrast"), packed[:4])}

def simulate(cell, data_seed, train_design, test_design):
    spec = STATE["protocol"]["cells"][cell]
    truth = formal.curve_from(spec["cuts"], spec["levels"])
    design = tuple(np.r_[a, b] for a, b in zip(train_design, test_design))
    person, trial, z, c = design
    rng = np.random.default_rng(data_seed)
    a = rng.normal(0, .4, N)
    y = .05*z + a[person] + c*truth[trial-1] + rng.normal(0, .4, len(person))
    n = len(train_design[0])
    return train_design + (y[:n],), test_design + (y[n:],), truth

def replicate(job):
    stage, cell, rep = job
    if stage not in ("smoke", "formal") or cell not in ("SN0", "SN1") or not 1 <= rep <= (1 if stage == "smoke" else 500):
        raise ValueError("job outside frozen replication set")
    target = OUT / stage / cell / f"R{rep:06d}.json"
    if target.exists():
        old = json.loads(target.read_text(encoding="utf-8"))
        if (old["protocol_sha256"] != formal.sha256_file(PROTOCOL)
                or old["data_seed"] != seed(stage, cell, rep, "DATA")):
            raise RuntimeError(f"resume identity mismatch: {target}")
        return {"job": job, "status": "cached" if old.get("status")=="complete" else "failed_cached"}
    record = {"stage": stage, "cell": cell, "rep": rep,
              "data_seed": seed(stage, cell, rep, "DATA"),
              "chain_seeds": [seed(stage, cell, rep, f"CHAIN{i}") for i in (1, 2, 3)],
              "protocol_sha256": formal.sha256_file(PROTOCOL),
              "execution_source_hashes": EXECUTION_SOURCE_HASHES,
              "execution_process_id": os.getpid(), "methods": []}
    start_all = time.perf_counter()
    try:
        train, test, truth = simulate(cell, record["data_seed"], STATE["train"], STATE["test"])
        record["outcomes_hash"] = formal.sha256_array(np.r_[train[-1], test[-1]])
        common_start = time.perf_counter()
        H, h, yqy = formal.collapsed(*train, N)
        record["common_sufficient_statistics_seconds"] = time.perf_counter()-common_start
        cuts_truth = tuple(STATE["protocol"]["cells"][cell]["cuts"])
        def metric(name, cuts, curve, wall, level_curve=None):
            level_rmse, level_bias = formal.ordered_segment_error(
                curve if level_curve is None else level_curve, cuts,
                STATE["protocol"]["cells"][cell]["levels"], len(cuts_truth))
            return {"method": name, "outcomes_hash": record["outcomes_hash"],
                    "design_hashes": STATE["protocol"]["design"]["hashes"],
                    "segment_level_RMSE": level_rmse, "segment_level_bias": level_bias, "estimated_cuts": list(cuts), "estimated_K": len(cuts),
                    "correct_K": int(len(cuts) == len(cuts_truth)),
                    "Hausdorff": formal.hausdorff(cuts, cuts_truth),
                    "trajectory_RMSE": float(np.sqrt(np.mean((curve-truth)**2))),
                    "test_RMSE": formal.test_rmse(train, test, curve, N),
                    "runtime_seconds": wall}
        start = time.perf_counter()
        diff, weights = pelt_inputs(train, H, h, yqy)
        input_wall = time.perf_counter()-start
        for name, fitter in ((METHODS[1], lambda: formal.pelt_bic(diff, weights, 8)),
                             (METHODS[2], lambda: constrained_dp(diff, weights)[0])):
            start = time.perf_counter()
            cuts = fitter()
            curve = formal.fixed_curve(H, h, yqy, len(train[-1]), cuts)
            record["methods"].append(metric(name, cuts, curve, time.perf_counter()-start+input_wall))
        # Witness exact equality to the original complete PELT wrapper, not just new code.
        old_cuts, old_curve = formal.pelt_fit(train, N, H, h, yqy, 8)
        assert list(old_cuts) == record["methods"][0]["estimated_cuts"]
        assert np.array_equal(old_curve, formal.fixed_curve(H, h, yqy, len(train[-1]), old_cuts))
        chains, stats = [], []
        start = time.perf_counter()
        for initial, chain_seed in zip((0, 2, 4), record["chain_seeds"]):
            chain, stat = formal.run_chain(STATE["library"], H, h, yqy, len(train[-1]),
                                          STATE["iterations"], STATE["burn"], chain_seed, initial, 4, 8, 1.)
            chains.append(chain); stats.append(stat)
        chain_wall = time.perf_counter()-start
        summary_start = time.perf_counter()
        posterior = formal.summarize_a3(chains, stats, chain_wall, 4)
        record["posterior_summary_seconds"] = time.perf_counter()-summary_start
        row = metric(METHODS[0], tuple(posterior["joint_map"][1:]), posterior["bma_curve"], posterior["three_chain_walltime_seconds"], posterior["joint_map_curve"])
        inside = (truth >= posterior["bma_q025"]) & (truth <= posterior["bma_q975"])
        width = posterior["bma_q975"]-posterior["bma_q025"]
        row.update(coverage=float(inside.mean()), interval_width=float(width.mean()),
                   coverage_by_bin=inside.astype(int).tolist(), width_by_bin=width.tolist(),
                   p_k=posterior["p_k"], max_split_Rhat=posterior["max_split_Rhat"],
                   min_bulk_ESS=posterior["min_bulk_ESS"], diagnostics=posterior["diagnostics"],
                   diagnostic_failure_flags=posterior["diagnostic_failure_flags"],
                   acceptance=posterior["acceptance"],
                   bma_curve=posterior["bma_curve"], q025=posterior["bma_q025"], q975=posterior["bma_q975"])
        row["diagnostic_pass"] = (not row["diagnostic_failure_flags"] and
            (row["max_split_Rhat"] is None or row["max_split_Rhat"] <= 1.05) and
            (row["min_bulk_ESS"] is None or row["min_bulk_ESS"] >= 1500))
        record["methods"].append(row)
        record["status"] = "complete"
    except Exception:
        record.update(status="failed", traceback=traceback.format_exc())
    record["total_wall_seconds"] = time.perf_counter()-start_all
    record["process_peak_working_set_bytes"] = formal.process_peak_working_set()
    formal.atomic_json(target, record)
    return {"job": job, "status": record["status"], "seconds": record["total_wall_seconds"]}

def wilson(success, total):
    if not total:
        return [None, None]
    z = 1.959963984540054
    p = success/total; d = 1+z*z/total
    center = (p+z*z/(2*total))/d
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [center-half, center+half]

def expand_layouts(train,test):
    """Copy split-specific design rows in frozen i modulo 21 order."""
    result=[]
    for template in (train,test):
        if set(np.unique(template[0]))!=set(range(21)):raise ValueError('exactly 21 ordered template IDs required')
        packs=[]
        for person,original in enumerate(MAPPING):
            mask=template[0]==original
            packs.append((np.full(int(mask.sum()),person,dtype=template[0].dtype),*(a[mask].copy() for a in template[1:4])))
        result.append(tuple(np.concatenate([pack[i] for pack in packs]) for i in range(4)))
    return tuple(result)


def synthetic_smoke_design():
    """Engineering-only balanced layouts; never used for formal evidence."""
    person=np.repeat(np.arange(21,dtype=np.int64),64)
    trial=np.tile(np.arange(1,65,dtype=np.int64),21)
    z=(trial.astype(float)-32.5)/np.std(np.arange(1,65))
    contrast=np.where((person+trial)%2,.5,-.5)
    train=(person,trial,z,contrast)
    test=(person.copy(),trial.copy(),z.copy(),-contrast)
    return expand_layouts(train,test)


def load_design(design_path=None,application_input=None):
    if design_path is not None:
        with np.load(design_path,allow_pickle=False) as a:
            train,test=[tuple(a[f'{label}_{i}'].copy() for i in range(4)) for label in ('train','test')]
        n=len(np.unique(train[0]))
        if n==21:return expand_layouts(train,test)
        if n!=35:raise ValueError('design NPZ requires 21 templates or the frozen 35-person expanded design')
        # Validate that 21..34 repeat only the template layouts, retaining split rows.
        for pack in (train,test):
            for person in range(21,35):
                for arr in pack[1:]:
                    if not np.array_equal(arr[pack[0]==person],arr[pack[0]==person%21]):raise ValueError('not the cyclic N35 layout')
        return train,test
    if application_input is None:raise ValueError('provide --design-input NPZ or --application-input extracted OSF directory')
    train,test,metadata=application.prepare_group(application_input,'Low')
    if metadata['participants']!=21:raise ValueError('Low-first source must contain 21 eligible layouts')
    return expand_layouts(train,test)


def pelt_inputs(train,H,h,yqy):return comparator.weighted_statistic(train,N,H,h,yqy)


def constrained_dp(values,weights,m=8,kmax=4):
    cuts=comparator.constrained_dp(values,weights,m,kmax)
    return cuts,comparator.partition_objective(values,weights,cuts,m)


def validate_record_identities(records):
    keys=[(r['cell'],r['rep']) for r in records]
    if len(keys)!=len(set(keys)):raise ValueError('duplicate N35 cell/rep identity')
    for r in records:
        if r['cell'] not in CELLS or not isinstance(r['rep'],int) or not 1<=r['rep']<=500:raise ValueError('invalid N35 cell/rep identity')
        methods=[row['method'] for row in r['methods']]
        if len(methods)!=len(set(methods)):raise ValueError('duplicate N35 method identity')


def aggregate(records,output,stage):
    validate_record_identities(records)
    flat=[];summary=[];coverage=[]
    for cell in CELLS:
        for method in METHODS:
            subset=[dict(stage=stage,cell=cell,rep=r['rep'],data_seed=r['data_seed'],chain_seeds=r['chain_seeds'],source_protocol_sha256=r['protocol_sha256'],**row) for r in records if r['cell']==cell for row in r['methods'] if row['method']==method]
            flat.extend(subset);n=len(subset)
            item=dict(cell=cell,method=method,n=n,planned=500 if stage=='formal' else 1,replication_failures=sum(r['status']!='complete' for r in records if r['cell']==cell))
            if n:
                success=sum(r['correct_K'] for r in subset);item.update(correct_K=success,correct_K_rate=success/n,correct_K_Wilson95=wilson(success,n))
                for key in ('Hausdorff','trajectory_RMSE','test_RMSE','runtime_seconds','segment_level_RMSE','coverage','interval_width'):
                    values=[r[key] for r in subset if r.get(key) is not None]
                    item[key+'_n']=len(values)
                    if values:item.update({key+'_mean':float(np.mean(values)),key+'_median':float(np.median(values)),key+'_MCSE':float(np.std(values,ddof=1)/math.sqrt(len(values))) if len(values)>1 else None})
                biases=[r['segment_level_bias'] for r in subset if r.get('segment_level_bias') is not None]
                if biases:
                    for j in range(len(biases[0])):
                        vals=[b[j] for b in biases];item[f'segment_{j+1}_bias_mean']=float(np.mean(vals));item[f'segment_{j+1}_bias_MCSE']=float(np.std(vals,ddof=1)/np.sqrt(len(vals))) if len(vals)>1 else None
                if cell=='SN0':item['false_cuts_rate']=sum(r['estimated_K']>0 for r in subset)/n
                if method==METHODS[0]:
                    item['diagnostic_failures']=sum(not r['diagnostic_pass'] for r in subset)
                    for j in range(64):
                        inside=[r['coverage_by_bin'][j] for r in subset];width=[r['width_by_bin'][j] for r in subset]
                        coverage.append(dict(cell=cell,bin=j+1,n=n,coverage=float(np.mean(inside)),replication_MCSE=float(np.std(inside,ddof=1)/np.sqrt(n)) if n>1 else None,width=float(np.mean(width))))
            summary.append(item)
    # Arrays/dicts are JSON in CSV, not Python repr; raw JSON records retain all fields.
    comparator.write_csv(output/'small_sample_results.csv',[{k:json.dumps(v) if isinstance(v,(list,dict,np.ndarray)) else v for k,v in r.items()} for r in flat])
    comparator.write_csv(output/'small_sample_summary.csv',summary);comparator.write_csv(output/'small_sample_coverage.csv',coverage)
    formal.atomic_json(output/'small_sample_summary.json',summary)
    return summary


def main(argv=None):
    global OUT,PROTOCOL
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke',action='store_true');mode.add_argument('--full',action='store_true')
    parser.add_argument('--output-root',type=Path,default=OUT);parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--design-input',type=Path,help='NPZ with train_0..3/test_0..3 design arrays only')
    parser.add_argument('--application-input',type=Path,help='Extracted OSF RawData_Experiment1 directory')
    parser.add_argument('--statistic-cache-root',type=Path,help='Optional read-only solver audit cache root containing statistics/N35')
    parser.add_argument('--cache-root',type=Path,help='Read-only legacy N35 root with formal/SN*/R*.json; aggregate without new MCMC')
    args=parser.parse_args(argv)
    if args.workers<1:parser.error('workers must be positive')
    OUT=args.output_root.resolve();PROTOCOL=OUT/'small_sample_manifest.json';stage='smoke' if args.smoke else 'formal'
    if args.cache_root and (OUT==args.cache_root.resolve() or args.cache_root.resolve() in OUT.parents):parser.error('output must be outside read-only cache root')
    synthetic_layout=args.smoke and args.design_input is None and args.application_input is None and args.cache_root is None
    if args.full and args.design_input is None and args.application_input is None:parser.error('--full requires application-derived --application-input or --design-input')
    train,test=synthetic_smoke_design() if synthetic_layout else load_design(args.design_input,args.application_input)
    STATE.update(protocol={'cells':CELLS,'design':{'hashes':design_hashes(train,test)}},train=train,test=test,iterations=500 if args.smoke else 15000,burn=100 if args.smoke else 2000)
    spec=dict(schema_version=1,stage=stage,formal_results=bool(args.full),namespace=NAMESPACE,cells=CELLS,participants=N,T=64,mapping=MAPPING,
        design_hashes=design_hashes(train,test),layout_origin='synthetic engineering smoke only' if synthetic_layout else 'application-derived Low-first layouts',source_hashes=EXECUTION_SOURCE_HASHES,replications_per_cell=1 if args.smoke else 500,
        sampler=dict(chains=3,iterations=STATE['iterations'],burn=STATE['burn'],initial_K=[0,2,4],kmax=4,minseg=8,prior_power=1.),
        data_generation='35 independent N(0,.4^2) intercepts; independent N(0,.4^2) observation errors; all train rows then all test rows',
        seed_protocol='deterministic_seed(namespace, stage|cell|R%06d|DATA or CHAIN1..3)',historical_extension='Replicates 1..200 were initially frozen; 201..500 were later added with the identical namespace and seed rule. This public configuration does not claim all500 were initially preregistered.',python=sys.version,numpy=np.__version__,
        dependency_versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','pandas','ruptures')},input_hashes={str(args.design_input):formal.sha256_file(args.design_input)} if args.design_input else ({str(p):formal.sha256_file(p) for p in application._source_files(args.application_input)} if args.application_input else {}),cache_root=str(args.cache_root.resolve()) if args.cache_root else None,statistic_cache_root=str(args.statistic_cache_root.resolve()) if args.statistic_cache_root else None)
    if args.cache_root:
        cache_paths=[args.cache_root/'formal'/cell/f'R{rep:06d}.json' for cell in CELLS for rep in range(1,501 if args.full else 2)]
        if args.statistic_cache_root:cache_paths += [args.statistic_cache_root/'statistics/N35'/cell/f'R{rep:06d}.npz' for cell in CELLS for rep in range(1,501 if args.full else 2)]
        spec['cache_input_sha256']={str(p.resolve()):formal.sha256_file(p) for p in cache_paths}
    OUT.mkdir(parents=True,exist_ok=True)
    if PROTOCOL.exists() and comparator.read(PROTOCOL)!=spec:raise ValueError('resume manifest fingerprint mismatch; use a new output root')
    formal.atomic_json(PROTOCOL,spec)
    if args.cache_root:
        records=[];hashes={}
        for cell in CELLS:
            for rep in range(1,501 if args.full else 2):
                p=args.cache_root/'formal'/cell/f'R{rep:06d}.json';r=comparator.read(p)
                if r['cell']!=cell or r['rep']!=rep or r['status']!='complete' or r['data_seed']!=seed('formal',cell,rep,'DATA') or r['chain_seeds']!=[seed('formal',cell,rep,f'CHAIN{i}') for i in (1,2,3)]:raise ValueError('legacy cache identity mismatch')
                tr,te,_=simulate(cell,r['data_seed'],train,test)
                if formal.sha256_array(np.r_[tr[-1],te[-1]])!=r['outcomes_hash']:raise ValueError('legacy outcomes identity mismatch')
                if any(row['design_hashes']!=spec['design_hashes'] for row in r['methods']):raise ValueError('legacy design mismatch')
                # Retain source record; update only comparator metrics in this new output.
                import copy
                r=copy.deepcopy(r)
                if args.statistic_cache_root:
                    cache=args.statistic_cache_root/'statistics/N35'/cell/f'R{rep:06d}.npz'
                    with np.load(cache) as a:
                        if str(a['source_sha256'])!=formal.sha256_file(p) or str(a['outcomes_hash'])!=r['outcomes_hash']:raise ValueError('statistic cache identity mismatch')
                        values,weights=a['statistic'].copy(),a['weights'].copy()
                        H,h,yqy=a['H'].copy(),a['h'].copy(),float(a['yqy'])
                else:
                    H,h,yqy=formal.collapsed(*tr,N)
                    values,weights=pelt_inputs(tr,H,h,yqy)
                truth=formal.curve_from(CELLS[cell]['cuts'],CELLS[cell]['levels'])
                for row in r['methods']:
                    if row['method'] not in METHODS[1:]:continue
                    row['archived_runtime_seconds']=row.get('runtime_seconds');row['runtime_seconds']=None
                    cuts=formal.pelt_bic(values,weights,8) if row['method']==METHODS[1] else comparator.constrained_dp(values,weights,8,4)
                    if list(cuts)!=row['estimated_cuts']:
                        curve=formal.fixed_curve(H,h,yqy,len(tr[-1]),cuts)
                        lr,lb=formal.ordered_segment_error(curve,cuts,CELLS[cell]['levels'],len(CELLS[cell]['cuts']))
                        row.update(estimated_cuts=list(cuts),estimated_K=len(cuts),correct_K=int(len(cuts)==len(CELLS[cell]['cuts'])),Hausdorff=formal.hausdorff(cuts,tuple(CELLS[cell]['cuts'])),trajectory_RMSE=float(np.sqrt(np.mean((curve-truth)**2))),test_RMSE=formal.test_rmse(tr,te,curve,N),segment_level_RMSE=lr,segment_level_bias=lb,runtime_seconds=None)
                records.append(r);hashes[str(p)]=formal.sha256_file(p)
        formal.atomic_json(OUT/'cache_provenance.json',hashes)
    else:
        STATE['library'],_=formal.compile_kernel(ROOT/'build')
        jobs=[(stage,cell,rep) for cell in CELLS for rep in range(1,501 if args.full else 2)]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(replicate,jobs):print(json.dumps(result),flush=True)
        records=[comparator.read(OUT/stage/cell/f'R{rep:06d}.json') for _,cell,rep in jobs]
    aggregate(records,OUT,stage)
    return int(any(r['status']!='complete' for r in records))

if __name__=='__main__':raise SystemExit(main())
