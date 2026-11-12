"""Generate final revision tables from immutable numerical outputs; no MCMC."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
import argparse,csv,hashlib,json,math,sys,time
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def native(x):
    if isinstance(x,dict):return {k:native(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [native(v) for v in x]
    if isinstance(x,np.generic):return native(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def save(p,x):p.write_text(json.dumps(native(x),indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf8')
def mcse(x):return float(np.std(x,ddof=1)/np.sqrt(len(x))) if len(x)>1 else None
def wilson(s,n):
    z=1.959963984540054;p=s/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return float(c-h),float(c+h)
def describe(x):
    x=np.asarray([v for v in x if v is not None],float)
    if not len(x):return dict(n=0,mean=None,mcse=None,median=None,q25=None,q75=None)
    if not np.isfinite(x).all():raise ValueError('Nonfinite numerical table input')
    return dict(n=len(x),mean=float(x.mean()),mcse=mcse(x),median=float(np.median(x)),q25=float(np.quantile(x,.25)),q75=float(np.quantile(x,.75)))
def aggregate_methods(records):
    groups=defaultdict(list)
    for r in records:groups[(r['cohort'],r['cell'],r['method'])].append(r)
    out=[]
    for (cohort,cell,method),g in sorted(groups.items()):
        r=dict(cohort=cohort,cell=cell,method=method,replications=len(g));c=[v['correct_K'] for v in g]
        r.update(correct_K=float(np.mean(c)),correct_K_mcse=mcse(c),correct_K_count=int(sum(c)))
        r['correct_K_wilson_lower'],r['correct_K_wilson_upper']=wilson(sum(c),len(c))
        for metric in ('trajectory_RMSE','test_RMSE','Hausdorff','runtime_seconds'):
            for k,v in describe([x.get(metric) for x in g]).items():r[metric+'_'+k]=v
        for k,v in describe([x.get('segment_level_RMSE') for x in g if x['correct_K']]).items():r['conditional_level_RMSE_'+k]=v
        r['changed_partitions']=sum(bool(x.get('cap_partition_changed')) for x in g);out.append(r)
    return out

def emit(out,name,rows):
    if not rows:raise ValueError('Missing required rows: '+name)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (out/(name+'.csv')).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(native(rows))
    def fmt(x):return '--' if x is None else (f'{x:.8g}' if isinstance(x,float) else str(x))
    text=['| '+' | '.join(fields)+' |','|'+'|'.join('---' for _ in fields)+'|']+['| '+' | '.join(fmt(r.get(k)).replace('|','/') for k in fields)+' |' for r in rows]
    (out/(name+'.md')).write_text('\n'.join(text)+'\n',encoding='utf8')
    def esc(x):return fmt(x).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
    tex=[r'\begin{tabular}{'+'l'*len(fields)+'}',r'\hline',' & '.join(esc(k) for k in fields)+r' \\']+[' & '.join(esc(r.get(k)) for k in fields)+r' \\' for r in rows]+[r'\hline',r'\end{tabular}']
    (out/(name+'.tex')).write_text('\n'.join(tex)+'\n',encoding='utf8')

def manuscript_tables(out,agg,small,coverage,application):
    """Compact, insertion-ready panels, with rounding applied only at rendering."""
    display={'B0':'NoEffect','B1':'ConstEffect','B2':'FixedSplit','A3':'CG-RJMCMC','Constrained DP-BIC':'DP-BIC'}
    def write(name,columns,rows):
        text=[r'\begin{tabular}{'+columns+'}',r'\hline']
        text += [' & '.join(str(v) for v in row)+r' \\' for row in rows]
        text += [r'\hline',r'\end{tabular}']
        (out/name).write_text('\n'.join(text)+'\n',encoding='utf8')
    cells=sorted({r['cell'] for r in agg});index={(r['cell'],r['method']):r for r in agg}
    rows=[]
    for label,metric in [('A: Trajectory RMSE','trajectory_RMSE_median'),('B: Held-out test RMSE','test_RMSE_median')]:
        rows.append([r'\multicolumn{6}{l}{'+label+'}'])
        rows.append(['DGP','NoEffect','ConstEffect','FixedSplit','PELT-BIC','CG-RJMCMC'])
        for c in cells:rows.append([c]+[f"{index[c,m][metric]:.5f}" for m in ('B0','B1','B2','PELT-BIC','A3')])
    rows.append([r'\multicolumn{6}{l}{C: Constrained DP-BIC ($K_{max}=4$)}'])
    rows.append(['DGP','Trajectory RMSE','Test RMSE','Correct $K$','Mean HD','Changed'])
    for c in cells:
        r=index[c,'Constrained DP-BIC'];rows.append([c,f"{r['trajectory_RMSE_median']:.5f}",f"{r['test_RMSE_median']:.5f}",f"{r['correct_K']:.3f}",f"{r['Hausdorff_mean']:.3f}",r['changed_partitions']])
    write('table2.tex','lrrrrr',rows)
    def iqr(r,p,d):return f"{r[p+'_median']:.{d}f} [{r[p+'_q25']:.{d}f}, {r[p+'_q75']:.{d}f}]" if r[p+'_median'] is not None else '--'
    rows=[[r'\multicolumn{7}{l}{A: Main-simulation structural recovery}'],['DGP','Method','Correct $K$','Total','Hausdorff','Level RMSE','Level count']]
    for r in agg:
        if r['method'] in ('A3','PELT-BIC'):rows.append([r['cell'],display.get(r['method'],r['method']),f"{r['correct_K']:.3f}",r['replications'],iqr(r,'Hausdorff',1),iqr(r,'conditional_level_RMSE',4),r['conditional_level_RMSE_n']])
    rows += [[r'\multicolumn{7}{l}{B: Small-sample structural recovery ($N=35$)}'],['Cell','Method',r'Correct $K$ (\%)','Total','Mean HD','Trajectory RMSE',r'Wilson 95\% (\%)']]
    for r in small:rows.append([r['cell'],display.get(r['method'],r['method']),f"{100*r['correct_K']:.1f}",r['replications'],f"{r['Hausdorff_mean']:.2f}",f"{r['trajectory_RMSE_median']:.4f}",f"[{100*r['correct_K_wilson_lower']:.1f}, {100*r['correct_K_wilson_upper']:.1f}]"])
    rows += [[r'\multicolumn{7}{l}{C: Small-sample intervals and diagnostics}'],['Cell','Method',r'Coverage (\%)','MCSE (pp)','Width',r'Max $\widehat R$','Min ESS']]
    for r in coverage:rows.append([r['cell'],'CG-RJMCMC',f"{100*r['coverage']:.2f}",f"{100*r['coverage_mcse']:.2f}",f"{r['mean_width']:.4f}",f"{r['maximum_rhat']:.5f}",f"{r['minimum_bulk_ess']:.0f}"])
    write('table3.tex','llrrrrr',rows)
    rows=[['Group','Method','Test RMSE',r'$\widehat K$','Cuts','Time (s)','IQR (s)']]
    for r in application:
        repeated=r['method']=='PELT-BIC';value=r.get('median_seconds') if repeated else r.get('runtime_seconds');cuts=r.get('cuts',[])
        rows.append([r['group'],r['method'],f"{r['test_RMSE']:.5f}",r.get('K'),', '.join(map(str,cuts)) or '--',f'{value:.6f}' if value is not None else '--',f"[{r['q25_seconds']:.5f}, {r['q75_seconds']:.5f}]" if repeated else '--'])
    write('table5.tex','llrrlrr',rows)


def timing_summary(rows,repeats=31):
    groups=defaultdict(list)
    for r in rows:groups[r['group']].append(r)
    out=[]
    for name,g in sorted(groups.items()):
        if sorted(int(x['repeat']) for x in g)!=list(range(1,repeats+1)):raise ValueError('Expected 31 distinct timing repeats per group')
        x=np.array([float(r['seconds']) for r in g])
        if not np.isfinite(x).all() or np.any(x<=0):raise ValueError('Invalid runtime')
        out.append(dict(group=name,repeats=repeats,median_seconds=float(np.median(x)),q25_seconds=float(np.quantile(x,.25)),q75_seconds=float(np.quantile(x,.75))))
    return out

def time_application(source,raw,out):
    if __package__:
        from . import application as app, simulation as sim
    else:
        import application as app
        import simulation as sim
    from threadpoolctl import threadpool_limits,threadpool_info
    old=read(source);inputs=[];hashes={str(source.resolve()):sha(source)}
    for p in app._source_files(raw):hashes[str(p.resolve())]=sha(p)
    for g in old['groups']:
        name=g['data']['first_pc'];train,_,meta=app.prepare_group(raw,name)
        if meta!=g['data']:raise ValueError('Application input differs from archive')
        n=meta['participants'];H,h,yqy=sim.collapsed(*train,n);inputs.append((name,train,n,H,h,yqy))
    rows=[]
    with threadpool_limits(limits=1):
        pools=threadpool_info()
        for _,tr,n,H,h,yqy in inputs:sim.pelt_fit(tr,n,H,h,yqy,8)
        for repeat in range(1,32):
            for name,tr,n,H,h,yqy in inputs:
                start=time.perf_counter();cuts,_=sim.pelt_fit(tr,n,H,h,yqy,8);elapsed=time.perf_counter()-start
                rows.append(dict(group=name,repeat=repeat,seconds=elapsed,cuts=list(cuts)))
    for p,h in hashes.items():
        if sha(p)!=h:raise RuntimeError('Timing input changed')
    emit(out,'runtime_repeats',rows)
    save(out/'timing_manifest.json',dict(source_sha256=hashes,threadpools=pools,new_CG_draws=0,scope='31 interleaved repeats plus one warmup; timer sim.pelt_fit(train,N,H,h,yqy,8); raw preparation and H/h/yQy excluded'))
    return rows

def normalize_small_method(method,cell,rep):
    """Validate repeated CSV identity fields, then assign canonical metadata."""
    row=dict(method)
    for key in ('cell',):
        if row.get(key) is not None and row[key]!=cell:raise ValueError('Small-sample method cell differs from dataset identity')
    for key in ('rep','replicate'):
        if row.get(key) is not None and row[key]!=rep:raise ValueError('Small-sample method replication differs from dataset identity')
    if row.get('cohort') not in (None,'N35'):raise ValueError('Small-sample method cohort differs')
    if row.get('diagnostic_pass') is not None:
        value=row['diagnostic_pass']
        if isinstance(value,str) and value.strip().lower() in ('true','false'):value=value.strip().lower()=='true'
        if not isinstance(value,(bool,np.bool_)):raise ValueError('diagnostic_pass must be a boolean')
        row['diagnostic_pass']=bool(value)
    row.update(cohort='N35',cell=cell,rep=rep)
    return row


def read_public_small_csv(path):
    """Read the public replication-level CSV, retaining all scientific values."""
    with path.open(encoding='utf8',newline='') as stream:csvrows=list(csv.DictReader(stream))
    bycase=defaultdict(list);seen=set()
    for row in csvrows:
        for key,value in list(row.items()):
            if isinstance(value,str) and value.strip().lower() in ('true','false'):
                row[key]=value.strip().lower()=='true'
            else:
                try:row[key]=json.loads(value) if value else None
                except (ValueError,TypeError):pass
        cell=row['cell'];rep=row.get('replicate') if row.get('replicate') is not None else row.get('rep')
        if cell not in ('SN0','SN1') or isinstance(rep,bool) or not isinstance(rep,int) or rep<1:raise ValueError('Invalid public small-sample dataset identity')
        row=normalize_small_method(row,cell,rep)
        key=(cell,rep,row['method'])
        if key in seen:raise ValueError('Duplicate public small-sample method row')
        seen.add(key);bycase[(cell,rep)].append(row)
    for methods in bycase.values():
        for key in ('data_seed','outcomes_hash','chain_seeds','design_hashes'):
            values=[row[key] for row in methods if row.get(key) is not None]
            if any(value!=values[0] for value in values[1:]):raise ValueError('Small-sample methods have inconsistent '+key)
    return [dict(cell=cell,replicate=rep,methods=methods) for (cell,rep),methods in bycase.items()]


def generate(args):
    out=args.output_root.resolve();sources={};rows=[];originals={};small=[];cov=[]
    roots=[*args.simulation,args.benchmark,args.small_sample,args.application.parent]
    if any(out==p.resolve() or out.is_relative_to(p.resolve()) for p in roots):raise ValueError('Output must be outside immutable input roots')
    def load(p):sources[str(p.resolve())]=sha(p);return read(p)
    found=set()
    comparator_paths=sorted((args.benchmark/'records/primary').glob('*/R*.json'))
    if comparator_paths:
        comparator_records=[load(p) for p in comparator_paths]
    else:
        audit=args.benchmark/'solver_audit.json'
        comparator_records=[r for r in load(audit)['records'] if r['cohort']=='primary']
    for r in comparator_records:
        key=(r['cell'],r['rep'])
        if r.get('source'):source=(ROOT/r['source']).resolve()
        else:
            candidates=[root/f'replicates/main/{key[0]}/rep_{key[1]:06d}/summary.json' for root in args.simulation]
            matching=[p for p in candidates if p.is_file() and sha(p)==r['source_sha256']]
            if len(matching)!=1:raise ValueError('Cannot uniquely identify original summary for solver audit '+str(key))
            source=matching[0].resolve()
        if not any(source.is_relative_to(root.resolve()) for root in args.simulation):raise ValueError('Comparator original is outside --simulation roots: '+str(source))
        original=load(source)
        if key in originals:raise ValueError('Duplicate comparator dataset '+str(key))
        if (original['case']['cell'],original['replicate'])!=key:raise ValueError('Original case identity differs')
        originals[key]=original
        rows.extend(dict(cohort='primary',cell=key[0],rep=key[1],**m) for m in original['methods'] if m['method']!='PELT-BIC')
        for k in ('data_hash','holdout_hash','training_mask_hash'):
            if r.get('provenance') and r['provenance'][k]!=original[k]:raise ValueError('Comparator identity mismatch '+str(key))
        if sha(source)!=r['source_sha256']:raise ValueError('Original summary hash mismatch')
        found.add(key)
        for kind,name in (('corrected_pelt','PELT-BIC'),('constrained_dp','Constrained DP-BIC')):
            m=dict(r[kind]);m['method']=name;rows.append(dict(cohort='primary',cell=key[0],rep=key[1],cap_partition_changed=r['audit']['cap_partition_changed'],**m))
    if not originals or found!=set(originals):raise ValueError('Need matching original and corrected records for every primary dataset')
    corrections={}
    for p in (args.benchmark/'records/N35').glob('*/R*.json'):
        r=load(p);corrections[(r['cell'],r['rep'])]=r
    smallpaths=sorted(args.small_sample.glob('SN*/R*.json'))+sorted(args.small_sample.glob('formal/SN*/R*.json'))
    smallrecords=[]
    for p in smallpaths:
        item=load(p);item.setdefault('replicate',int(p.stem[1:]));smallrecords.append(item)
    if not smallrecords and (args.small_sample/'small_sample_results.csv').is_file():
        csvpath=args.small_sample/'small_sample_results.csv';sources[str(csvpath.resolve())]=sha(csvpath)
        smallrecords=read_public_small_csv(csvpath)
    for s in smallrecords:
        cell=s['cell'];rep=s['replicate']
        for m in s['methods']:
            row=normalize_small_method(m,cell,rep)
            if m['method']=='PELT-BIC' and (cell,rep) in corrections:row=dict(corrections[(cell,rep)]['corrected_pelt']);row['method']='PELT-BIC'
            if 'Constrained' in m['method']:row['method']='Constrained DP-BIC'
            small.append(normalize_small_method(row,cell,rep))
            if m['method'] in ('A3','CG-RJMCMC'):cov.append(normalize_small_method(m,cell,rep))
    if not args.allow_partial:
        if {k:sum(c==k for c,_ in originals) for k in {c for c,_ in originals}}!={f'D{i}':500 for i in range(6)}:raise ValueError('Formal Table 2 needs D0-D5 with 500 datasets each')
        if any(sum(r['cell']==c for r in cov)!=500 for c in ('SN0','SN1')):raise ValueError('Formal Table 3 needs 500 SN0 and SN1 fits')
    out.mkdir(parents=True,exist_ok=True)
    old=load(args.application)
    if args.timing:
        if args.application_input is None:raise ValueError('--timing requires --application-input')
        timing=time_application(args.application,args.application_input,out)
    elif args.runtime_repeats:
        sources[str(args.runtime_repeats.resolve())]=sha(args.runtime_repeats)
        with args.runtime_repeats.open(encoding='utf8',newline='') as f:timing=list(csv.DictReader(f))
    else:raise ValueError('Need --runtime-repeats or --timing --application-input')
    runtime={r['group']:r for r in timing_summary(timing)};application=[]
    for g in old['groups']:
        name=g['data']['first_pc']
        if name not in runtime and name+'-first' not in runtime:raise ValueError('Missing timing group '+name)
        for m in g['methods']:
            row=dict(m);row['group']=name
            if m['method']=='PELT-BIC':
                row.update(runtime.get(name,runtime.get(name+'-first')));row['runtime_seconds']=None;row['runtime_definition']='31-repeat corrected PELT median and IQR, single thread'
                if g.get('PELT_BIC'):row['cuts']=g['PELT_BIC'].get('cuts',m.get('cuts'))
            elif m['method']=='CG-RJMCMC':row['runtime_definition']='archived original single three-chain execution'
            application.append(row)
    fingerprint=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()+Path(__file__).read_bytes()).hexdigest();manifest=out/'table_manifest.json'
    if manifest.exists() and read(manifest)['fingerprint']!=fingerprint:raise ValueError('Output identity differs; use a new output directory')
    agg=aggregate_methods(rows);sagg=aggregate_methods(small)
    for panel,metric in (('a','trajectory_RMSE_median'),('b','test_RMSE_median')):
        emit(out,'table2_panel_'+panel,[{k:r[k] for k in ('cell','method',metric,'replications')} for r in agg if r['method']!='Constrained DP-BIC'])
    emit(out,'table2_panel_c',[r for r in agg if r['method']=='Constrained DP-BIC'])
    emit(out,'table3_primary_structural',[r for r in agg if r['method'] in ('A3','PELT-BIC','Constrained DP-BIC')])
    emit(out,'table3_small_structural',sagg)
    coverage=[]
    for c in sorted({r['cell'] for r in cov}):
        g=[r for r in cov if r['cell']==c];diagnostics=[d for r in g for d in r.get('diagnostics',[])]
        coverage.append(dict(cell=c,replications=len(g),coverage=float(np.mean([r['coverage'] for r in g])),coverage_mcse=mcse([r['coverage'] for r in g]),mean_width=describe([r.get('interval_width',r.get('mean_interval_width',r.get('mean_width'))) for r in g])['mean'],maximum_rhat=max((d.get('split_Rhat',d.get('Rhat',d.get('rhat',0))) or 0 for d in diagnostics),default=None),minimum_bulk_ess=min((d.get('bulk_ESS',d.get('ess_bulk',float('inf'))) or float('inf') for d in diagnostics),default=None),diagnostic_pass_count=sum(r.get('diagnostic_pass') is True for r in g)))
    emit(out,'table3_small_coverage_diagnostics',coverage);emit(out,'table5_application',application)
    manuscript_tables(out,agg,sagg,coverage,application)
    for p,h in sources.items():
        if sha(p)!=h:raise RuntimeError('Table input changed during aggregation')
    save(manifest,dict(fingerprint=fingerprint,source_sha256=sources,script_sha256=sha(__file__),scope='partial_development' if args.allow_partial else 'formal',new_MCMC_runs=0,outputs={p.name:sha(p) for p in out.glob('*') if p.is_file() and p!=manifest}))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--simulation','--simulation-results',type=Path,action='append',required=True)
    p.add_argument('--application','--application-results',type=Path,required=True,help='Original results.json file')
    p.add_argument('--benchmark','--comparator-results',type=Path,required=True,help='Corrected records/primary and records/N35 tree')
    p.add_argument('--small-sample','--small-sample-results',type=Path,required=True,help='Root containing SN0/R*.json and SN1/R*.json')
    p.add_argument('--output-root','--output',type=Path,required=True)
    p.add_argument('--runtime-repeats',type=Path)
    p.add_argument('--timing',action='store_true',help='Execute 31 corrected PELT timings per application group')
    p.add_argument('--application-input',type=Path)
    p.add_argument('--allow-partial',action='store_true')
    generate(p.parse_args())
if __name__=='__main__':main()
