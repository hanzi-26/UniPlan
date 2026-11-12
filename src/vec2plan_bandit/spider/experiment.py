"""Spider: fresh training with Wiki v7 calibration and frozen nine-method evaluation."""
import hashlib
import json
from pathlib import Path
import numpy as np

from .adapter import Query
from vec2plan_bandit.bird.experiment import Experiment as BirdExperiment
from vec2plan_bandit.bird.shared_experiment import SharedExperiment, write, read_rows
from vec2plan_bandit.experiments.wiki_optional_acceleration import WikiOptionalExperiment
from vec2plan_bandit.queries.wikipedia_sql_adapter import semantic_query_groups
from vec2plan_bandit.queries.grouped_stratification import stratified_group_split
from vec2plan_bandit.policies.optional_acceleration import describe_query
from .runtime import Runtime


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class Experiment(SharedExperiment):
    features=BirdExperiment.features
    prepare=BirdExperiment.prepare
    calibration_actions=WikiOptionalExperiment.calibration_actions
    measurement_router=WikiOptionalExperiment.measurement_router
    calibrate=WikiOptionalExperiment.calibrate
    load_frozen=WikiOptionalExperiment.load_frozen
    benchmark=WikiOptionalExperiment.benchmark

    def __init__(self,path):
        self.dataset=json.loads(Path(path).read_text())
        self.config=json.loads(Path(self.dataset['experiment_config']).read_text())
        self.output=Path(self.dataset['output_dir']);self.output.mkdir(parents=True,exist_ok=True)
        self.worker=None;self.acceleration_router=None;self.disabled_acceleration=False;self.budget_faiss=False
        self.prepared={r['query_id']:r for r in read_rows(self.output/'prepare/prepared.jsonl')}
        assets=json.loads(Path(self.dataset['catalog']).read_text())['assets']
        self.corpus_sizes={a['asset_id']:a['estimated_points'] for a in assets}
        # Preserve released point identity rather than equating SQLite rowid with PG CTID.
        assets=[{**a,'primary_key':['rowid']} for a in assets]
        metadata={r['db_id']:r for r in json.loads(Path(self.dataset['metadata']).read_text())}
        columns=json.loads(Path(self.dataset['schema']).read_text())
        for db,cs in columns.items():metadata[db]['_server_columns']=cs
        source=json.loads(Path(self.dataset['source']).read_text());self.queries=[];excluded=[]
        for i,row in enumerate(source,1):
            try:
                q=Query(i,row,metadata[row['db_id']],assets);q.query_id=f'S{i:03d}'
                vectors=[np.zeros(384) for _ in q.blocks]
                q.render(vectors,exact=True)
                for b in q.blocks:q.eligible_sql(b,vectors)
                self.queries.append(q)
            except Exception as e:
                excluded.append({'query_id':f'S{i:03d}','db_id':row['db_id'],'reason':str(e),'source_hash':sha_row(row)})
        write(self.output/'query-adapter-audit.json',{'source_queries':len(source),'accepted':len(self.queries),'excluded':excluded,'exclusion_basis':'missing released vector columns or unsupported SQL, determined before quality/timing measurements'})
        if [e['query_id'] for e in excluded] != self.dataset['allowed_missing_queries']:
            raise ValueError('Unexpected query coverage change; inspect query-adapter-audit.json')
        self.routable=[q for q in self.queries if q.blocks];self.by_id={q.query_id:q for q in self.queries}
        labels={}
        for q in self.routable:
            n=sum(self.corpus_sizes[b.asset_key] for b in q.blocks)
            desc=describe_query(q,{'candidate_count':n,'estimated_selectivity':1.},corpus_size=n)
            labels[q.query_id]=desc['key'] if desc else 'unsupported_structure'
        split_path=Path(self.dataset['splits'])
        if split_path.exists():self.split=json.loads(split_path.read_text())
        else:
            self.split,audit=stratified_group_split(semantic_query_groups(self.routable),labels,seed=self.dataset['seed'])
            write(split_path,self.split);write(self.output/'split-audit.json',{**audit,'strata':'SQL structure, corpus size, k; no quality or latency outcomes','semantic_group_disjoint':True})
        self.binding={'source_sha256':sha(self.dataset['source']),'splits_sha256':sha(split_path),'catalog_sha256':sha(self.dataset['catalog']),'schema_sha256':sha(self.dataset['schema']),'encoder':self.config['embedding']}
        implementation=[Path(__file__),Path(__file__).with_name('runtime.py'),Path(__file__).with_name('assets.py'),Path(__file__).with_name('adapter.py')]
        contract={'dataset':self.dataset,'config':self.config,'binding':self.binding,'source_queries':len(source),'vector_queries':len(self.routable),'excluded':excluded,'split_counts':{r:len(ids) for r,ids in self.split.items()},'test_labels_used_for_training_or_calibration':False,'methods':self.dataset['evaluation_methods'],'implementation_sha256':{str(p):sha(p) for p in implementation},'baseline_sha256':sha(self.dataset['baselines'])}
        target=self.output/'run_contract.json'
        if target.exists() and json.loads(target.read_text())!=contract:raise ValueError('Run contract changed; use new output directory')
        write(target,contract)

    def runtime(self):
        if self.worker is None:self.worker=Runtime(self.config)
        return self.worker
    def request(self,query,*args,**kwargs):
        self.runtime().prepare_query(query)
        return SharedExperiment.request(self,query,*args,**kwargs,acceleration_router=None if self.disabled_acceleration else self.acceleration_router)
    def describe(self,query,stats):
        return describe_query(query,stats.get(query.query_id),corpus_size=sum(self.corpus_sizes[b.asset_key] for b in query.blocks))
    def plan(self):
        write(self.output/'experiment_plan.json',{'source_queries':len(self.queries)+len(self.dataset['allowed_missing_queries']),'vector_queries':len(self.routable),'splits':{r:len(ids) for r,ids in self.split.items()},'recall':.8,'quality_probability_min':.8,'evaluation_budget_ms':200,'training_budget_ms':10000,'methods':self.dataset['evaluation_methods'],'encoder':self.config['embedding']})
    def report(self):
        import csv
        results={r:json.loads((self.output/r/'summary.json').read_text()) for r in ('dev','test') if (self.output/r/'summary.json').exists()}
        write(self.output/'results.json',results)
        rows=[{'split':r,'method_id':m,**{k:v for k,v in s.items() if not isinstance(v,(dict,list))}} for r,ms in results.items() for m,s in ms.items()]
        if rows:
            with (self.output/'comparison.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))));writer.writeheader();writer.writerows(rows)
        (self.output/'report.md').write_text('Spider MiniLM/384 cosine experiment. Recall >=0.8; main quality probability threshold 0.8. Training budget 10000ms, evaluation budget 200ms. Fresh Spider train model; matrix action features and shared Pool inference. FAISS parameters selected on train, accepted/rejected on dev with complete-k, recall, deadline and >5% mean E2E gain gates. Calibration is empirical support, not a probability guarantee. Dev is a calibration split; frozen test is the primary result. All nine methods measured in one interleaved schedule, five repeats and fresh embedding. Missing released vector columns are listed in query-adapter-audit.json; relational-only queries are audited but not vector benchmarks.\n')

def sha_row(row):return hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()
