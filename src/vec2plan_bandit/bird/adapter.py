"""BIRD cosine SQL adapter. Keeps join multiplicity using physical row identities."""
import hashlib,json
from dataclasses import dataclass
from pathlib import Path
import sqlglot
from sqlglot import exp
from vec2plan_bandit.queries.lightweight import cached_plan, parsed_select

def quote(v): return '"'+v.replace('"','""')+'"'
def parse(s):
 t=parsed_select(s)
 if not isinstance(t,exp.Select): raise ValueError('Only SELECT supported')
 if any(t.find(c) for c in (exp.Insert,exp.Delete,exp.Update,exp.Into)): raise ValueError('Read-only SELECT required')
 return t

def relations(s):
 if not s.args.get('from_'):raise ValueError('SELECT without FROM')
 r=[s.args['from_'].this]+[j.this for j in s.args.get('joins',[])]
 if any(not isinstance(x,exp.Table) for x in r):raise ValueError('Derived FROM unsupported')
 return r

def distances(t):
 return [f.parent for f in t.find_all(exp.Anonymous) if f.name.lower()=='lembed']

@dataclass
class Block:
 number:int;table:str;alias:str;column:str;text:str;k:int;asset:dict
 @property
 def asset_key(self):return self.asset['asset_id']

class Query:
 def __init__(self,n,row,metadata,assets):
  self.query_id=f'B{n:03d}';self.db_id=row['db_id'];self.source_sql=row['sql'];self.question=row['question']
  self.source_hash=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()
  self.columns={t:[c for ti,c in metadata['column_names_original'] if ti==i and c!='*'] for i,t in enumerate(metadata['table_names_original'])}
  for table,entries in metadata.get('semantic_rich_column',{}).items():
   for e in entries:
    for name in [e['column_name'],e['column_name']+'_embedding']:
     if name not in self.columns[table]:self.columns[table].append(name)
  if metadata.get('_server_columns'):self.columns=metadata['_server_columns']
  t=parse(self.source_sql);ctes={c.alias_or_name:c.this for c in t.find_all(exp.CTE)}
  if any(s.args.get('distinct') for s in t.find_all(exp.Select)):raise ValueError('DISTINCT SELECT requires explicit identity')
  def base(r,seen=()):
   if r.name in self.columns:return r.name
   if r.name in ctes and r.name not in seen:
    rr=relations(ctes[r.name]);return base(rr[0],(*seen,r.name)) if len(rr)==1 else None
  blocks=[]
  for i,d in enumerate(distances(t)):
   f=d.expression
   if not isinstance(f,exp.Anonymous) or f.name.lower()!='lembed':raise ValueError('Unsupported vector expression')
   model,text=[v.this for v in f.expressions]
   if model!='all-MiniLM-L6-v2':raise ValueError('Unexpected embedding model')
   col=d.this;s=d.find_ancestor(exp.Select)
   targets=[r for r in relations(s) if (not col.table or r.alias_or_name==col.table) and any(a['db_id']==self.db_id and a['table']==base(r) and a['column']==col.name for a in assets)]
   if len(targets)!=1 or not s.args.get('limit'):raise ValueError('Ambiguous vector relation or missing LIMIT')
   r=targets[0];table=base(r);asset=next(a for a in assets if (a['db_id'],a['table'],a['column'])==(self.db_id,table,col.name))
   blocks.append(Block(i,table,r.alias_or_name,col.name,text,int(s.args['limit'].expression.this),asset))
  self.blocks=tuple(blocks);limit=t.args.get('limit');self.k=int(limit.expression.this) if limit else max((b.k for b in blocks),default=0)
  self.group_id=hashlib.sha256(self.semantic_text.encode()).hexdigest()[:16]
  self.predicates=();self.embedding_column=blocks[0].column if blocks else ''
 @property
 def semantic_text(self):return ' '.join(b.text for b in self.blocks)
 def identity(self,s):
  if s.args.get('group'):
   return 'ARRAY['+','.join('('+e.sql(dialect="postgres")+')::text' for e in s.args['group'].expressions)+']'
  return ' || '.join(f'ARRAY[{quote(r.alias_or_name)}.ctid::text]' if r.name in self.columns else f'{quote(r.alias_or_name)}."__bird_ids"' for r in relations(s))
 def tree(self,ids_only=True):
  t=parse(self.source_sql);ctes={c.alias_or_name:c.this for c in t.find_all(exp.CTE)}
  def columns(name,seen=()):
   if name in self.columns:return self.columns[name]
   if name in seen or name not in ctes:raise ValueError('Unknown wildcard lineage')
   return [e.alias_or_name for e in expanded(ctes[name],(*seen,name))]
  def expanded(s,seen=()):
   out=[]
   for e in s.expressions:
    if isinstance(e,exp.Star) or isinstance(e,exp.Column) and isinstance(e.this,exp.Star):
     for r in relations(s):
      if isinstance(e,exp.Column) and e.table!=r.alias_or_name:continue
      out.extend(exp.column(c,table=r.alias_or_name,quoted=True) for c in columns(r.name,seen))
    else:out.append(e.copy())
   return out
  sels=[c.this for c in t.find_all(exp.CTE)]+[t];projections={id(s):expanded(s) for s in sels}
  for s in sels:
   if s is t and ids_only:
    expressions=[exp.alias_(exp.Var(this=self.identity(s)),'__bird_ids',quoted=True)]
    # Preserve ORDER BY projection aliases, but omit them from recall identities.
    expressions+= [e for e in projections[id(s)] if e.alias_or_name and e.alias_or_name not in ('*','__bird_ids')]
    s.set('expressions',expressions)
   else:
    s.set('expressions',projections[id(s)]);s.append('expressions',exp.alias_(exp.Var(this=self.identity(s)),'__bird_ids',quoted=True))
  for i,d in enumerate(distances(t)):
   d.expression.replace(exp.Var(this=f'%(v_{i})s::vector'))
   s=d.find_ancestor(exp.Select)
   if s.args.get('order'):s.args['order'].append('expressions',exp.Ordered(this=exp.Var(this=self.identity(s))))
  if t.args.get('order') and not any(v.find_ancestor(exp.Select) is t for v in t.find_all(exp.Var) if v.this.startswith('%(v_')):
   t.args['order'].append('expressions',exp.Ordered(this=exp.Var(this=self.identity(t))))
  if getattr(self,'deterministic_limit_ordering',False):
   for s in sels:
    if s.args.get('limit') and not s.args.get('order'):
     s.set('order',exp.Order(expressions=[exp.Ordered(this=exp.Var(this=self.identity(s)))]))
  return t
 def binds(self,vectors):return {f'v_{i}':'['+','.join(str(float(v)) for v in vec)+']' for i,vec in enumerate(vectors)}
 def render(self,vectors,candidates=None,exact=False,ids_only=True):
  statement=cached_plan(self,('render',candidates is not None,exact,ids_only),
    lambda:self._render_template(candidates is not None,exact,ids_only))
  binds=self.binds(vectors)
  if candidates is not None:
   binds.update({f'keys_{b.number}':[json.dumps(x,ensure_ascii=False) for x in candidates[b.number]] for b in self.blocks})
  return statement,binds
 def _render_template(self,with_candidates,exact,ids_only):
  t=self.tree(ids_only)
  # Locate operators via placeholders after lembed has been replaced.
  ds=[v.parent for v in t.find_all(exp.Var) if v.this.startswith('%(v_')]
  for b,d in zip(self.blocks,ds):
   if with_candidates:
    s=d.find_ancestor(exp.Select)
    fields=b.asset['primary_key'] or ['ctid'];parts=[]
    for key in fields:parts.extend(["'"+key.replace("'","''")+"'",f'{quote(b.alias)}.{quote(key)}'+('::text' if key=='ctid' else '')])
    predicate=f'jsonb_build_object({",".join(parts)}) = ANY(%(keys_{b.number})s::jsonb[])'
    s.where(exp.Var(this=predicate),append=True,copy=False)
   if exact:d.replace(exp.Paren(this=exp.Add(this=exp.Paren(this=d.copy()),expression=exp.Literal.number('0.0'))))
  from vec2plan_bandit.queries.cosine_sql import DIALECT
  return t.sql(dialect=DIALECT)
 def eligible_sql(self,b,vectors):
  statement=cached_plan(self,('eligible',b.number),lambda:self._eligible_template(b))
  return (statement,self.binds(vectors)) if statement is not None else None
 def _eligible_template(self,b):
  t=self.tree(False);ds=[v.parent for v in t.find_all(exp.Var) if v.this.startswith('%(v_')];s=ds[b.number].find_ancestor(exp.Select).copy()
  r=next(r for r in relations(s) if r.alias_or_name==b.alias)
  if r.name!=b.table:raise ValueError('Eligibility over derived CTE requires explicit key propagation')
  if not s.args.get('where') and not s.args.get('joins'):return None
  fields=b.asset['primary_key'] or ['ctid']
  s.set('expressions',[exp.Var(this=f'{quote(b.alias)}.{quote(k)}'+('::text' if k=='ctid' else '')) for k in fields]);s.set('distinct',exp.Distinct())
  for field in ('order','limit','offset'):s.set(field,None)
  if t.args.get('with_'):s.set('with_',t.args['with_'].copy())
  from vec2plan_bandit.queries.cosine_sql import DIALECT
  return s.sql(dialect=DIALECT)
