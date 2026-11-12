"""Shared non-overlapping execution stages; server time remains nested."""
from contextlib import contextmanager
import time

STAGES=('semantic_check_ms','sql_compile_ms','filter_prepare_ms','ann_search_ms','candidate_filter_ms',
        'id_mapping_ms','result_assembly_ms','pg_execute_ms')

class ExecutionTiming:
    def __init__(self):
        self.spans={k:0. for k in STAGES};self.stack=[];self.start=time.perf_counter_ns()
    @contextmanager
    def span(self,name):
        if name not in self.spans:raise ValueError('Unknown execution stage')
        frame=[name,time.perf_counter_ns(),0.];self.stack.append(frame)
        try:yield
        finally:
            elapsed=(time.perf_counter_ns()-frame[1])/1e6
            self.stack.pop();self.spans[name]+=max(0.,elapsed-frame[2])
            if self.stack:self.stack[-1][2]+=elapsed
    def finish(self):
        total=(time.perf_counter_ns()-self.start)/1e6
        return {**self.spans,'execution_unattributed_ms':max(0.,total-sum(self.spans.values()))}


def instrument_runtime(worker):
    """Request-scoped timing with guarded SQL nesting and interruption auditing."""
    if getattr(worker,'_detailed_timing_instrumented',False):return
    from functools import wraps
    from vec2plan_bandit.execution.timing import record_request_audit
    for method,stage in (('sql','pg_execute_ms'),('eligible','filter_prepare_ms'),('keys','id_mapping_ms')):
        if not hasattr(worker,method):continue
        original=getattr(worker,method)
        def wrapped(*args,_original=original,_stage=stage,**kwargs):
            timing=getattr(worker,'_execution_timing',None)
            if timing is None:return _original(*args,**kwargs)
            with timing.span(_stage):return _original(*args,**kwargs)
        setattr(worker,method,wrapped)
    original=worker.execute
    @wraps(original)
    def execute(*args,**kwargs):
        timing=ExecutionTiming();worker._execution_timing=timing
        try:
            row=original(*args,**kwargs)
            row['execution_stages_ms']=timing.finish()
            worker.last_execution_stages_ms=row['execution_stages_ms']
            return row
        finally:
            record_request_audit(execution_stages_ms=timing.finish())
            worker._execution_timing=None
    worker.execute=execute
    if hasattr(worker,'native'):
        native_original=worker.native
        def native(*args,**kwargs):
            timing=ExecutionTiming();worker._execution_timing=timing
            try:
                row=native_original(*args,**kwargs);row['execution_stages_ms']=timing.finish()
                return row
            finally:
                record_request_audit(execution_stages_ms=timing.finish());worker._execution_timing=None
        worker.native=native
    worker._detailed_timing_instrumented=True


def stage(worker,name):
    from contextlib import nullcontext
    timing=getattr(worker,'_execution_timing',None)
    return timing.span(name) if timing is not None else nullcontext()
