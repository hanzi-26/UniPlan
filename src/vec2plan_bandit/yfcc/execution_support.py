"""YFCC-only stage auditing and deadline-safe PostgreSQL connections."""
from contextlib import contextmanager
import math
import resource
import threading
import time
from pathlib import Path

from vec2plan_bandit.execution.timing import evaluation_deadline_ns,record_request_audit,EvaluationDeadlineExceeded


class ExecutionSupport:
    @staticmethod
    def _read_bytes(pid='self'):
        try:
            return int(next(line.split(':')[1] for line in Path(f'/proc/{pid}/io').read_text().splitlines() if line.startswith('read_bytes:')))
        except (OSError,StopIteration,ValueError):return None

    def _init_execution_support(self,config):
        self._conn_lock=threading.Lock()
        self._active_search=None
        self._active_pg=None
        self._audit_key=None
        self.last_audit={}
        self._audit_enabled=config.get('yfcc',{}).get('execution_diagnostics',False)

    def _begin_execution(self):
        if not hasattr(self,'_conn_lock'):self._init_execution_support(getattr(self,'config',{}))
        if not self._audit_enabled:return
        key=evaluation_deadline_ns()
        if key is None or key!=self._audit_key:
            self._audit_key=key
            self.last_audit={'stages_ms':{}}
            self._usage_start=resource.getrusage(resource.RUSAGE_SELF)
            self._io_start=self._read_bytes()
            self._qdrant_pid=getattr(self,'config',{}).get('yfcc',{}).get('qdrant_pid')
            self._qdrant_io_start=self._read_bytes(self._qdrant_pid) if self._qdrant_pid else None

    def _emit_audit(self):
        if not getattr(self,'_audit_enabled',False):return
        usage=resource.getrusage(resource.RUSAGE_SELF)
        self.last_audit.update(minor_faults=usage.ru_minflt-self._usage_start.ru_minflt,
            major_faults=usage.ru_majflt-self._usage_start.ru_majflt,
            user_cpu_ms=1000*(usage.ru_utime-self._usage_start.ru_utime),
            system_cpu_ms=1000*(usage.ru_stime-self._usage_start.ru_stime))
        for label,start,pid in [('process_read_mib',self._io_start,'self'),('qdrant_process_read_mib',self._qdrant_io_start,self._qdrant_pid)]:
            end=self._read_bytes(pid) if start is not None else None
            if end is not None:self.last_audit[label]=(end-start)/1048576
        record_request_audit(yfcc_execution={**self.last_audit,'stages_ms':dict(self.last_audit['stages_ms'])})

    def _audit_values(self,**values):
        if getattr(self,'_audit_enabled',False):self.last_audit.update(values);self._emit_audit()

    @contextmanager
    def _stage(self,name):
        if not getattr(self,'_audit_enabled',False):yield;return
        started=time.perf_counter()
        try:yield
        finally:
            stages=self.last_audit['stages_ms'];stages[name]=stages.get(name,0.)+1000*(time.perf_counter()-started)
            self._emit_audit()

    @property
    def connection(self):
        import psycopg
        from .data import pg_parameters
        if self.conn is None or self.conn.closed:
            values=pg_parameters(self.config['database']['config_path'],database=self.config['database']['name'])
            values['application_name']='yfcc-l2-safe-runtime'
            conn=psycopg.connect(**values,options='-c default_transaction_read_only=on -c search_path=public -c statement_timeout=10000')
            with self._conn_lock:self.conn=conn
        return self.conn

    def cancel_pending(self):
        # The watchdog may cancel, but never rolls back/closes a connection
        # concurrently with its owning SQL thread.
        event=getattr(self,'_active_search',None)
        if event is not None:event.set()
        with self._conn_lock:
            conn=self._active_pg
            if conn is not None and not conn.closed:
                try:conn.cancel_safe(timeout=1)
                except Exception:pass

    def _discard_connection(self,conn):
        with self._conn_lock:
            if self.conn is conn:self.conn=None
            try:conn.close()
            except Exception:pass

    def sql(self,statement,parameters=None,settings=()):
        import psycopg
        from vec2plan_bandit.experiments.collector import _psycopg_sql
        if not hasattr(self,'_conn_lock'):self._init_execution_support(self.config)
        if not getattr(self,'last_audit',None):self._begin_execution()
        conn=self.connection
        with self._conn_lock:self._active_pg=conn
        try:
            with self._stage('pg_select_ms'):
                with conn.cursor() as cursor:
                    if self.config.get('runtime',{}).get('unified_resources',False):
                        cursor.execute('SET LOCAL max_parallel_workers_per_gather=0')
                        cursor.execute('SET LOCAL jit=off')
                    for setting in settings:cursor.execute(setting)
                    deadline=evaluation_deadline_ns()
                    if deadline is not None:
                        remaining=max(1,math.ceil((deadline-time.perf_counter_ns())/1e6))
                        cursor.execute(f'SET LOCAL statement_timeout={remaining}')
                    cursor.execute(_psycopg_sql(statement) if parameters else statement,parameters or None)
                    result=cursor.fetchall()
                with self._conn_lock:conn.rollback()
                return result
        except BaseException as error:
            # SIGALRM can interrupt libpq while it is busy. Rollback on this
            # connection is unsafe: discard it and reconnect next request.
            with self._stage('pg_discard_ms'):self._discard_connection(conn)
            self._audit_values(pg_connection_discarded=True)
            if isinstance(error,psycopg.errors.QueryCanceled) and evaluation_deadline_ns() is not None:
                raise EvaluationDeadlineExceeded() from error
            raise
        finally:
            with self._conn_lock:
                if self._active_pg is conn:self._active_pg=None
