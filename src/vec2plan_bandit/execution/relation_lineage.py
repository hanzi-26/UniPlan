"""Offline catalog resolution for indexable view dependencies and CTID maps."""
from __future__ import annotations


def relation_info(conn, schema, table):
    row = conn.execute(
        'SELECT c.oid,c.relkind FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace '
        'WHERE n.nspname=%s AND c.relname=%s', (schema, table)).fetchone()
    if row is None:
        raise ValueError(f'Missing index relation: {schema}.{table}')
    return row


def dependencies(conn, oid):
    return conn.execute(
        "SELECT DISTINCT n.nspname,c.relname,c.relkind FROM pg_rewrite r "
        "JOIN pg_depend d ON d.classid='pg_rewrite'::regclass AND d.objid=r.oid "
        "AND d.refclassid='pg_class'::regclass JOIN pg_class c ON c.oid=d.refobjid "
        "JOIN pg_namespace n ON n.oid=c.relnamespace WHERE r.ev_class=%s "
        "AND c.oid<>r.ev_class AND c.relkind IN ('r','p','m','v')", (oid,)).fetchall()


def index_targets(conn, schema, tables):
    """Resolve real relations, preserving all logical owners for receipts."""
    targets = {}
    def visit(namespace, table, owner, seen):
        identity = (namespace, table)
        if identity in seen:
            raise ValueError(f'Cyclic index dependency: {identity}')
        oid, kind = relation_info(conn, namespace, table)
        if kind in ('r', 'p', 'm'):
            targets.setdefault(identity, set()).add(owner)
        elif kind == 'v':
            deps = dependencies(conn, oid)
            if not deps:
                raise ValueError(f'View has no indexable dependencies: {identity}')
            for child_schema, child, _ in deps:
                visit(child_schema, child, owner, seen | {identity})
        else:
            raise ValueError(f'Unsupported index relation kind {kind}: {identity}')
    for table in sorted(set(tables)):
        namespace=schema
        try:relation_info(conn,namespace,table)
        except ValueError:
            if schema=='public':raise
            # The deployed connection uses experiment_schema,public. Ordinary
            # join tables without synthetic row_id views live in public.
            relation_info(conn,'public',table)
            namespace='public'
        visit(namespace, table, table, set())
    return [(namespace, table, sorted(owners)) for (namespace, table), owners in sorted(targets.items())]


def filter_join_columns(queries):
    """Index predicate/join columns, not large text merely projected in results."""
    from collections import defaultdict
    from sqlglot import exp, parse_one
    result = defaultdict(set)
    for query in queries:
        tree = parse_one(query.source_sql, read='postgres')
        aliases = defaultdict(set)
        for ref in tree.find_all(exp.Table):
            if ref.name in query.columns:
                aliases[ref.alias_or_name].add(ref.name)
        conditions = [node.this for node in tree.find_all(exp.Where)]
        conditions += [node.args['on'] for node in tree.find_all(exp.Join) if node.args.get('on') is not None]
        # USING columns are unqualified but refer only to participating tables.
        conditions += [column for node in tree.find_all(exp.Join) for column in node.args.get('using', [])]
        for condition in conditions:
            columns = [condition] if isinstance(condition, exp.Identifier) else condition.find_all(exp.Column)
            for col in columns:
                name = col.name
                qualifier = col.table if isinstance(col, exp.Column) else ''
                owners = aliases.get(qualifier, set()) if qualifier else set().union(*aliases.values()) if aliases else set()
                for table in owners:
                    if name in query.columns[table]:
                        result[(query.db_id, table)].add(name)
    return dict(result)


def ctid_mapping_lineage(conn, schema, table):
    """Recognize a direct identity-preserving view; fail closed on other SQL.

    Catalog dependencies alone do not prove that a scalar index filters vector
    rows. Also require the actual CTID equality and a unique mapping key.
    """
    from sqlglot import exp, parse_one
    try:oid, kind = relation_info(conn, schema, table)
    except ValueError:
        if schema=='public':raise
        oid,kind=relation_info(conn,'public',table)
    if kind != 'v':
        return None
    definition = conn.execute('SELECT pg_get_viewdef(%s,true)', (oid,)).fetchone()[0]
    tree = parse_one(definition, read='postgres')
    if not isinstance(tree, exp.Select) or any(tree.args.get(k) for k in ('group', 'distinct', 'limit', 'where', 'with_')):
        return None
    source = tree.args.get('from_')
    joins = tree.args.get('joins', [])
    if source is None or not isinstance(source.this, exp.Table) or len(joins) != 1:
        return None
    base, join = source.this, joins[0]
    mapping, equality = join.this, join.args.get('on')
    if not isinstance(mapping, exp.Table) or not isinstance(equality, exp.EQ):
        return None
    if join.args.get('side', '').upper() not in ('', 'LEFT'):
        return None
    columns = (equality.this, equality.expression)
    if not all(isinstance(c, exp.Column) for c in columns):
        return None
    if {(c.table, c.name) for c in columns} != {(base.alias_or_name, 'ctid'), (mapping.alias_or_name, 'source_ctid')}:
        return None
    deps = {(s, t): k for s, t, k in dependencies(conn, oid)}
    def physical(ref):
        matches = [(s, t) for (s, t), k in deps.items()
                   if t == ref.name and (not ref.db or ref.db == s) and k in ('r', 'p')]
        return matches[0] if len(matches) == 1 else None
    physical_base, physical_mapping = physical(base), physical(mapping)
    if physical_base is None or physical_mapping is None:
        return None
    unique = conn.execute(
        'SELECT EXISTS(SELECT 1 FROM pg_index i JOIN pg_class c ON c.oid=i.indrelid '
        'JOIN pg_namespace n ON n.oid=c.relnamespace JOIN pg_attribute a ON a.attrelid=c.oid '
        "AND a.attname='source_ctid' WHERE n.nspname=%s AND c.relname=%s "
        "AND a.atttypid='tid'::regtype AND i.indisunique AND i.indisvalid "
        'AND i.indnkeyatts=1 AND a.attnum=ANY(i.indkey) AND i.indpred IS NULL)', physical_mapping).fetchone()[0]
    if not unique:
        return None
    return dict(base_schema=physical_base[0], base_table=physical_base[1],
                mapping_schema=physical_mapping[0], mapping_table=physical_mapping[1],
                mapping_unique=True, join_key='ctid=source_ctid')
