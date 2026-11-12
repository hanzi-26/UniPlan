"""Batch reads of snapshot identity sidecars; never confuse rowid with CTID."""


def fetch_point_values(connection, table, value_column, point_ids):
    # Identifiers come from runtime metadata, not user SQL. Quote them anyway.
    quote = lambda name: '"' + name.replace('"', '""') + '"'
    ids = list(dict.fromkeys(int(i) for i in point_ids))
    result = {}
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        placeholders = ','.join('?' for _ in chunk)
        result.update(connection.execute(
            f'SELECT point_id,{quote(value_column)} FROM {quote(table)} '
            f'WHERE point_id IN ({placeholders})', chunk))
    return result
