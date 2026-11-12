"""Spider-specific database and sharded ACORN routing."""
import json
from pathlib import Path
from qdrant_client import QdrantClient
from vec2plan_bandit.bird.runtime import Runtime as SharedRuntime

class RoutedQdrant:
    def __init__(self, routes):
        self.routes=json.loads(Path(routes).read_text())['collections']
        self.clients={}
    def query_points(self,collection,*args,**kwargs):
        # Shared runtime names its collections for BIRD; Spider release uses this suffix.
        name='spider_'+collection.removeprefix('bird_').removesuffix('_acorn_v1')+'_cosine_acorn_v1'
        url=self.routes[name]
        if url not in self.clients:self.clients[url]=QdrantClient(url=url,timeout=60,trust_env=False)
        return self.clients[url].query_points(name,*args,**kwargs)
    def close(self):
        for client in self.clients.values():client.close()

class Runtime(SharedRuntime):
    def __init__(self,config):
        super().__init__(config)
        self.qdrant.close()
        self.qdrant=RoutedQdrant(config['acorn']['routes'])
    def open(self,db):
        prefix=self.config['database']['prefix']
        return super().open(db if db.startswith(prefix) else prefix+db)
