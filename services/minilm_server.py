"""Local, offline MiniLM HTTP encoder; one model and serialized inference."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8333)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-seq-length', type=int, default=256)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    import torch
    from sentence_transformers import SentenceTransformer
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing silent CPU fallback')
    model = SentenceTransformer(str(args.model_path), device=args.device, local_files_only=True)
    # Match the deployed MiniLM encoder and the pinned model configuration.
    model.max_seq_length = args.max_seq_length
    torch.set_num_threads(args.threads)
    lock = Lock()
    name = 'all-MiniLM-L6-v2'

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != '/health':
                return self.reply(404, {'error': 'unknown endpoint'})
            self.reply(200, {'status': 'ok', 'loaded_models': [name],
                'dimension': 384, 'normalized': True, 'max_seq_length': args.max_seq_length,
                'revision': 'c9745ed1d9f207416be6d2e6f8de32d1f16199bf',
                'device': str(model.device),
                'gpu': torch.cuda.get_device_name(model.device) if model.device.type == 'cuda' else None})

        def do_POST(self):
            if self.path != '/embed':
                return self.reply(404, {'error': 'unknown endpoint'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 1_000_000:
                    return self.reply(413, {'error': 'invalid request size'})
                payload = json.loads(self.rfile.read(size))
                texts = payload.get('texts')
                if payload.get('model') != name or not isinstance(texts, list) or not 1 <= len(texts) <= 256 or not all(isinstance(t, str) for t in texts):
                    return self.reply(400, {'error': 'expected model and 1–256 texts'})
                with lock:
                    vectors = model.encode(texts, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)
                self.reply(200, {'model': name, 'embeddings': vectors.tolist()})
            except (ValueError, TypeError):
                self.reply(400, {'error': 'invalid JSON request'})
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(json.dumps({'stage': 'ready', 'port': args.port, 'device': str(model.device)}), flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
