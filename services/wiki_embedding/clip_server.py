"""Offline LAION CLIP text encoder matching the official Wiki vector space."""
import argparse
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

MODEL = 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8341)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    import torch
    from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast
    torch.set_num_threads(args.threads)
    model = CLIPTextModelWithProjection.from_pretrained(args.model_path, local_files_only=True).to(args.device).eval()
    tokenizer = CLIPTokenizerFast.from_pretrained(args.model_path, local_files_only=True)
    if model.config.projection_dim != 512:
        raise ValueError('Official Wiki CLIP must project to 512 dimensions')
    digest = hashlib.sha256()
    with (args.model_path/'model.safetensors').open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    weights_sha256 = digest.hexdigest()
    lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, payload):
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != '/health':
                return self.reply(404, {'error': 'unknown endpoint'})
            self.reply(200, {'status': 'ok', 'loaded_models': [MODEL], 'dimension': 512,
                'normalized': True, 'max_seq_length': 77, 'device': args.device,
                'weights_sha256': weights_sha256, 'protocol': 'wiki-laion-clip-text-v1'})

        def do_POST(self):
            if self.path != '/embed':
                return self.reply(404, {'error': 'unknown endpoint'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 1000000:
                    return self.reply(413, {'error': 'invalid request size'})
                payload = json.loads(self.rfile.read(size))
                texts = payload.get('texts')
                if (payload.get('model') != MODEL or not isinstance(texts, list) or
                        not 1 <= len(texts) <= 256 or not all(isinstance(t, str) for t in texts)):
                    return self.reply(400, {'error': 'Expected official CLIP model and texts'})
                with lock, torch.inference_mode():
                    tokens = tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors='pt').to(args.device)
                    matrix = model(**tokens).text_embeds
                    matrix = torch.nn.functional.normalize(matrix, dim=-1)
                    if matrix.shape != (len(texts), 512) or not torch.isfinite(matrix).all():
                        raise ValueError('Invalid CLIP output')
                    vectors = matrix.cpu().float().tolist()
                self.reply(200, {'model': MODEL, 'embeddings': vectors})
            except (ValueError, TypeError):
                self.reply(400, {'error': 'invalid request'})
            except (BrokenPipeError, ConnectionResetError):
                pass
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(json.dumps({'stage': 'ready', 'port': args.port, 'model': MODEL}), flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
