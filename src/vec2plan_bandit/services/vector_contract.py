"""Shared encoder and retrieval contract for active experiments."""

MODEL = 'all-MiniLM-L6-v2'
DIMENSION = 384
METRIC = 'cosine'
CLIP_MODEL = 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K'
ENCODER_DIMENSIONS = {MODEL: DIMENSION, CLIP_MODEL: 512}


def validate_vector_contract(config):
    embedding = config['embedding']
    if (config.get('distance_metric') != METRIC or embedding.get('model') not in ENCODER_DIMENSIONS
            or embedding.get('dimension') != ENCODER_DIMENSIONS.get(embedding.get('model'))
            or embedding.get('normalized') is not True):
        raise ValueError('Active retrieval requires cosine, a matching MiniLM/384 or LAION CLIP/512 encoder, and normalized queries')
