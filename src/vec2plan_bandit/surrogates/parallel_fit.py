"""Offline-only GPU ensemble fitting; inference remains single-threaded CPU."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import numpy as np

_POOLS = {}


def shutdown_training_workers():
    """Release GPU contexts before calibration or timed evaluation begins."""
    for pool in _POOLS.values():
        pool.shutdown(wait=True,cancel_futures=True)
    _POOLS.clear()


def _fit_one(payload):
    features,target,options,seed,device = payload
    from catboost import CatBoostRegressor
    rng=np.random.default_rng(seed)
    indices=rng.integers(0,len(target),size=len(target))
    model=CatBoostRegressor(**options,random_seed=seed,task_type='GPU',devices=str(device),
        gpu_ram_part=.25,verbose=False,allow_const_label=True,allow_writing_files=False,thread_count=1)
    model.fit(features[indices],target[indices])
    return model


def fit_gpu_ensemble(ensemble,features,target):
    devices=ensemble.training_devices
    options=dict(iterations=ensemble.iterations,depth=ensemble.depth,
        learning_rate=ensemble.learning_rate,loss_function='RMSE')
    tasks=[(features,target,options,ensemble.seed+i,devices[i%len(devices)])
           for i in range(ensemble.ensemble_size)]
    # Fixed waves prevent two active jobs being assigned to the same GPU when
    # model durations differ. This pool lives only during an offline fit.
    models=[]
    key=tuple(devices)
    if key not in _POOLS:
        _POOLS[key]=ProcessPoolExecutor(max_workers=len(devices),mp_context=multiprocessing.get_context('spawn'))
    pool=_POOLS[key]
    for start in range(0,len(tasks),len(devices)):
        models.extend(pool.map(_fit_one,tasks[start:start+len(devices)]))
    ensemble.models=models;ensemble.feature_count_=features.shape[1]
    ensemble.task_type='GPU';ensemble.devices=':'.join(devices)
    return ensemble
