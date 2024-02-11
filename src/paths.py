from pathlib import Path

BASELINE_BASE_PATH = Path('./data/baseline')
HYBRID_BASE_PATH = Path('./data/hybrid')
PLN_BASE_PATH = Path('./data/pln')

def baseline_mp(org_name: str, n_splits: int, base: Path=BASELINE_BASE_PATH) -> Path:
    base.mkdir(exist_ok=True)
    return base / f'mp-{org_name}-{n_splits}.csv'

def _gen_fname(prefix, org_name, splits_freq, normalize, ext='csv') -> str:
    return f'{prefix}-{org_name}-{splits_freq}{"-normalize" if normalize else ""}.{ext}'

def perfect_mp_freq(org_name: str, splits_freq: str, normalize: bool, base: Path = BASELINE_BASE_PATH) -> Path:
    base.mkdir(exist_ok=True)
    return base / _gen_fname('perfect-freq', org_name, splits_freq, normalize)

def baseline_mp_freq(org_name: str, splits_freq: str, normalize: bool, base: Path = BASELINE_BASE_PATH) -> Path:
    base.mkdir(exist_ok=True)
    return base / _gen_fname('mp-freq', org_name, splits_freq, normalize)

def hybrid_best_hparams(org_name: str, splits_freq: str, normalize: bool, base: Path=HYBRID_BASE_PATH) -> Path:
    base.mkdir(exist_ok=True)
    return base / _gen_fname('best', org_name, splits_freq, normalize)

def hybrid_realisitc_hparams(org_name: str, splits_freq: str, normalize: bool, base: Path=HYBRID_BASE_PATH) -> Path:
    base.mkdir(exist_ok=True)
    return base / _gen_fname('realistic', org_name, splits_freq, normalize)

def hybrid_recs(org_name: str, splits_freq: str, normalize: bool, base: Path=HYBRID_BASE_PATH) -> Path:
    base.mkdir(exist_ok=True)
    return base / _gen_fname('recommendations', org_name, splits_freq, normalize)

def hybrid_results(org_name: str, splits_freq: str, normalize: bool, merge_func: str, base: Path=HYBRID_BASE_PATH) -> Path:
    base.mkdir(exist_ok=True)
    return base / _gen_fname(f'results-{merge_func}', org_name, splits_freq, normalize)

def hybrid_debug(org_name: str, splits_freq: str, normalize: bool, merge_func: str, base: Path=HYBRID_BASE_PATH) -> Path:
    base.mkdir(exist_ok=True)
    return base / _gen_fname(f'debug-{merge_func}', org_name, splits_freq, normalize)

def pln_embeddings_cache(model, base: Path = PLN_BASE_PATH):
    import hashlib
    modelhash = hashlib.sha256(str(model).encode())
    
    return base / f'embeddings-{modelhash.hexdigest()}.pkl'

def pln_mdf(org_name, splits_freq, normalize, base: Path=PLN_BASE_PATH):
    base.mkdir(exist_ok=True)
    return base / _gen_fname('mdf', org_name, splits_freq, normalize, ext='pkl')
