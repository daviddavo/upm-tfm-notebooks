from pathlib import Path
from warnings import warn

HYBRID_BASE_PATH = Path('./data/hybrid')
PLN_BASE_PATH = Path('./data/pln')

def _gen_fname(prefix, org_name, splits_freq, normalize, ext='csv', **kwargs) -> str:
    other_args = "-".join([ f"{k}={v}" for k,v in kwargs.items() if v ])
    
    other_args = "-" + other_args if other_args else ""
    return f'{prefix}_{org_name}_{splits_freq}{"_normalize" if normalize else ""}{other_args}.{ext}'

def hybrid_best_hparams(org_name: str, splits_freq: str, normalize: bool, cutoff_date, *, base: Path=HYBRID_BASE_PATH) -> Path:
    base.mkdir(exist_ok=True)
    return base / _gen_fname('best', org_name, splits_freq, normalize, cutoff_date=cutoff_date)

def hybrid_realisitc_hparams(org_name: str, splits_freq: str, normalize: bool, cutoff_date, *, base: Path=HYBRID_BASE_PATH) -> Path:
    base.mkdir(exist_ok=True)
    return base / _gen_fname('realistic', org_name, splits_freq, normalize, cutoff_date=cutoff_date)

def hybrid_recs(org_name: str, splits_freq: str, normalize: bool, cutoff_date, *, base: Path=HYBRID_BASE_PATH) -> Path:
    base.mkdir(exist_ok=True)
    return base / _gen_fname('recommendations', org_name, splits_freq, normalize, cutoff_date=cutoff_date)

def hybrid_results(org_name: str, splits_freq: str, normalize: bool, cutoff_date: str, *, base: Path=HYBRID_BASE_PATH) -> Path:
    base.mkdir(exist_ok=True)
    return base / _gen_fname('results', org_name, splits_freq, normalize, cutoff_date=cutoff_date)

def hybrid_debug(org_name: str, splits_freq: str, normalize: bool, cutoff_date: str, *, base: Path=HYBRID_BASE_PATH) -> Path:
    base.mkdir(exist_ok=True)
    return base / _gen_fname(f'debug', org_name, splits_freq, normalize, cutoff_date=cutoff_date)

def pln_embeddings_cache(model, base: Path = PLN_BASE_PATH):
    import hashlib
    modelhash = hashlib.sha256(str(model).encode())
    
    return base / f'embeddings-{modelhash.hexdigest()}.pkl'

def pln_mdf(org_name, splits_freq, normalize, cutoff_date, *args, base: Path=PLN_BASE_PATH):
    assert not args, "Unrecognized args"

    base.mkdir(exist_ok=True)
    return base / _gen_fname('mdf', org_name, splits_freq, normalize, ext='pkl', cutoff_date=cutoff_date)
