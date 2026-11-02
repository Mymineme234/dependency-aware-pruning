"""Explicit migration guard for Stage-00 *legacy stat* dataset inventories.

Historical Stage 00 saved only a global SHA256 of (path,size,mtime_ns).  It
cannot prove which entry subsequently changed.  A cache-specific exception is
therefore an explicitly acknowledged provenance gap, NEVER cryptographic proof
that old image/label contents were identical.  From the migration point onward
we pin SHA256 CONTENT hashes of all current non-cache files and verify them in
Stages 05 and 06.  This module does not alter Stage 00–04 gates or their code.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path

from pruning_common import dataset_fingerprint, hash_object, read_json, sha256_file

VERSION = 1
POLICY = 'EXPLICIT_LEGACY_CACHE_DRIFT_ACKNOWLEDGEMENT'
VERIFIED_POLICY = 'STAGE00_STAT_MATCHED_AT_CONTENT_BASELINE_CREATION'


def _inventory_roots(dataset: dict) -> tuple[list[Path], list[Path]]:
    root = Path(dataset['root'])
    dirs: set[Path] = set()
    direct: set[Path] = set()
    for split_paths in dataset['splits'].values():
        for raw in split_paths:
            p = Path(raw)
            if p.is_dir():
                dirs.add(p)
                if p.name.lower() == 'images' and (p.parent / 'labels').is_dir():
                    dirs.add(p.parent / 'labels')
                if p.parent.name.lower() == 'images' and (p.parent.parent / 'labels' / p.name).is_dir():
                    dirs.add(p.parent.parent / 'labels' / p.name)
            elif p.is_file():
                direct.add(p)
            else:
                raise FileNotFoundError(f'Dataset split missing during inventory: {p}')
    if (root / 'labels').is_dir():
        dirs.add(root / 'labels')
    return sorted(dirs, key=str), sorted(direct, key=str)


def current_content_inventory(dataset: dict) -> dict:
    """Hash all non-cache image/label/manifest *contents*, not cache metadata.

    Caching itself may change `.cache` files, which are listed separately.  Do
    not filter arbitrary other extensions: any data-side file addition/removal
    alters the inventory.  A unique physical path is counted just once.
    """
    roots, direct = _inventory_roots(dataset)
    files: set[Path] = set(direct)
    for directory in roots:
        files.update(p for p in directory.rglob('*') if p.is_file())
    records: list[list] = []
    caches: list[dict] = []
    for p in sorted(files, key=str):
        st = p.stat()
        if p.suffix.lower() == '.cache':
            caches.append({'path': str(p), 'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns})
            continue
        records.append([str(p), int(st.st_size), sha256_file(p)])
    if not records:
        raise RuntimeError('Empty non-cache dataset inventory; refusing legacy bridge')
    # Fingerprint includes content hashes rather than timestamps. Immutable
    # bytes + paths yield same SHA if only file mtime changes.
    return {'mode': 'sha256_path_size_contents_excluding_dot_cache',
            'num_files': len(records), 'manifest_sha256': hash_object(records),
            'cache_files': caches, 'files': records}


def create_legacy_bridge(dataset: dict, stage00_gate: dict, stage00_gate_sha: str,
                         cache_path: str | Path) -> dict:
    original = stage00_gate.get('provenance', {}).get('dataset_manifest', {})
    if original.get('mode') != 'stat_filename_size_mtime_ns':
        raise RuntimeError('Legacy waiver applies only to the original stat-based Stage00 manifest')
    if dataset['yaml_sha256'] != stage00_gate.get('provenance', {}).get('dataset_yaml_sha256'):
        raise RuntimeError('Dataset YAML changed; legacy cache exception is forbidden')
    current = dataset_fingerprint(dataset)
    if current['manifest_sha256'] == original.get('manifest_sha256'):
        raise RuntimeError('Legacy waiver unnecessary: old dataset fingerprint matches')
    # Reject missing/extra files. Restoring the existing old cache path first is
    # mandatory; a presence mismatch cannot be treated as cache metadata drift.
    if current['num_files'] != original.get('num_files'):
        raise RuntimeError(f'Legacy waiver requires identical raw file count: Stage00={original.get("num_files")}, current={current["num_files"]}. Restore removed cache first.')
    path = Path(cache_path).expanduser().resolve()
    dataset_root = Path(dataset['root']).resolve()
    if not path.is_relative_to(dataset_root) or not path.is_file() or path.suffix.lower() != '.cache':
        raise RuntimeError(f'--legacy-cache-path must name an existing .cache file inside {dataset_root}: {path}')
    inventory = current_content_inventory(dataset)
    if str(path) not in {v['path'] for v in inventory['cache_files']}:
        raise RuntimeError(f'Cache path {path} is not included in the active Stage00 dataset inventory')
    return {
        'version': VERSION, 'policy': POLICY,
        'historical_verification': 'UNVERIFIABLE_WITH_STAGE00_GLOBAL_HASH_ONLY',
        'user_acknowledgement': 'Historical non-cache content equality cannot be established from Stage00 aggregate hash. I accept an explicitly logged legacy cache-drift waiver.',
        'stage00_gate_sha256': stage00_gate_sha,
        'dataset_yaml_sha256': dataset['yaml_sha256'],
        'old_stat_manifest': original,
        'migration_stat_manifest': current,
        'named_legacy_cache': str(path),
        'non_cache_inventory': inventory,
    }


def create_verified_bridge(dataset: dict, stage00_gate: dict, stage00_gate_sha: str) -> dict:
    """Automatic safe bridge when Stage00 legacy stat hash STILL matches exactly.

    This establishes a new non-cache content baseline at a moment when the
    original full Stage00 inventory is identical, so later `.cache` rewrites can
    be tolerated without an historical verification gap.
    """
    original = stage00_gate.get('provenance', {}).get('dataset_manifest', {})
    current = dataset_fingerprint(dataset)
    if current['manifest_sha256'] != original.get('manifest_sha256'):
        raise RuntimeError('Cannot create verified bridge: Stage00 stat fingerprint differs')
    if dataset['yaml_sha256'] != stage00_gate.get('provenance', {}).get('dataset_yaml_sha256'):
        raise RuntimeError('Cannot create verified bridge: data YAML differs')
    return {
        'version': VERSION, 'policy': VERIFIED_POLICY,
        'historical_verification': 'STAT_INVENTORY_MATCHED_AT_BASELINE_CREATION',
        'stage00_gate_sha256': stage00_gate_sha,
        'dataset_yaml_sha256': dataset['yaml_sha256'],
        'old_stat_manifest': original,
        'migration_stat_manifest': current,
        'named_legacy_cache': None,
        'non_cache_inventory': current_content_inventory(dataset),
    }


def validate_legacy_bridge(dataset: dict, stage00_gate: dict, stage00_gate_sha: str,
                           bridge: dict, *, compare_to_migration_stat: bool = False) -> dict:
    """Fail-closed for all non-cache bytes and paths post-migration.

    Old historical bytes are unknowable; this only protects the explicit new
    baseline, and leaves Stage00/Stage04 identity and original SHA untouched.
    """
    if bridge.get('version') != VERSION or bridge.get('policy') not in (POLICY, VERIFIED_POLICY):
        raise RuntimeError('Unexpected dataset migration policy/version')
    if bridge.get('stage00_gate_sha256') != stage00_gate_sha:
        raise RuntimeError('Stage00 gate changed since migration')
    if bridge.get('dataset_yaml_sha256') != dataset['yaml_sha256']:
        raise RuntimeError('Dataset YAML changed since migration')
    original = stage00_gate.get('provenance', {}).get('dataset_manifest', {})
    if bridge.get('old_stat_manifest') != original:
        raise RuntimeError('Migration stored different Stage00 manifest')
    expected = bridge.get('non_cache_inventory', {})
    actual = current_content_inventory(dataset)
    if (actual['manifest_sha256'] != expected.get('manifest_sha256') or
            actual['num_files'] != expected.get('num_files')):
        old = {r[0]: r[2] for r in expected.get('files', [])}
        new = {r[0]: r[2] for r in actual['files']}
        added = sorted(set(new) - set(old))[:5]
        removed = sorted(set(old) - set(new))[:5]
        changed = sorted(k for k in old.keys() & new.keys() if old[k] != new[k])[:5]
        raise RuntimeError(f'Non-cache dataset CONTENT inventory changed since migration: added={added}; removed={removed}; modified={changed}')
    if compare_to_migration_stat:
        now = dataset_fingerprint(dataset)
        if now['manifest_sha256'] != bridge.get('migration_stat_manifest', {}).get('manifest_sha256'):
            raise RuntimeError('Legacy migration stat snapshot changed before Stage05 verification')
    return actual


@contextmanager
def scoped_legacy_stage04_guard(stage4, dataset: dict, original_gate: dict,
                                original_gate_sha: str, bridge: dict):
    """Narrowly waive ONLY Stage04's legacy aggregate-hash comparison.

    Other Stage04 ancestry/security checks still run in original unmodified
    source.  The previous stage00 SHA is supplied only to that ONE comparison.
    After the context, the original fingerprint function is restored.
    """
    validate_legacy_bridge(dataset, original_gate, original_gate_sha, bridge)
    true_fingerprint = stage4.dataset_fingerprint
    expected_original = original_gate['provenance']['dataset_manifest']['manifest_sha256']
    def bridge_fingerprint(ds):
        raw = true_fingerprint(ds)
        # The actual on-disk stat fingerprint can drift later due to regenerated
        # caches. Validate CONTENTS for non-cache files on every use.
        validate_legacy_bridge(ds, original_gate, original_gate_sha, bridge)
        if raw['manifest_sha256'] == expected_original:
            return raw
        if raw.get('mode') != 'stat_filename_size_mtime_ns':
            raise RuntimeError('Unrecognized legacy manifest mode')
        return {**raw, 'manifest_sha256': expected_original}
    stage4.dataset_fingerprint = bridge_fingerprint
    try:
        yield
    finally:
        stage4.dataset_fingerprint = true_fingerprint
