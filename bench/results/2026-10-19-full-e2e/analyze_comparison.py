"""Validate the complete matched runs before producing chart-ready results."""
from pathlib import Path
import csv
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parent
raw = {host: json.loads((ROOT / host / 'e2e-results.json').read_text()) for host in ('m1', 'm4')}
build = json.loads((ROOT / 'build-manifest.json').read_text())
wheel = ROOT / 'dist/mlx_geospatial-0.1.0-py3-none-any.whl'
# The downloadable bundle includes the wheel; the repository evidence keeps
# module hashes and raw results without vendoring a binary distribution.
if wheel.exists():
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == build['wheel_sha256']
assert build['wheel_sha256'] == '923cacf8770e9334c52dfdf0dcce6b557130310abdcc7811706d7643ee8f1273'
ref = raw['m4']
checks = 0
machines = {}
for host, run in raw.items():
    assert '610 passed' in (ROOT / 'previous-validation' / f'{host}-tests.log').read_text()
    assert run['status'] == 'completed', (host, run['status'])
    assert run['correctness_fixture']['status'] == 'passed'
    assert [c['id'] for c in run['cases']] == list(range(6))
    assert run['input_fingerprints'] == ref['input_fingerprints']
    assert run['reference_fingerprints_verified'] == 25
    assert run['protocol'] == ref['protocol']
    meta = run['metadata']
    assert meta['package_python_file_sha256'] == build['python_modules']
    assert meta['package_python_manifest_sha256'] == '0f50eb9f97d215e63fc17e29086ebb18ac5b7912467839f878856fc36022dc02'
    for key in ('versions', 'e2e_runner_sha256', 'bench_sha256', 'runner_sha256', 'geos_version'):
        assert meta[key] == ref['metadata'][key], (host, key)
    assert meta['python_version'].split()[0] == ref['metadata']['python_version'].split()[0] == '3.14.2'
    assert meta['metal_available'] is True
    assert meta['power']['power_source'] == 'AC Power'
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        assert meta['environment'][name] == '1'
    assert meta['environment']['MTL_DEBUG_LAYER'] is None
    gpu = next(g for g in meta['gpu_hardware'] if g.get('sppci_cores'))
    machines[host] = {'name': run['machine_label'], 'cpu_cores': meta['cpu_count'],
                      'gpu_cores': int(gpu['sppci_cores']), 'macos': meta['macos'][0]}
    for case, other in zip(run['cases'], ref['cases']):
        assert case['status'] == 'completed'
        for key in ('id', 'n', 'm', 'vertices', 'output_pair_sha256', 'output_rows', 'input_fingerprints'):
            assert case[key] == other[key], (host, case['id'], key)
        pairs = case['pairs']
        assert len(pairs) == 4
        assert [p['phase'] for p in pairs] == ['warmup'] + ['measured'] * 3
        for pair in pairs:
            assert pair['validation']['status'] == 'passed'
            assert pair['validation']['rows'] == case['output_rows']
            assert pair['validation']['pair_sha256'] == case['output_pair_sha256']
            assert pair['order'] == (['mlx', 'geopandas'] if (case['id'] + pair['trial']) % 2 == 0 else ['geopandas', 'mlx'])
            checks += 1
        for method, timing in case['timings'].items():
            samples = [p['seconds'][method] for p in pairs if p['phase'] == 'measured']
            assert len(samples) == 3 and min(samples) > 0
            assert samples == timing['samples_seconds']
            assert timing['median_seconds'] == statistics.median(samples)
            assert timing['minimum_seconds'] == min(samples)
            assert timing['maximum_seconds'] == max(samples)
            assert timing['range_seconds'] == max(samples) - min(samples)
assert checks == 48
joins = []
for i, case in enumerate(ref['cases']):
    row = {key: case[key] for key in ('id', 'label', 'n', 'm', 'vertices', 'output_rows')}
    row['detail'] = f"{case['vertices']:,} vertices"
    for host in machines:
        timings = raw[host]['cases'][i]['timings']
        ml, gp = timings['mlx'], timings['geopandas']
        row[host] = {'mlx_seconds': ml['median_seconds'], 'geopandas_seconds': gp['median_seconds'],
                     'speedup': gp['median_seconds'] / ml['median_seconds'],
                     'mlx_samples': ml['samples_seconds'], 'geopandas_samples': gp['samples_seconds'],
                     'mlx_range': [ml['minimum_seconds'], ml['maximum_seconds']],
                     'geopandas_range': [gp['minimum_seconds'], gp['maximum_seconds']]}
    row['m1_over_m4_mlx'] = row['m1']['mlx_seconds'] / row['m4']['mlx_seconds']
    joins.append(row)
summary = {}
for host in machines:
    ratios = [row[host]['speedup'] for row in joins]
    summary[host] = {'min': min(ratios), 'max': max(ratios), 'median': statistics.median(ratios),
                     'faster_cases': sum(r > 1 for r in ratios), 'case_count': len(ratios)}
methodology = {
    'date': '2026-10-19', 'repetitions': 3, 'warmups': 1, 'paired_output_checks': checks,
    'output_equality_checked': True,
    'timing': 'Median of three alternating paired measured trials after one untimed full warmup per method and workload.',
    'included': 'Fresh PolygonSet/grid/edge indexes, point extraction, GPU query and materialization, exact float64/GEOS boundary refinement, complete joined GeoDataFrames with attributes, and canonical sorting.',
    'excluded': 'Ready input GeoDataFrame construction/copies, imports and runtime initialization, cold kernel compilation, I/O, correctness validation.',
    'predicate': 'within', 'join': 'inner', 'matching': 'All matching point-polygon pairs.',
    'mlx_adapter': 'Unchanged all-pairs query plus GeoDataFrame attribute assembly; native ms.sjoin is validated separately, not timed here.',
    'cpu_baseline': 'Native geopandas.sjoin(left, right, how=inner, predicate=within), without parallel partitioning.',
    'verification': '48 paired full-output checks passed; all 25 input fingerprints and every output pair hash match between hosts. The exact same final optimized package was used on both.',
    'prior_tests': 'The same unchanged wheel previously passed 610 tests per host with Metal validation enabled; this rerun validates all benchmark outputs afresh.',
    'scope': 'Six synthetic point-in-polygon joins from ready GeoDataFrames; includes index builds, excludes process startup and cold kernel compilation. Different macOS versions across hosts.',
}
data = {'machines': machines, 'joins': joins, 'summary': summary, 'methodology': methodology,
        'wheel_sha256': build['wheel_sha256']}
(ROOT / 'comparison.json').write_text(json.dumps(data, indent=2) + '\n')
(ROOT / 'methodology.txt').write_text('\n\n'.join(f'{key}: {value}' for key, value in methodology.items()) + '\n')
with (ROOT / 'comparison.csv').open('w', newline='') as file:
    writer = csv.writer(file, lineterminator="\n")
    writer.writerow(['machine', 'case', 'vertices', 'output_rows', 'mlx_s', 'geopandas_s', 'geopandas_over_mlx', 'mlx_min_s', 'mlx_max_s', 'geopandas_min_s', 'geopandas_max_s'])
    for row in joins:
        for host in machines:
            v = row[host]
            writer.writerow([machines[host]['name'], row['label'], row['vertices'], row['output_rows'], v['mlx_seconds'], v['geopandas_seconds'], v['speedup'], *v['mlx_range'], *v['geopandas_range']])
print(json.dumps({'paired_output_checks': checks, 'summary': summary, 'joins': joins}, indent=2))
