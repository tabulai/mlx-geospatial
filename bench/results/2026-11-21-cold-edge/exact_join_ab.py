import gc
import itertools
import json
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

import shapely
from compare_dispatch import load_before
import run_e2e as e2e

before = load_before(sys.argv[1])
after = e2e.bench.ms
ref = json.loads(Path(sys.argv[2]).read_text())
replay = e2e.Replay({"input_fingerprints": []}, ref["input_fingerprints"])
replay.prefix()
for case in e2e.CASES:
    points, zones = replay.case(case)
    if case["id"] not in (4, 5):
        continue
    left, right = e2e.frames_from_geometry(shapely.points(points), zones)
    samples = [[], []]
    for trial, order in enumerate(itertools.islice(itertools.cycle(((0, 1), (1, 0))), 9)):
        outputs = [None, None]
        for i in order:
            e2e.bench.ms = (before, after)[i]
            fresh_left, fresh_right = e2e.fresh_inputs(left, right)
            start = perf_counter()
            outputs[i] = e2e.mlx_join(fresh_left, fresh_right)
            duration = (perf_counter() - start) * 1e3
            if trial:
                samples[i].append(duration)
        if trial in (0, 8):
            e2e.validate_outputs(*outputs)
        del outputs
        gc.collect()
    print(json.dumps(dict(
        case=case["label"], vertices=case["vertices"],
        before_ms=median(samples[0]), tuned_ms=median(samples[1]),
        before_samples_ms=samples[0], tuned_samples_ms=samples[1], equality=True,
    )), flush=True)
