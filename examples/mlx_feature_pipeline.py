"""Spatial features straight into an MLX model, without leaving unified memory.

The array-native pattern mlx-spatial is built around:

1. Ingest the polygon layer once (``PolygonSet`` + ``GridIndex``).
2. Re-center the float64 point table once (``PolygonSet.to_local``) so the
   fp32 precision contract holds; the result is a ``LocalPoints`` object that
   lives in unified memory.
3. Per training step, slice a batch of ``LocalPoints``, ask for zone ids, and
   compute distance-to-POI features with ``haversine_cross``; every result is an MLX array
   that feeds the model directly (no NumPy, no copies).

Everything here is synthetic so the script runs anywhere with Metal:

    python3 examples/mlx_feature_pipeline.py
"""

from __future__ import annotations

import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import shapely

import mlx_spatial as ms

rng = np.random.default_rng(0)

# --- 1. a polygon layer: 400 "zones" tiling a Manhattan-sized box -----------
# Real data: zones = gpd.read_file("taxi_zones.geojson").geometry
lon0, lon1, lat0, lat1 = -74.03, -73.90, 40.70, 40.82
gx = np.linspace(lon0, lon1, 21)
gy = np.linspace(lat0, lat1, 21)
zones = np.array(
    [shapely.box(gx[i], gy[j], gx[i + 1], gy[j + 1]) for j in range(20) for i in range(20)],
    dtype=object,
)
polyset = ms.PolygonSet(zones)
index = ms.GridIndex(polyset)
n_zones = len(polyset)

# --- 2. a point table (float64 lon/lat) and a few points of interest ---------
n = 500_000
pickups = np.column_stack([rng.uniform(lon0 - 0.01, lon1 + 0.01, n), rng.uniform(lat0 - 0.01, lat1 + 0.01, n)])
pois = np.array([[-73.9857, 40.7484], [-74.0134, 40.7128], [-73.9772, 40.7527]])  # (lon, lat)
target = rng.normal(size=n).astype(np.float32)  # some label to regress

# Re-center ONCE in float64; from here on the points stay on the device.
local = polyset.to_local(pickups)
# Distance features want raw lon/lat in radians as an MLX array; do that once too.
pickups_mx = mx.array(np.radians(pickups).astype(np.float32))
pois_mx = mx.array(np.radians(pois).astype(np.float32))
target_mx = mx.array(target)


def features(batch: slice) -> tuple[mx.array, mx.array]:
    """Zone id (categorical) and distance-to-POI features (continuous) for a batch."""
    zone = index.query(local[batch])  # lazy MLX int32, computed on the GPU
    # -1 means "no zone"; nn.Embedding would silently read the LAST row for
    # -1, so give "no zone" its own row instead.
    zone = mx.where(zone < 0, n_zones, zone)
    dist = ms.haversine_cross(pickups_mx[batch], pois_mx, degrees=False)  # (B, 3) meters
    return zone, dist / 10_000.0  # scale to ~unit range


class ZoneModel(nn.Module):
    def __init__(self, n_zones: int, n_pois: int, d: int = 16):
        super().__init__()
        self.zone = nn.Embedding(n_zones + 1, d)  # extra row = no zone
        self.head = nn.Sequential(nn.Linear(d + n_pois, 32), nn.ReLU(), nn.Linear(32, 1))

    def __call__(self, zone, dist):
        x = mx.concatenate([self.zone(zone), dist], axis=1)
        return self.head(x).squeeze(-1)


model = ZoneModel(n_zones, len(pois))
optimizer = optim.Adam(learning_rate=1e-3)


def loss_fn(model, zone, dist, y):
    return nn.losses.mse_loss(model(zone, dist), y)


step = nn.value_and_grad(model, loss_fn)

batch_size = 65_536
t0 = time.perf_counter()
for epoch in range(2):
    for start in range(0, n, batch_size):
        batch = slice(start, min(start + batch_size, n))
        zone, dist = features(batch)
        loss, grads = step(model, zone, dist, target_mx[batch])
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
    print(f"epoch {epoch}: loss {loss.item():.4f}")
elapsed = time.perf_counter() - t0

zone_all = index.query(local)
mx.eval(zone_all)
print(
    f"{n:,} points x {n_zones} zones: {2 * n / elapsed:,.0f} points/s through features + "
    f"train step; {int((zone_all >= 0).sum().item()):,} points fell in a zone; "
    f"labels are {type(zone_all).__name__} {zone_all.dtype} in unified memory"
)
