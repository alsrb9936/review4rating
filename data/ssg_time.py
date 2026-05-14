from __future__ import annotations

import numpy as np


def ssg_time_handler(review_timestamps: object, current_timestamp: float, percentile: float, max_rel: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Original SSG per-sample temporal bucketing.

    Mirrors ``PartitionDataset.time_handler`` from the reference implementation:
    only reviews strictly earlier than the target interaction are valid, position
    indices are reversed over valid reviews, and relative-time buckets are
    computed from this sample's adjacent time gaps rather than from a global
    train-set scale.
    """
    times = np.asarray(review_timestamps).reshape(-1).tolist()
    cur_time = float(current_timestamp)
    renum = 0
    pos_ind: list[int] = []
    rel_dt: list[float] = []
    abs_dt: list[float] = []

    for timestamp in times:
        t = float(timestamp)
        if t < cur_time:
            delta = cur_time - t
            rel_dt.append(delta)
            abs_dt.append(delta)
            pos_ind.append(renum)
            renum += 1
        else:
            rel_dt.append(0.0)
            abs_dt.append(0.0)
            pos_ind.append(0)

    for idx in range(len(pos_ind)):
        if idx < renum:
            pos_ind[idx] = renum - pos_ind[idx]

    deltas: list[float] = []
    for idx in range(1, len(rel_dt)):
        deltas.append(rel_dt[idx - 1] - rel_dt[idx])
        if rel_dt[idx] == 0:
            break

    non_zero = np.asarray([delta for delta in deltas if delta != 0], dtype=np.float64)
    scale = max(float(np.percentile(non_zero, percentile)), 1.0) if non_zero.size > 0 else 1.0
    rel_bucket = [min(int(delta // scale), int(max_rel)) for delta in rel_dt]

    return (
        int(renum),
        np.asarray(pos_ind, dtype=np.int64),
        np.asarray(rel_bucket, dtype=np.int64),
        np.asarray(abs_dt, dtype=np.float32),
    )
