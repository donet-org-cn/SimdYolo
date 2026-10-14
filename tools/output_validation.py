"""共享的数值校验：端到端 TopK 必须匹配唯一候选及其全部附加字段。"""
import numpy as np


def compare(actual, expected, end_to_end, dense=None, nc=0):
    """返回配对后的实际值、参考值与是否通过；允许 TopK 合法同分排序差异。"""
    if actual.shape != expected.shape:
        raise AssertionError(f"Shape mismatch {actual.shape} != {expected.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return actual, expected, False
    if not end_to_end:
        return actual, expected, bool(np.allclose(actual, expected, atol=3e-3, rtol=3e-4))
    if actual.ndim != 3 or actual.shape[2] < 6:
        raise AssertionError("End-to-end output must have shape [batch, selected, 6 + extras]")
    if np.any(actual[:, :, 5] != np.floor(actual[:, :, 5])):
        return actual, expected, False
    if dense is not None:
        if nc <= 0 or dense.ndim != 3 or dense.shape[0] != actual.shape[0] or dense.shape[2] != nc + actual.shape[2] - 2:
            raise AssertionError("Dense decoded candidates do not match batch/classes/extra fields")
        if not np.isfinite(dense).all():
            return actual, expected, False
    ordered = np.empty_like(actual)
    for batch in range(actual.shape[0]):
        if dense is not None:
            if not np.allclose(np.sort(actual[batch, :, 4]), np.sort(expected[batch, :, 4]), atol=2e-7, rtol=3e-4):
                print("TopK score mismatch", float(np.max(np.abs(np.sort(actual[batch, :, 4]) - np.sort(expected[batch, :, 4])))))
                return actual, expected, False
            reference_rows = np.empty_like(actual[batch])
            for category in np.unique(actual[batch, :, 5]).astype(int):
                if category < 0 or category >= nc:
                    return actual, expected, False
                rows = np.flatnonzero(actual[batch, :, 5] == category)
                candidates = np.concatenate((dense[batch, :, :4], dense[batch, :, 4 + category:5 + category],
                    np.full((dense.shape[1], 1), category, dtype=np.float32), dense[batch, :, 4 + nc:]), axis=1)
                matches = np.all(np.isclose(actual[batch, rows, None, :], candidates[None, :, :], atol=3e-3, rtol=3e-4), axis=-1)
                matches &= np.isclose(actual[batch, rows, None, 4], candidates[None, :, 4], atol=2e-7, rtol=3e-4)
                owner = np.full(candidates.shape[0], -1, dtype=np.int32)

                def assign_dense(row, seen):
                    for candidate in np.flatnonzero(matches[row]):
                        if seen[candidate]:
                            continue
                        seen[candidate] = True
                        if owner[candidate] < 0 or assign_dense(int(owner[candidate]), seen):
                            owner[candidate] = row
                            return True
                    return False

                for row in np.argsort(matches.sum(axis=1), kind="stable"):
                    if not assign_dense(int(row), np.zeros(candidates.shape[0], dtype=bool)):
                        print("TopK payload mismatch", category, int(rows[row]), actual[batch, rows[row]])
                        return actual, expected, False
                for candidate in np.flatnonzero(owner >= 0):
                    reference_rows[rows[owner[candidate]]] = candidates[candidate]
            ordered[batch] = reference_rows
            continue
        # 即使没有完整候选，也必须对包括 mask/关键点在内的整行作一对一匹配。
        matches = np.all(np.isclose(expected[batch, :, None, :], actual[batch, None, :, :], atol=3e-3, rtol=3e-4), axis=-1)
        matches &= expected[batch, :, None, 5] == actual[batch, None, :, 5]
        owner = np.full(actual.shape[1], -1, dtype=np.int32)

        def assign(row, seen):
            for candidate in np.flatnonzero(matches[row]):
                if seen[candidate]:
                    continue
                seen[candidate] = True
                if owner[candidate] < 0 or assign(int(owner[candidate]), seen):
                    owner[candidate] = row
                    return True
            return False

        for row in np.argsort(matches.sum(axis=1), kind="stable"):
            if not assign(int(row), np.zeros(actual.shape[1], dtype=bool)):
                return actual, expected, False
        for candidate, row in enumerate(owner):
            ordered[batch, row] = actual[batch, candidate]
    return (actual, ordered, True) if dense is not None else (ordered, expected, True)
