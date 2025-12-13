import numpy as np

def canonicalize_array(arr: np.ndarray, fixed_rows_indices: list[int], enable_sort_col: bool) -> np.ndarray:
    """
    Returns a canonicalized version of the array by sorting its elements.

    fixed_rows_indices: list of row indices that should remain fixed during sorting.

    Process:
    1. Sort columns of each row. (if enable_sort_col is True)
    2. Fix the rows specified in fixed_rows_indices.
    3. Sort rows.

    TODO: support fixed columns.
    """
    # 1. Sort columns of each row
    if enable_sort_col:
        arr_sorted_cols = np.sort(arr, axis=1)
    else:
        arr_sorted_cols = arr.copy()
    
    # 2. Fix the rows specified in fixed_rows_indices.
    all_rows = np.arange(arr_sorted_cols.shape[0])
    fixed_rows_mask = np.isin(all_rows, fixed_rows_indices)
    fixed_rows = arr_sorted_cols[fixed_rows_mask]
    non_fixed_rows = arr_sorted_cols[~fixed_rows_mask]

    # 3. Sort rows.
    non_fixed_sorter = np.lexsort(non_fixed_rows.T)
    non_fixed_rows_sorted = non_fixed_rows[non_fixed_sorter]
    
    # 4. Final assembly
    fixed_rows_sorted_by_index = arr_sorted_cols[fixed_rows_indices]
    canonical_arr = np.vstack([fixed_rows_sorted_by_index, non_fixed_rows_sorted])
    
    return canonical_arr

def to_hashable_key(canonical_arr):
    return canonical_arr.tobytes()

if __name__ == "__main__":
    arr = np.array([[False, True, True],
                    [True, False, True], # fix
                    [False, True, True], # fix
                    [False, False, True], # fix
                    [True, False, False]], dtype=bool)
    arr2 = np.array([[False, False, True],
                     [True, True, False], # fix
                     [True, True, False], # fix
                     [False, False, True], # fix
                     [True, False, True]], dtype=bool)
    fixed_rows = [1, 2, 3]
    can_arr = canonicalize_array(arr, fixed_rows, enable_sort_col=True)
    can_arr2 = canonicalize_array(arr2, fixed_rows, enable_sort_col=True)
    print(can_arr)
    print(can_arr2)
    print(to_hashable_key(arr))
    print(to_hashable_key(arr2))
    print(to_hashable_key(can_arr))
    print(to_hashable_key(can_arr2))