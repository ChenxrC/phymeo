"""Gather prepared physical fields without running mechanics or FK."""

import tyx as tx


def gather_scalar(
    count: tx.i32,
    width: tx.i32,
    rows: tx.ndarray(tx.i32, 2),
    columns: tx.ndarray(tx.i32, 2),
    source: tx.ndarray(tx.f32, 2),
    ticks: tx.ndarray(tx.i32, 1),
    output: tx.ndarray(tx.f32, 2),
    output_ticks: tx.ndarray(tx.i32, 1),
):
    for i in tx.range(count):
        for j in range(width):
            output[i, j] = source[rows[i, 0], columns[j, 1]]
        output_ticks[i] = ticks[rows[i, 1]]


def gather_tree(
    count: tx.i32,
    width: tx.i32,
    rows: tx.ndarray(tx.i32, 2),
    columns: tx.ndarray(tx.i32, 2),
    source: tx.ndarray(tx.f32, 3),
    ticks: tx.ndarray(tx.i32, 1),
    output: tx.ndarray(tx.f32, 2),
    output_ticks: tx.ndarray(tx.i32, 1),
):
    for i in tx.range(count):
        for j in range(width):
            output[i, j] = source[rows[i, 0], columns[j, 0], columns[j, 1]]
        output_ticks[i] = ticks[rows[i, 1]]


def make_field_read_kernels(tree):
    function = gather_tree if tree else gather_scalar
    return tx.kernel(function), tx.kernel(function)
