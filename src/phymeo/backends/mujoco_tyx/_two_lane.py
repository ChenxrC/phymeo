"""Compensated FP32 scalar operations for retained device state precision."""

import tyx as tx

from ._mesh_query_kernels import sum_error


@tx.func
def add_tail(a: tx.f32, al: tx.f32, b: tx.f32, bl: tx.f32) -> tx.f32:
    return sum_error(a, b, tx.add_rn(a, b)) + al + bl


@tx.func
def add_hi(a: tx.f32, al: tx.f32, b: tx.f32, bl: tx.f32) -> tx.f32:
    return tx.add_rn(tx.add_rn(a, b), add_tail(a, al, b, bl))


@tx.func
def add_lo(a: tx.f32, al: tx.f32, b: tx.f32, bl: tx.f32) -> tx.f32:
    return sum_error(tx.add_rn(a, b), add_tail(a, al, b, bl), add_hi(a, al, b, bl))


@tx.func
def mul_tail(a: tx.f32, al: tx.f32, b: tx.f32, bl: tx.f32) -> tx.f32:
    return tx.fma(a, b, -tx.mul_rn(a, b)) + a * bl + al * b + al * bl


@tx.func
def mul_hi(a: tx.f32, al: tx.f32, b: tx.f32, bl: tx.f32) -> tx.f32:
    return tx.add_rn(tx.mul_rn(a, b), mul_tail(a, al, b, bl))


@tx.func
def mul_lo(a: tx.f32, al: tx.f32, b: tx.f32, bl: tx.f32) -> tx.f32:
    return sum_error(tx.mul_rn(a, b), mul_tail(a, al, b, bl), mul_hi(a, al, b, bl))


@tx.func
def div_tail(a: tx.f32, al: tx.f32, b: tx.f32, bl: tx.f32) -> tx.f32:
    return (tx.fma(-(a / b), b, a) + al - (a / b) * bl) / b


@tx.func
def div_hi(a: tx.f32, al: tx.f32, b: tx.f32, bl: tx.f32) -> tx.f32:
    return tx.add_rn(a / b, div_tail(a, al, b, bl))


@tx.func
def div_lo(a: tx.f32, al: tx.f32, b: tx.f32, bl: tx.f32) -> tx.f32:
    return sum_error(a / b, div_tail(a, al, b, bl), div_hi(a, al, b, bl))


@tx.func
def sqrt_tail(a: tx.f32, al: tx.f32) -> tx.f32:
    return (tx.fma(-tx.sqrt(a), tx.sqrt(a), a) + al) / (2.0 * tx.sqrt(a))


@tx.func
def sqrt_hi(a: tx.f32, al: tx.f32) -> tx.f32:
    return tx.add_rn(tx.sqrt(a), sqrt_tail(a, al))


@tx.func
def sqrt_lo(a: tx.f32, al: tx.f32) -> tx.f32:
    return sum_error(tx.sqrt(a), sqrt_tail(a, al), sqrt_hi(a, al))
