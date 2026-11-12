"""Memory-safety guards before the formal kernel's ABI is called."""
import ctypes
from types import SimpleNamespace
import numpy as np
import pytest
from scripts import simulation


def arguments():
    return [np.eye(64), np.zeros(64), 100.0, 80, 10, 2, 123, 0, 4, 8, 1.0]


@pytest.mark.parametrize("index,value", [(0,np.eye(63)), (1,np.zeros(65)), (2,float("nan")), (3,-1), (4,2), (5,-1), (6,-1), (7,5), (8,6), (9,0), (10,float("inf"))])
def test_invalid_native_arguments_never_enter_c(index, value):
    args=arguments(); args[index]=value
    def forbidden(*_):
        raise AssertionError("invalid input entered C")
    with pytest.raises(ValueError):
        simulation.run_chain(SimpleNamespace(r61_formal_run=forbidden), *args)


def test_noncontiguous_float32_buffers_are_converted_before_c():
    args=arguments()
    args[0]=np.arange(4096,dtype=np.float32).reshape(64,64).T
    args[1]=np.arange(128,dtype=np.float32)[::2]
    def inspect(H,h,*tail):
        np.testing.assert_array_equal(np.ctypeslib.as_array(H,shape=(4096,)).reshape(64,64),args[0])
        np.testing.assert_array_equal(np.ctypeslib.as_array(h,shape=(64,)),args[1])
        return 0
    draws,stats=simulation.run_chain(SimpleNamespace(r61_formal_run=inspect),*args)
    assert draws.shape==(8,simulation.DRAW_WIDTH) and draws.dtype==np.float64
    assert stats.shape==(8,) and stats.dtype==np.int64
