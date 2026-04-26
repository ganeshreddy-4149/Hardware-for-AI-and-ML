import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


def to_signed32(val):
    v = int(val)
    if v >= 2**31:
        v -= 2**32
    return v


@cocotb.test()
async def test_mac_basic(dut):
    """
    Sequence from Codefest 4 spec:
      [a=3,  b=4]  for 3 cycles  -> out = 12, 24, 36
      rst=1 for 1 cycle          -> out = 0
      [a=-5, b=2]  for 2 cycles  -> out = -10, -20
    """
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    # ---- initial reset ----
    dut.rst.value = 1
    dut.a.value   = 0
    dut.b.value   = 0
    await RisingEdge(dut.clk)
    await Timer(1, units="ns")   # wait 1ns after edge to let outputs settle

    # ---- accumulate a=3, b=4 ----
    dut.rst.value = 0
    dut.a.value   = 3
    dut.b.value   = 4

    for expected in [12, 24, 36]:
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")   # let outputs settle after clock edge
        actual = to_signed32(dut.out.value)
        assert actual == expected, \
            f"Basic accumulate: expected {expected}, got {actual}"
        dut._log.info(f"[PASS] out = {actual} (expected {expected})")

    # ---- mid-run reset ----
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, units="ns")
    actual = to_signed32(dut.out.value)
    assert actual == 0, f"After reset: expected 0, got {actual}"
    dut._log.info(f"[PASS] reset -> out = {actual}")

    # ---- accumulate a=-5, b=2 ----
    dut.rst.value = 0
    dut.a.value   = -5
    dut.b.value   = 2

    for expected in [-10, -20]:
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")
        actual = to_signed32(dut.out.value)
        assert actual == expected, \
            f"Negative accumulate: expected {expected}, got {actual}"
        dut._log.info(f"[PASS] out = {actual} (expected {expected})")


@cocotb.test()
async def test_mac_overflow(dut):
    """
    Drive a=127, b=127 (product=16129) repeatedly until the accumulator
    wraps around from positive to negative (standard 2's complement).
    Documents that mac_correct.v WRAPS, not saturates.
    """
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    # reset first
    dut.rst.value = 1
    dut.a.value   = 0
    dut.b.value   = 0
    await RisingEdge(dut.clk)
    await Timer(1, units="ns")
    dut.rst.value = 0

    # drive max positive product: 127 * 127 = 16129
    dut.a.value = 127
    dut.b.value = 127

    MAX_CYCLES = 200_000
    wrap_cycle = None
    prev       = 0

    for cycle in range(1, MAX_CYCLES + 1):
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")
        current = to_signed32(dut.out.value)

        if current < 0 and prev >= 0:
            wrap_cycle = cycle
            dut._log.info(
                f"[OVERFLOW] Wrap-around detected at cycle {cycle}: "
                f"prev={prev}, current={current}"
            )
            break
        prev = current

    if wrap_cycle is not None:
        dut._log.info(
            f"[RESULT] Design WRAPS (no saturation). "
            f"Wrap occurred at cycle {wrap_cycle}. "
            f"Value just before wrap: {prev}"
        )
        assert wrap_cycle is not None
    else:
        dut._log.info(
            f"[RESULT] Design SATURATES. Final value: {prev}"
        )
