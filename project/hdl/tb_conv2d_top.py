"""
tb_conv2d_top.py — cocotb testbench stub for conv2d_top
ECE 510 Spring 2026 — COPT Part B

Purpose:
  Instantiates conv2d_top, drives reset, applies one representative
  AXI4-Lite write transaction (start pulse), and checks that the
  'busy' signal goes high. Full bit-exact verification against the
  Python golden model is deferred to Milestone 2.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


async def axil_write(dut, addr, data):
    """Minimal AXI4-Lite single-beat write helper."""
    dut.s_axil_awaddr.value  = addr
    dut.s_axil_awvalid.value = 1
    dut.s_axil_wdata.value   = data
    dut.s_axil_wstrb.value   = 0xF
    dut.s_axil_wvalid.value  = 1
    dut.s_axil_bready.value  = 1

    await RisingEdge(dut.clk)
    dut.s_axil_awvalid.value = 0
    dut.s_axil_wvalid.value  = 0
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_reset_and_start(dut):
    """
    1. Apply synchronous reset.
    2. Write start=1 to control register (offset 0x00).
    3. Check busy goes high (COMPUTE state entered).
    4. Wait for done pulse.
    """
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    # ---- de-assert all AXI signals ----
    dut.s_axil_awaddr.value  = 0
    dut.s_axil_awvalid.value = 0
    dut.s_axil_wdata.value   = 0
    dut.s_axil_wstrb.value   = 0
    dut.s_axil_wvalid.value  = 0
    dut.s_axil_bready.value  = 0
    dut.s_axil_araddr.value  = 0
    dut.s_axil_arvalid.value = 0
    dut.s_axil_rready.value  = 0

    # ---- synchronous reset ----
    dut.rst.value = 1
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    # Confirm idle after reset
    assert dut.done.value == 0, "done should be 0 after reset"
    assert dut.busy.value == 0, "busy should be 0 after reset"
    dut._log.info("[PASS] Reset: done=0, busy=0")

    # ---- write start=1 to control register 0x00 ----
    await axil_write(dut, addr=0x00, data=0x1)
    dut._log.info("AXI write: start=1 sent")

    # ---- wait for busy (COMPUTE state) ----
    for _ in range(10):
        await RisingEdge(dut.clk)
        if dut.busy.value == 1:
            dut._log.info("[PASS] busy=1 — COMPUTE state entered")
            break
    else:
        assert False, "busy never went high — FSM did not reach COMPUTE"

    # ---- wait for done (up to 200 cycles for stub FSM) ----
    for _ in range(200):
        await RisingEdge(dut.clk)
        if dut.done.value == 1:
            dut._log.info("[PASS] done=1 — computation finished")
            break
    else:
        dut._log.warning(
            "done never pulsed within 200 cycles — "
            "expected for stub; full compute takes H_out*W_out*C_out cycles"
        )

    dut._log.info("Testbench stub complete — full assertions deferred to M2")
