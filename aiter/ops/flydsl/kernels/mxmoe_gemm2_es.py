# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Expert-stationary MXFP4 MoE GEMM2 writing MXFP4 route rows (persistent, occupancy 2).

A fixed grid of 2 workgroups per CU never re-dispatches. Work units are panels: one
expert's sorted rows times 256 output columns. A workgroup keeps a panel's whole W2
(K = 512, all K-tiles, plus its e8m0 words) in registers while the next panel's W2 is in
flight, and streams the expert's rows through an LDS double buffer 32 at a time,
computing 16-row halves only up to the expert's last valid row. Each half is quantized in
registers (W2 is the MFMA's first operand, so a lane holds 4 consecutive columns of one
row) and stored as unweighted [model_dim/2 e2m1 | model_dim/32 e8m0] rows indexed by
(token, slot), which moe_reduce's "fp4" path weights and sums.

The pipeline follows AKLR's expert-stationary stage 2. Panels take the experts in the
order of the sort's expert table (``moe_sorting_opus_fwd``'s ``expert_info``: first row,
tokens, expert; descending tokens). Without that table the experts' row ranges are
recovered in-kernel from sorted_expert_ids, in expert order.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, llvm
from flydsl._mlir.dialects import rocdl as rocdl_ods
from flydsl._mlir.extras import types as _T
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.runtime.device import is_rdna_arch

from .mxfp4_gemm_common import fp4out_row_bytes
from .tensor_shim import _run_compiled

_TW = (
    4  # 16-column MFMA tiles per wave: a workgroup panel is 4 waves x 64 = 256 columns
)
_PANEL_N = 64 * _TW


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _rsrc(addr, nbytes=None):
    """Buffer resource over a raw device address; with nbytes, accesses past it are dropped."""
    flags = (7 << 12) | (4 << 15)
    if is_rdna_arch(None):
        flags |= (1 << 24) | (2 << 28)
    base = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr"), _raw(fx.Int64(addr))).result
    i32, i16, i64 = _T.i32(), _T.i16(), _T.i64()
    fl = arith.ConstantOp(i32, ir.IntegerAttr.get(i32, flags)).result
    stride = arith.ConstantOp(i16, ir.IntegerAttr.get(i16, 0)).result
    if nbytes is None:
        nrec = arith.ConstantOp(i64, ir.IntegerAttr.get(i64, 0xFFFFFFFF)).result
    else:
        nrec = arith.extui(i64, _raw(fx.Int32(nbytes)))
    return rocdl_ods.MakeBufferRsrcOp(
        ir.Type.parse("!llvm.ptr<8>"), base, stride, nrec, fl
    ).result


def _ld4(rsrc, off, aux=0):
    return fx.Vector(
        rocdl.raw_ptr_buffer_load(T.i32x4, rsrc, fx.Int32(off), fx.Int32(0), aux)
    )


def _ld1(rsrc, off):
    return fx.Int32(rocdl.raw_ptr_buffer_load(T.i32, rsrc, fx.Int32(off), fx.Int32(0)))


def _st(rsrc, off, value):
    rocdl_ods.raw_ptr_buffer_store(
        _raw(value), rsrc, _raw(fx.Int32(off)), _raw(fx.Int32(0))
    )


def _uniform(x):
    return fx.Int32(rocdl.readfirstlane(T.i32, _raw(fx.Int32(x))))


def _lds(base, off):
    ptr_ty = fx.PointerType.get(T.i8, fx.AddressSpace.Shared)
    return fx.to_llvm_ptr(fx.inttoptr(ptr_ty, fx.Int64(base + off)))


def _lds_ldi(base, off):
    return fx.Int32(llvm.LoadOp(T.i32, _lds(base, off), alignment=4).res)


def _lds_sti(base, off, v):
    llvm.StoreOp(_raw(fx.Int32(v)), _lds(base, off), alignment=4)


def _umax(a, b):
    return (fx.Uint32(a) > fx.Uint32(b)).select(a, b)


def _swap_max(m, op):
    """Unsigned max of m with its partner row (permlane16_swap: lane ^ 16, permlane32_swap: lane ^ 32)."""
    r = op(ir.Type.parse("!llvm.struct<(i32, i32)>"), _raw(m), _raw(m), False, False)
    return _umax(
        fx.Int32(llvm.ExtractValueOp(T.i32, r, [0]).res),
        fx.Int32(llvm.ExtractValueOp(T.i32, r, [1]).res),
    )


def _pack_fp4(vals, scale):
    """Eight floats to one dword of e2m1 (element j in nibble j)."""
    q = fx.Int32(0)
    for j in range_constexpr(len(vals) // 2):
        q = fx.Int32(
            rocdl.cvt_scalef32_pk_fp4_f32(
                T.i32, _raw(q), _raw(vals[2 * j]), _raw(vals[2 * j + 1]), _raw(scale), j
            )
        )
    return q


def _default_peel(rows_per_expert):
    """(chunks unrolled ahead of the chunk loop, slices of the next panel's 20 loads they
    issue, whether chunk 0 issues its slice before staging chunk 1). Panels with more rows
    spread the next panel's W2 over more chunks."""
    if rows_per_expert <= 48:
        return 2, (0, 20, 20), False
    if rows_per_expert <= 96:
        return 3, (0, 8, 14, 20), False
    return 5, (0, 8, 11, 14, 17, 20), True


@functools.cache
def compile_gemm2_es(
    *, model_dim, inter_dim, topk, sbm, num_experts, n_wg, peel, ext_einfo=False
):
    """ext_einfo: the expert table comes from the sort (``moe_sorting_opus_fwd``'s
    ``expert_info``: experts by descending tokens); else it is built from sorted_expert_ids.
    """
    PEEL, SL, EARLY = peel
    K = inter_dim
    N = model_dim
    TW = _TW
    NB = N // _PANEL_N  # panels per expert
    G = n_wg
    assert (
        K == 512
    ), "the whole K of a panel is register-resident: inter_dim must be 512"
    assert N % _PANEL_N == 0 and sbm % 32 == 0 and G % 8 == 0
    assert SL[-1] == 5 * TW and len(SL) == PEEL + 1
    PITCH = fp4out_row_bytes(N)
    RB = 2 * 16 * TW  # packed bytes of one row in a 256-column panel
    SB = TW // 2  # e8m0 bytes per wave per row
    OBUF = 32 * RB + 32 * 4 * SB  # one chunk of output: 32 rows of values, then scales
    AROW = K // 2  # bytes of an A row
    ABUF = (
        32 * AROW + 32 * (K // 32) + 32 * 4
    )  # A chunk: rows, their e8m0 words, route ids
    assert 32 * AROW == 2 * 256 * 16
    MAXS = num_experts + 1  # expert runs (+ end sentinel)
    OFF_TF = 2 * ABUF + 2 * OBUF  # first sort block of each expert run
    OFF_TE = OFF_TF + 4 * MAXS  # expert id of each run
    OFF_WT = OFF_TE + 4 * MAXS  # per-wave counts of the run scan
    LDS_BYTES = OFF_WT + 16
    # A run's rows are counted from its last sort block's route ids, one per lane.
    CSTRIDE = max(1, sbm // 64)
    CLANES = sbm // CSTRIDE
    assert 16 % CSTRIDE == 0 and CLANES * CSTRIDE == sbm

    @fx.struct
    class Smem:
        raw: fx.Array[fx.Uint8, LDS_BYTES, 16]

    name = (
        f"gemm2_a4w4_es_n{N}_k{K}_tk{topk}_sbm{sbm}_ne{num_experts}_wg{G}_p{PEEL}"
        + ("_ei" if ext_einfo else "")
    )

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def gemm2_es(
        A: fx.Int64,
        AS: fx.Int64,
        W2: fx.Int64,
        WS2: fx.Int64,
        SID: fx.Int64,
        EIDS: fx.Int64,
        CUMSUM: fx.Int64,
        EINFO: fx.Int64,
        OUT: fx.Int64,
        i32_M: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        g = fx.Int32(gpu.block_id("x"))
        lane = tid % 64
        wave = _uniform(tid // 64)
        l16 = lane % 16
        lg = lane // 16
        w = _rsrc(W2)
        ws = _rsrc(WS2)
        a = _rsrc(A)
        asr = _rsrc(AS)
        sid = _rsrc(SID)
        eids = _rsrc(EIDS)
        NREC = i32_M * fx.Int32(topk * PITCH)
        o = _rsrc(OUT, NREC)
        lb = fx.Int32(fx.ptrtoint(fx.SharedAllocator().allocate(Smem).peek().raw.ptr))
        ob0 = lb + 2 * ABUF
        i8 = ir.IntegerType.get_signless(8)
        orow = tid // 8
        oseg = tid % 8
        # A chunk units: rows j * 16 + tid / 16, segment tid % 16 stored at segment
        # (tid % 16) ^ (tid / 16), so the MFMA operand reads spread over all banks.
        wseg = ((tid % 16) ^ (tid // 16)) * 16

        if const_expr(ext_einfo):
            # Rows [first row, tokens, expert, 0], experts by descending tokens, then
            # [experts with tokens, 0, 0, 0].
            ei = _rsrc(EINFO)
            NWG = _uniform(_ld1(ei, num_experts * 16)) * NB  # panels
        else:
            # ---- Expert runs: sorted_expert_ids repeats an expert over its sort blocks; the
            # s-th run starts at block TF[s] (TF[n_runs] = n_blocks) with expert TE[s].
            c32 = arith.ConstantOp(T.i64, ir.IntegerAttr.get(T.i64, 32)).result
            n_blocks = _uniform(_ld1(_rsrc(CUMSUM), 0)) // fx.Int32(sbm)
            n_rounds = (n_blocks + 255) // 256
            for rnd, run in range(
                fx.Int32(0), n_rounds, fx.Int32(1), init=[_raw(fx.Int32(0))]
            ):
                b = fx.Int32(rnd) * 256 + tid
                inb = b < n_blocks
                bc = inb.select(b, fx.Int32(0))
                eb = _ld1(eids, bc * 4)
                ep = _ld1(eids, (bc > 0).select(bc - 1, fx.Int32(0)) * 4)
                start = inb & ((b == fx.Int32(0)) | (eb != ep))
                bal = rocdl.ballot(T.i64, _raw(start))
                # Run starts in the lanes below this one.
                pre = fx.Int32(
                    rocdl.mbcnt_hi(
                        T.i32,
                        arith.trunci(T.i32, arith.shrui(bal, c32)),
                        rocdl.mbcnt_lo(
                            T.i32, arith.trunci(T.i32, bal), _raw(fx.Int32(0))
                        ),
                    )
                )
                nw = fx.Int32(arith.trunci(T.i32, llvm.intr_ctpop(bal)))
                if lane == fx.Int32(0):
                    _lds_sti(lb, OFF_WT + wave * 4, nw)
                gpu.barrier()
                wc = [_lds_ldi(lb, OFF_WT + 4 * i) for i in range_constexpr(4)]
                idx = (
                    fx.Int32(run[0])
                    + pre
                    + (wave > 0).select(wc[0], fx.Int32(0))
                    + (wave > 1).select(wc[1], fx.Int32(0))
                    + (wave > 2).select(wc[2], fx.Int32(0))
                )
                if start:
                    _lds_sti(lb, OFF_TF + idx * 4, b)
                    _lds_sti(lb, OFF_TE + idx * 4, eb)
                gpu.barrier()
                n_runs_r = yield [
                    _raw(fx.Int32(run[0]) + wc[0] + wc[1] + wc[2] + wc[3])
                ]
            n_runs = _uniform(fx.Int32(n_runs_r))  # one carried value comes back bare
            if tid == fx.Int32(0):
                _lds_sti(lb, OFF_TF + n_runs * 4, n_blocks)
            gpu.barrier()
            NWG = n_runs * NB  # panels

        def clampp(p):
            p = (p < NWG).select(p, NWG - 1)
            return (p > 0).select(p, fx.Int32(0))

        if const_expr(ext_einfo):

            def einfo_regs(p):
                """Panel p's expert: [first row, tokens, expert], loaded per lane."""
                r = clampp(p) // NB
                return [_raw(_ld1(ei, r * 16 + 4 * i)) for i in range_constexpr(3)]

            def einfo_count(ein):
                return ein

        else:

            def einfo_regs(p):
                """Panel p's expert run: first row, rows of its full sort blocks, expert id, and
                the route ids at every CSTRIDE-th row of its last block (one per lane).
                """
                s = clampp(p) // NB
                fb = _uniform(_lds_ldi(lb, OFF_TF + s * 4))
                fe = _uniform(_lds_ldi(lb, OFF_TF + (s + 1) * 4))
                e = _uniform(_lds_ldi(lb, OFF_TE + s * 4))
                last = (fe - 1) * sbm
                v = _ld1(sid, (last + (lane % CLANES) * CSTRIDE) * 4)
                return [_raw(fb * sbm), _raw(last - fb * sbm), _raw(e), _raw(v)]

            def einfo_count(ein):
                """[first row, rows, expert]. Padding follows an expert's rows, so the sampled ids
                give the rows rounded up to CSTRIDE, which divides 16: the 16-row halves are exact.
                """
                ok = (fx.Int32(ein[3]) & fx.Int32(0xFFFFFF)) < i32_M
                if const_expr(CLANES < 64):
                    ok = ok & (lane < CLANES)
                nv = fx.Int32(
                    arith.trunci(T.i32, llvm.intr_ctpop(rocdl.ballot(T.i64, _raw(ok))))
                )
                return [ein[0], _raw(fx.Int32(ein[1]) + nv * CSTRIDE), ein[2]]

        def w2_regs(p, e, lo=0, hi=5 * TW):
            """Loads lo .. hi - 1 of the panel's W2: TW e8m0 dwords, then 4 TW 16-byte units."""
            pc = clampp(p)
            n0 = (pc % NB) * _PANEL_N + wave * (16 * TW)
            res = []
            for i in range_constexpr(lo, hi):
                if const_expr(i < TW):
                    h = i // 2
                    q = i % 2
                    res.append(
                        _raw(
                            _ld1(
                                ws,
                                e * (N * K // 32)
                                + ((n0 >> 5) + h) * K
                                + lg * 64
                                + l16 * 4
                                + q * 256,
                            )
                        )
                    )
                else:
                    t = (i - TW) // 4
                    kk = (i - TW) % 4
                    res.append(
                        _raw(
                            _ld4(
                                w,
                                e * (N * K // 2)
                                + ((n0 >> 4) + t) * (K * 8)
                                + kk * 1024
                                + lane * 16,
                                2,
                            )
                        )
                    )
            return res

        def load_chunk(base, nch, ci):
            cc = (ci < nch).select(ci, nch - 1)
            cc = (cc < 0).select(fx.Int32(0), cc)
            r = base + cc * 32
            d = [_ld4(a, r * AROW + (j * 256 + tid) * 16) for j in range_constexpr(2)]
            sc = _ld1(asr, (r // 32) * K + (tid & 127) * 4)
            f = _ld1(sid, (r + (tid & 31)) * 4)
            return [_raw(d[0]), _raw(d[1]), _raw(sc), _raw(f)]

        def to_lds(regs, buf):
            for j in range_constexpr(2):
                llvm.StoreOp(
                    regs[j],
                    _lds(buf + (j * 16 + tid // 16) * AROW + wseg, 0),
                    alignment=16,
                )
            if tid < 128:
                llvm.StoreOp(regs[2], _lds(buf + 32 * AROW + tid * 4, 0), alignment=4)
            if tid < 32:
                llvm.StoreOp(
                    regs[3], _lds(buf + 32 * AROW + 512 + tid * 4, 0), alignment=4
                )

        # Prologue: the first panel's W2, rows and first two chunks; the second panel's run.
        gl = (g % 8) * (G // 8) + g // 8
        ein0 = einfo_regs(gl)
        ein1 = einfo_regs(gl + G)
        w0 = w2_regs(gl, _uniform(fx.Int32(ein0[2])))
        ein0 = einfo_count(ein0)
        base0 = _uniform(fx.Int32(ein0[0]))
        cnt0 = _uniform(fx.Int32(ein0[1]))
        nch0 = (cnt0 + 31) // 32
        to_lds(load_chunk(base0, nch0, fx.Int32(0)), lb)
        a1 = load_chunk(base0, nch0, fx.Int32(1))
        ein1 = einfo_count(ein1)
        gpu.barrier()
        npan = (gl < NWG).select((NWG - 1 - gl) // G + 1, fx.Int32(0))
        init = w0 + ein1 + a1 + [_raw(base0), _raw(cnt0)]
        for j, ost in range(fx.Int32(0), npan, fx.Int32(1), init=init):
            p = gl + fx.Int32(j) * G
            nb = p % NB
            bs_raw = [fx.Int32(ost[i]) for i in range_constexpr(TW)]
            bv = [fx.Vector(ost[TW + i]) for i in range_constexpr(4 * TW)]
            ein_n = [ost[5 * TW + i] for i in range_constexpr(3)]
            a_1 = [ost[5 * TW + 3 + i] for i in range_constexpr(4)]
            base = fx.Int32(ost[5 * TW + 7])
            cnt = fx.Int32(ost[5 * TW + 8])
            e_n = _uniform(fx.Int32(ein_n[2]))
            nch = (cnt + 31) // 32

            def chunk(
                ci,
                ist,
                ahead,
                early=False,
                nch=nch,
                cnt=cnt,
                nb=nb,
                bv=bv,
                bs_raw=bs_raw,
            ):
                """Chunk ci: chunk ci + 1 (ist) to LDS, ahead's loads, chunk ci's MFMAs up to
                the expert's last valid row, quantization and stores."""
                if const_expr(early):
                    nxt = ahead()
                if ci + 1 < nch:
                    to_lds(list(ist), lb + ((ci + 1) & 1) * ABUF)
                if const_expr(not early):
                    nxt = ahead()
                abuf = lb + (ci & 1) * ABUF
                as_raw = [
                    fx.Int32(
                        llvm.LoadOp(
                            T.i32,
                            _lds(abuf + 32 * AROW + q * 256 + lg * 64 + l16 * 4, 0),
                            alignment=4,
                        ).res
                    )
                    for q in range_constexpr(2)
                ]
                fo = _lds_ldi(abuf + 32 * AROW + 512 + orow * 4, 0)
                live = ci < nch
                bb = ob0 + (ci & 1) * OBUF

                def mma(rt, accs):
                    rocdl.s_setprio(1)
                    av = [
                        fx.Vector(
                            llvm.LoadOp(
                                T.i32x4,
                                _lds(
                                    abuf
                                    + (rt * 16 + l16) * AROW
                                    + (((kk * 4 + lg) ^ l16) * 16),
                                    0,
                                ),
                                alignment=16,
                            ).res
                        )
                        for kk in range_constexpr(4)
                    ]
                    for t in range_constexpr(TW):
                        for kk in range_constexpr(4):
                            accs[t][rt] = fx.Vector(
                                rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                    T.f32x4,
                                    [
                                        bv[t * 4 + kk],
                                        av[kk],
                                        accs[t][rt],
                                        4,
                                        4,
                                        (kk & 1) * 2,
                                        bs_raw[(t >> 1) * 2 + (kk >> 1)]
                                        >> (8 * (t & 1)),
                                        (kk & 1) * 2,
                                        as_raw[kk >> 1] >> (8 * rt),
                                    ],
                                )
                            )

                def quant(rt, accs):
                    rocdl.s_setprio(0)
                    row = rt * 16 + l16
                    for b in range_constexpr(TW // 2):
                        vals = [
                            accs[2 * b + h][rt][i]
                            for h in range_constexpr(2)
                            for i in range_constexpr(4)
                        ]
                        fm = None
                        for jj in range_constexpr(8):
                            x = fx.Float32(llvm.intr_fabs(_raw(vals[jj])))
                            fm = (
                                x
                                if fm is None
                                else fx.Float32(arith.maximumf(_raw(fm), _raw(x)))
                            )
                        m = fm.bitcast(fx.Int32)
                        m = _swap_max(m, rocdl_ods.permlane16_swap)
                        m = _swap_max(m, rocdl_ods.permlane32_swap)
                        ex = ((m >> 23) & 255) - 2
                        ex = (ex < 1).select(fx.Int32(1), ex)
                        ex = (m == 0).select(fx.Int32(0), ex)
                        scale = (m == 0).select(
                            fx.Float32(1.0), (ex << 23).bitcast(fx.Float32)
                        )
                        qv = _pack_fp4(vals, scale)
                        ob = bb + row * RB + wave * (16 * TW // 2) + 16 * b + 2 * lg
                        llvm.StoreOp(
                            arith.trunci(T.i16, _raw(qv)), _lds(ob, 0), alignment=2
                        )
                        llvm.StoreOp(
                            arith.trunci(T.i16, _raw(qv >> 16)),
                            _lds(ob + 8, 0),
                            alignment=2,
                        )
                        if lg == 0:
                            llvm.StoreOp(
                                arith.trunci(i8, _raw(ex)),
                                _lds(bb + 32 * RB + row * (4 * SB) + wave * SB + b, 0),
                                alignment=1,
                            )

                def half(rt):
                    accs = [
                        [
                            fx.Vector.filled(4, 0.0, fx.Float32)
                            for _ in range_constexpr(2)
                        ]
                        for _ in range_constexpr(TW)
                    ]
                    mma(rt, accs)
                    quant(rt, accs)

                # One 16-row half at a time keeps half the accumulators live; halves without rows are skipped.
                if ci * 32 + 16 < cnt:
                    half(0)
                    half(1)
                if (ci * 32 + 16 >= cnt) & live:
                    half(0)
                gpu.barrier()
                # Padding rows carry token M, slot topk: their destination lies past NREC.
                dest = ((fo & 0xFFFFFF) * topk + (fo >> 24)) * PITCH
                dest = live.select(dest, NREC)
                for hh in range_constexpr(RB // 128):
                    v = fx.Vector(
                        llvm.LoadOp(
                            T.i32x4,
                            _lds(bb + orow * RB + (oseg + 8 * hh) * 16, 0),
                            alignment=16,
                        ).res
                    )
                    _st(o, dest + nb * RB + (oseg + 8 * hh) * 16, v)
                # Unconditional, so wait counts stay exact: lanes past a row's scale dwords aim past NREC.
                sseg = oseg % SB
                sv = fx.Int32(
                    llvm.LoadOp(
                        T.i32,
                        _lds(bb + 32 * RB + orow * (4 * SB) + sseg * 4, 0),
                        alignment=4,
                    ).res
                )
                _st(
                    o,
                    (oseg < SB).select(dest + N // 2 + nb * (4 * SB) + sseg * 4, NREC),
                    sv,
                )
                return nxt

            # Chunk 2 ahead of the next panel's W2: waiting for it does not wait for that W2.
            nxt2 = load_chunk(base, nch, fx.Int32(2))
            base_n = _uniform(fx.Int32(ein_n[0]))
            cnt_n = _uniform(fx.Int32(ein_n[1]))
            nch_n = (cnt_n + 31) // 32
            wn = []
            cur = a_1
            for k in range_constexpr(PEEL):

                def ahead(
                    k=k, base=base, nch=nch, p=p, e_n=e_n, base_n=base_n, nch_n=nch_n
                ):
                    r = []
                    if const_expr(k >= 1):
                        r.append(load_chunk(base, nch, fx.Int32(k + 2)))
                    r.append(w2_regs(p + G, e_n, SL[k], SL[k + 1]))
                    if const_expr(k == PEEL - 1):
                        r.append(einfo_regs(p + 2 * G))
                        r.append(load_chunk(base_n, nch_n, fx.Int32(0)))
                    return r

                r = chunk(fx.Int32(k), cur, ahead, early=EARLY and k == 0)
                if const_expr(k >= 1):
                    cur = r[0]
                    r = r[1:]
                else:
                    cur = nxt2
                wn = wn + r[0]
                if const_expr(k == PEEL - 1):
                    ein_nn = r[1]
                    an0 = r[2]
            for c, ist in range(fx.Int32(PEEL), nch, fx.Int32(1), init=cur):
                ci = fx.Int32(c)
                yield chunk(
                    ci,
                    ist,
                    lambda ci=ci, base=base, nch=nch: load_chunk(base, nch, ci + 2),
                )
            # Every read of both A buffers is behind the loop's last barrier.
            to_lds(an0, lb)
            # an0 was loaded after ein_nn's route ids: counting them does not wait.
            ein_nn = einfo_count(ein_nn)
            an1 = load_chunk(base_n, nch_n, fx.Int32(1))
            gpu.barrier()
            yield wn + ein_nn + an1 + [_raw(base_n), _raw(cnt_n)]

    @flyc.jit
    def launch(
        A: fx.Int64,
        AS: fx.Int64,
        W2: fx.Int64,
        WS2: fx.Int64,
        SID: fx.Int64,
        EIDS: fx.Int64,
        CUMSUM: fx.Int64,
        EINFO: fx.Int64,
        OUT: fx.Int64,
        i32_M: fx.Int32,
        stream: fx.Stream,
    ):
        gemm2_es(A, AS, W2, WS2, SID, EIDS, CUMSUM, EINFO, OUT, i32_M).launch(
            grid=(G, 1, 1), block=(256, 1, 1), stream=stream
        )

    launch.compile_hints = {
        "waves_per_eu": 2,
        "llvm_options": {
            "amdgpu-kernarg-preload": True,
            "amdgpu-kernarg-preload-count": 16,
        },
    }
    return launch


def mxfp4_moe_gemm2_es(
    inter_sorted_quant,
    inter_sorted_shuffled_scale,
    w2_u8,
    w2_scale_u8,
    sorted_expert_ids,
    cumsum_tensor,
    sorted_token_ids,
    out,
    *,
    M_logical,
    NE,
    D_HIDDEN,
    D_INTER,
    topk,
    SBM,
    cu_num,
    expert_info=None,
    stream=None,
):
    """Expert-stationary GEMM2 into fp4 route rows ``out`` (uint8 [M*topk, fp4out_row_bytes]).
    2 workgroups per CU, panels of 256 output columns, inter_dim 512."""
    import torch

    peel = _default_peel(M_logical * topk / max(1, NE))
    launch = compile_gemm2_es(
        model_dim=D_HIDDEN,
        inter_dim=D_INTER,
        topk=topk,
        sbm=SBM,
        num_experts=NE,
        n_wg=2 * cu_num,
        peel=peel,
        ext_einfo=expert_info is not None,
    )
    # Through the cached CompiledFunction: calling the jit launcher re-traces each call.
    _run_compiled(
        launch,
        inter_sorted_quant.data_ptr(),
        inter_sorted_shuffled_scale.data_ptr(),
        w2_u8.data_ptr(),
        w2_scale_u8.data_ptr(),
        sorted_token_ids.data_ptr(),
        sorted_expert_ids.data_ptr(),
        cumsum_tensor.data_ptr(),
        expert_info.data_ptr() if expert_info is not None else 0,
        out.data_ptr(),
        M_logical,
        torch.cuda.current_stream() if stream is None else stream,
    )
    return out
