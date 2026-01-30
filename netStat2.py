import numpy as np
from numba import njit


def _mix_u64_vec(a, b=None, c=None, d=None, tag=0):
    a = a.astype(np.uint64, copy=False)
    x = a ^ np.uint64(tag)
    if b is not None:
        x ^= b.astype(np.uint64, copy=False) * np.uint64(0xBF58476D1CE4E5B9)
    if c is not None:
        x ^= c.astype(np.uint64, copy=False) * np.uint64(0x94D049BB133111EB)
    if d is not None:
        x ^= d.astype(np.uint64, copy=False) * np.uint64(0x9E3779B97F4A7C15)
    x ^= x >> np.uint64(33)
    x *= np.uint64(0xFF51AFD7ED558CCD)
    x ^= x >> np.uint64(33)
    x *= np.uint64(0xC4CEB9FE1A85EC53)
    x ^= x >> np.uint64(33)
    return x


class FlowIndexer:
    def __init__(self):
        self._py_map = {}  # python dict: uint64(int) -> int
        self.size = 0

    def map_keys(self, keys_u64: np.ndarray) -> np.ndarray:
        keys_u64 = np.asarray(keys_u64, dtype=np.uint64)
        out = np.empty(keys_u64.shape[0], dtype=np.int64)
        m = self._py_map
        nxt = self.size
        for i in range(keys_u64.shape[0]):
            k = int(keys_u64[i])
            v = m.get(k)
            if v is None:
                v = nxt
                m[k] = v
                nxt += 1
            out[i] = v
        self.size = nxt
        return out


class VectorizedNetStat:
    def __init__(self, lambdas=None, init_capacity=100_000, dtype=np.float32):
        if lambdas is None:
            self.lambdas = np.array([5.0, 3.0, 1.0, 0.1, 0.01], dtype=np.float64)
        else:
            self.lambdas = np.asarray(lambdas, dtype=np.float64)
        self.L = int(self.lambdas.shape[0])
        self.dtype = dtype
        self.mi_index = FlowIndexer()
        self.h_index = FlowIndexer()
        self.hh_cov_index = FlowIndexer()
        self.jit_index = FlowIndexer()
        self.hp_index = FlowIndexer()
        self.hphp_cov_index = FlowIndexer()

        self._alloc_states(max(int(init_capacity), 1024))

    def _alloc_states(self, cap):
        L = self.L
        eps = 1e-20

        # MI
        self.cap_mi = cap
        self.mi_state = np.zeros((cap, L, 3), dtype=np.float64)
        self.mi_last = np.zeros(cap, dtype=np.float64)
        self.mi_state[:, :, 0] = eps

        # H (hosts)
        self.cap_h = cap
        self.h_state = np.zeros((cap, L, 3), dtype=np.float64)
        self.h_last = np.zeros(cap, dtype=np.float64)
        self.h_state[:, :, 0] = eps

        # HH cov
        self.cap_hh_cov = cap
        self.hh_cov_a = np.full((cap,), -1, dtype=np.int64)
        self.hh_cov_b = np.full((cap,), -1, dtype=np.int64)
        self.hh_CF3 = np.zeros((cap, L), dtype=np.float64)
        self.hh_w3 = np.full((cap, L), eps, dtype=np.float64)
        self.hh_last_cf3 = np.zeros((cap, L), dtype=np.float64)
        self.hh_ex_t = np.zeros((cap, L, 2, 3), dtype=np.float64)
        self.hh_ex_v = np.zeros((cap, L, 2, 3), dtype=np.float64)
        self.hh_ex_indx = np.zeros((cap, L, 2), dtype=np.int64)
        self.hh_ex_n = np.zeros((cap, L, 2), dtype=np.int64)

        # JIT
        self.cap_jit = cap
        self.jit_state = np.zeros((cap, L, 3), dtype=np.float64)
        self.jit_last = np.zeros(cap, dtype=np.float64)
        self.jit_prev = np.zeros(cap, dtype=np.float64)
        self.jit_state[:, :, 0] = eps

        # Hp (endpoints)
        self.cap_hp = cap
        self.hp_state = np.zeros((cap, L, 3), dtype=np.float64)
        self.hp_last = np.zeros(cap, dtype=np.float64)
        self.hp_state[:, :, 0] = eps

        # HpHp cov
        self.cap_hphp_cov = cap
        self.hphp_cov_a = np.full((cap,), -1, dtype=np.int64)
        self.hphp_cov_b = np.full((cap,), -1, dtype=np.int64)
        self.hphp_CF3 = np.zeros((cap, L), dtype=np.float64)
        self.hphp_w3 = np.full((cap, L), eps, dtype=np.float64)
        self.hphp_last_cf3 = np.zeros((cap, L), dtype=np.float64)
        self.hphp_ex_t = np.zeros((cap, L, 2, 3), dtype=np.float64)
        self.hphp_ex_v = np.zeros((cap, L, 2, 3), dtype=np.float64)
        self.hphp_ex_indx = np.zeros((cap, L, 2), dtype=np.int64)
        self.hphp_ex_n = np.zeros((cap, L, 2), dtype=np.int64)

    def _ensure_capacity(self, needed, which):
        # grow helper
        if which == 'mi':
            cap = self.cap_mi
        elif which == 'h':
            cap = self.cap_h
        elif which == 'hh_cov':
            cap = self.cap_hh_cov
        elif which == 'jit':
            cap = self.cap_jit
        elif which == 'hp':
            cap = self.cap_hp
        else:
            cap = self.cap_hphp_cov

        if needed <= cap:
            return

        new_cap = cap
        while new_cap < needed:
            new_cap = int(new_cap * 1.5) + 1024

        L = self.L
        eps = 1e-20

        if which == 'mi':
            s = np.zeros((new_cap, L, 3), dtype=np.float64)
            s[:cap] = self.mi_state
            s[cap:, :, 0] = eps
            self.mi_state = s
            last = np.zeros(new_cap, dtype=np.float64)
            last[:cap] = self.mi_last
            self.mi_last = last
            self.cap_mi = new_cap

        elif which == 'h':
            s = np.zeros((new_cap, L, 3), dtype=np.float64)
            s[:cap] = self.h_state
            s[cap:, :, 0] = eps
            self.h_state = s
            last = np.zeros(new_cap, dtype=np.float64)
            last[:cap] = self.h_last
            self.h_last = last
            self.cap_h = new_cap

        elif which == 'hh_cov':
            self.hh_cov_a = np.concatenate([self.hh_cov_a, np.full((new_cap - cap,), -1, dtype=np.int64)])
            self.hh_cov_b = np.concatenate([self.hh_cov_b, np.full((new_cap - cap,), -1, dtype=np.int64)])
            cf3 = np.zeros((new_cap, L), dtype=np.float64)
            cf3[:cap] = self.hh_CF3
            self.hh_CF3 = cf3
            w3 = np.full((new_cap, L), eps, dtype=np.float64)
            w3[:cap] = self.hh_w3
            self.hh_w3 = w3
            lcf3 = np.zeros((new_cap, L), dtype=np.float64)
            lcf3[:cap] = self.hh_last_cf3
            self.hh_last_cf3 = lcf3
            et = np.zeros((new_cap, L, 2, 3), dtype=np.float64)
            et[:cap] = self.hh_ex_t
            self.hh_ex_t = et
            ev = np.zeros((new_cap, L, 2, 3), dtype=np.float64)
            ev[:cap] = self.hh_ex_v
            self.hh_ex_v = ev
            ei = np.zeros((new_cap, L, 2), dtype=np.int64)
            ei[:cap] = self.hh_ex_indx
            self.hh_ex_indx = ei
            en = np.zeros((new_cap, L, 2), dtype=np.int64)
            en[:cap] = self.hh_ex_n
            self.hh_ex_n = en
            self.cap_hh_cov = new_cap

        elif which == 'jit':
            s = np.zeros((new_cap, L, 3), dtype=np.float64)
            s[:cap] = self.jit_state
            s[cap:, :, 0] = eps
            self.jit_state = s
            last = np.zeros(new_cap, dtype=np.float64)
            last[:cap] = self.jit_last
            self.jit_last = last
            prev = np.zeros(new_cap, dtype=np.float64)
            prev[:cap] = self.jit_prev
            self.jit_prev = prev
            self.cap_jit = new_cap

        elif which == 'hp':
            s = np.zeros((new_cap, L, 3), dtype=np.float64)
            s[:cap] = self.hp_state
            s[cap:, :, 0] = eps
            self.hp_state = s
            last = np.zeros(new_cap, dtype=np.float64)
            last[:cap] = self.hp_last
            self.hp_last = last
            self.cap_hp = new_cap

        else:  # hphp_cov
            self.hphp_cov_a = np.concatenate([self.hphp_cov_a, np.full((new_cap - cap,), -1, dtype=np.int64)])
            self.hphp_cov_b = np.concatenate([self.hphp_cov_b, np.full((new_cap - cap,), -1, dtype=np.int64)])
            cf3 = np.zeros((new_cap, L), dtype=np.float64)
            cf3[:cap] = self.hphp_CF3
            self.hphp_CF3 = cf3
            w3 = np.full((new_cap, L), eps, dtype=np.float64)
            w3[:cap] = self.hphp_w3
            self.hphp_w3 = w3
            lcf3 = np.zeros((new_cap, L), dtype=np.float64)
            lcf3[:cap] = self.hphp_last_cf3
            self.hphp_last_cf3 = lcf3
            et = np.zeros((new_cap, L, 2, 3), dtype=np.float64)
            et[:cap] = self.hphp_ex_t
            self.hphp_ex_t = et
            ev = np.zeros((new_cap, L, 2, 3), dtype=np.float64)
            ev[:cap] = self.hphp_ex_v
            self.hphp_ex_v = ev
            ei = np.zeros((new_cap, L, 2), dtype=np.int64)
            ei[:cap] = self.hphp_ex_indx
            self.hphp_ex_indx = ei
            en = np.zeros((new_cap, L, 2), dtype=np.int64)
            en[:cap] = self.hphp_ex_n
            self.hphp_ex_n = en
            self.cap_hphp_cov = new_cap

    def _init_cov_endpoints_first_seen(self, cov_ids, a, b, cov_a_arr, cov_b_arr):
        for k in range(cov_ids.shape[0]):
            cid = int(cov_ids[k])
            if cov_a_arr[cid] == -1:
                cov_a_arr[cid] = int(a[k])
                cov_b_arr[cid] = int(b[k])

    def process_arrays(self, sip_int, dip_int, src_port, dst_port, proto, pkt_len, ts_sec):
        sip_int = np.asarray(sip_int, dtype=np.int64)
        dip_int = np.asarray(dip_int, dtype=np.int64)
        src_port = np.asarray(src_port, dtype=np.int64)
        dst_port = np.asarray(dst_port, dtype=np.int64)
        proto = np.asarray(proto, dtype=np.int64)
        pkt_len = np.asarray(pkt_len, dtype=np.float64)
        ts_sec = np.asarray(ts_sec, dtype=np.float64)

        mi_keys = _mix_u64_vec(sip_int, tag=0xA1)
        mi_ids = self.mi_index.map_keys(mi_keys)
        self._ensure_capacity(self.mi_index.size, 'mi')

        host_src_keys = _mix_u64_vec(sip_int, tag=0xB0)
        host_dst_keys = _mix_u64_vec(dip_int, tag=0xB0)

        h_src_ids = self.h_index.map_keys(host_src_keys)
        h_dst_ids = self.h_index.map_keys(host_dst_keys)
        self._ensure_capacity(self.h_index.size, 'h')

        a = np.minimum(h_src_ids, h_dst_ids)
        b = np.maximum(h_src_ids, h_dst_ids)
        hh_cov_keys = _mix_u64_vec(a, b, tag=0xC0)
        hh_cov_ids = self.hh_cov_index.map_keys(hh_cov_keys)
        self._ensure_capacity(self.hh_cov_index.size, 'hh_cov')
        self._init_cov_endpoints_first_seen(hh_cov_ids, a, b, self.hh_cov_a, self.hh_cov_b)

        jit_keys = _mix_u64_vec(sip_int, dip_int, tag=0xC3)
        jit_ids = self.jit_index.map_keys(jit_keys)
        self._ensure_capacity(self.jit_index.size, 'jit')

        is_tcpudp = (proto == 6) | (proto == 17)
        proto_class = np.where(proto == 1, 1, np.where(is_tcpudp, 2, 0)).astype(np.int64)
        sp = np.where(is_tcpudp, src_port, 0)
        dp = np.where(is_tcpudp, dst_port, 0)

        ep_src_keys = _mix_u64_vec(sip_int, proto_class, sp, tag=0xD1)
        ep_dst_keys = _mix_u64_vec(dip_int, proto_class, dp, tag=0xD1)

        hp_src_ids = self.hp_index.map_keys(ep_src_keys)
        hp_dst_ids = self.hp_index.map_keys(ep_dst_keys)
        self._ensure_capacity(self.hp_index.size, 'hp')

        a2 = np.minimum(hp_src_ids, hp_dst_ids)
        b2 = np.maximum(hp_src_ids, hp_dst_ids)
        hphp_cov_keys = _mix_u64_vec(a2, b2, tag=0xD2)
        hphp_cov_ids = self.hphp_cov_index.map_keys(hphp_cov_keys)
        self._ensure_capacity(self.hphp_cov_index.size, 'hphp_cov')
        self._init_cov_endpoints_first_seen(hphp_cov_ids, a2, b2, self.hphp_cov_a, self.hphp_cov_b)

        feat_mi = update_1d_stats_kernel(mi_ids, ts_sec, pkt_len, self.lambdas, self.mi_state, self.mi_last)
        feat_hh = update_1d2d_exact_kernel(
            h_src_ids,
            h_dst_ids,
            ts_sec,
            pkt_len,
            self.lambdas,
            self.h_state,
            self.h_last,
            hh_cov_ids,
            self.hh_cov_a,
            self.hh_cov_b,
            self.hh_CF3,
            self.hh_w3,
            self.hh_last_cf3,
            self.hh_ex_t,
            self.hh_ex_v,
            self.hh_ex_indx,
            self.hh_ex_n,
        )

        jitter_vals = calc_jitter_values(jit_ids, ts_sec, self.jit_prev)
        feat_jit = update_1d_stats_kernel(jit_ids, ts_sec, jitter_vals, self.lambdas, self.jit_state, self.jit_last)

        feat_hphp = update_1d2d_exact_kernel(
            hp_src_ids,
            hp_dst_ids,
            ts_sec,
            pkt_len,
            self.lambdas,
            self.hp_state,
            self.hp_last,
            hphp_cov_ids,
            self.hphp_cov_a,
            self.hphp_cov_b,
            self.hphp_CF3,
            self.hphp_w3,
            self.hphp_last_cf3,
            self.hphp_ex_t,
            self.hphp_ex_v,
            self.hphp_ex_indx,
            self.hphp_ex_n,
        )

        X = np.concatenate([feat_mi, feat_hh, feat_jit, feat_hphp], axis=1)
        if self.dtype == np.float32:
            finfo = np.finfo(np.float32)
            X = np.nan_to_num(X, nan=0.0, posinf=finfo.max, neginf=-finfo.max)
            X = np.clip(X, -finfo.max, finfo.max)

        return X.astype(self.dtype, copy=False)


@njit(cache=True, fastmath=True)
def update_1d_stats_kernel(flow_ids, timestamps, values, lambdas, state_mem, last_ts_mem):
    n_packets = timestamps.shape[0]
    L = lambdas.shape[0]
    feats = np.empty((n_packets, L * 3), dtype=np.float64)

    for i in range(n_packets):
        fid = flow_ids[i]
        t = timestamps[i]
        v = values[i]

        last_t = last_ts_mem[fid]
        dt = t - last_t
        if dt < 0.0:
            dt = 0.0
        last_ts_mem[fid] = t

        base = 0
        for j in range(L):
            lam = lambdas[j]
            factor = 2.0 ** (-lam * dt)

            w = state_mem[fid, j, 0] * factor
            s1 = state_mem[fid, j, 1] * factor
            s2 = state_mem[fid, j, 2] * factor

            w += 1.0
            s1 += v
            s2 += v * v

            state_mem[fid, j, 0] = w
            state_mem[fid, j, 1] = s1
            state_mem[fid, j, 2] = s2

            mean = s1 / w
            var = s2 / w - mean * mean
            if var < 0.0:
                var = 0.0

            feats[i, base + 0] = w
            feats[i, base + 1] = mean
            feats[i, base + 2] = var
            base += 3

    return feats


@njit(cache=True, fastmath=True)
def update_1d2d_exact_kernel(
    src_ids,
    dst_ids,
    timestamps,
    values,
    lambdas,
    st_state,
    st_last,
    cov_ids,
    cov_a,
    cov_b,
    cov_CF3,
    cov_w3,
    cov_last,
    ex_t,
    ex_v,
    ex_indx,
    ex_n,
):
    n = timestamps.shape[0]
    L = lambdas.shape[0]
    feats = np.empty((n, 7 * L), dtype=np.float64)

    for i in range(n):
        sid = src_ids[i]
        did = dst_ids[i]
        cid = cov_ids[i]
        t = timestamps[i]
        v = values[i]

        base = 0

        for j in range(L):
            lam = lambdas[j]

            # ---- update src incStat (processDecay+insert)
            last_t = st_last[sid]
            dt = t - last_t
            if dt < 0.0:
                dt = 0.0
            st_last[sid] = t

            factor = 2.0 ** (-lam * dt)

            w1 = st_state[sid, j, 0] * factor
            s1_1 = st_state[sid, j, 1] * factor
            s2_1 = st_state[sid, j, 2] * factor

            w1 += 1.0
            s1_1 += v
            s2_1 += v * v

            st_state[sid, j, 0] = w1
            st_state[sid, j, 1] = s1_1
            st_state[sid, j, 2] = s2_1

            mean1 = s1_1 / w1
            var1 = s2_1 / w1 - mean1 * mean1
            if var1 < 0.0:
                var1 = 0.0
            std1 = var1**0.5

            # ---- dst stats (NO decay at time t)
            w2 = st_state[did, j, 0]
            if w2 <= 0.0:
                w2 = 1e-20
            mean2 = st_state[did, j, 1] / w2
            var2 = st_state[did, j, 2] / w2 - mean2 * mean2
            if var2 < 0.0:
                var2 = 0.0
            std2 = var2**0.5

            # ---- cov decay (processDecay)
            last_c = cov_last[cid, j]
            dtc = t - last_c
            if dtc < 0.0:
                dtc = 0.0
            if dtc > 0.0:
                factor_c = 2.0 ** (-lam * dtc)
                cov_CF3[cid, j] *= factor_c
                cov_w3[cid, j] *= factor_c
                cov_last[cid, j] = t

            a_id = cov_a[cid]
            side = 0
            if sid != a_id:
                side = 1

            idx = ex_indx[cid, j, side]
            nn = ex_n[cid, j, side]
            ex_t[cid, j, side, idx] = t
            idx = (idx + 1) % 3
            nn += 1
            ex_indx[cid, j, side] = idx
            ex_n[cid, j, side] = nn
            ex_v[cid, j, side, (idx - 1) % 3] = v

            o_side = 1 - side
            o_idx = ex_indx[cid, j, o_side]
            o_n = ex_n[cid, j, o_side]

            if o_n < 2:
                if o_n == 1:
                    v_other = ex_v[cid, j, o_side, (o_idx - 1) % 3]
                else:
                    v_other = 0.0
            else:
                last_ot = ex_t[cid, j, o_side, (o_idx - 1) % 3]
                if o_n == 2:
                    md = ex_t[cid, j, o_side, (o_idx % 3)] - ex_t[cid, j, o_side, (o_idx - 1) % 3]
                else:
                    md = (
                        ex_t[cid, j, o_side, (o_idx % 3)]
                        - ex_t[cid, j, o_side, (o_idx - 1) % 3]
                        + ex_t[cid, j, o_side, (o_idx - 1) % 3]
                        - ex_t[cid, j, o_side, (o_idx - 2) % 3]
                    ) / 2.0

                if (t - last_ot) / (md + 1e-10) > 10.0:
                    v_other = ex_v[cid, j, o_side, (o_idx - 1) % 3]
                else:
                    if o_n == 2:
                        t0 = ex_t[cid, j, o_side, 0]
                        t1 = ex_t[cid, j, o_side, 1]
                        y0 = ex_v[cid, j, o_side, 0]
                        y1 = ex_v[cid, j, o_side, 1]
                        v_other = (y0 * ((t - t1) / (t0 - t1 + 1e-20))) + (y1 * ((t - t0) / (t1 - t0 + 1e-20)))
                    else:
                        if o_idx == 0:
                            t0 = ex_t[cid, j, o_side, 0]
                            t1 = ex_t[cid, j, o_side, 1]
                            t2 = ex_t[cid, j, o_side, 2]
                            y0 = ex_v[cid, j, o_side, 0]
                            y1 = ex_v[cid, j, o_side, 1]
                            y2 = ex_v[cid, j, o_side, 2]
                        elif o_idx == 1:
                            t0 = ex_t[cid, j, o_side, 1]
                            t1 = ex_t[cid, j, o_side, 2]
                            t2 = ex_t[cid, j, o_side, 0]
                            y0 = ex_v[cid, j, o_side, 1]
                            y1 = ex_v[cid, j, o_side, 2]
                            y2 = ex_v[cid, j, o_side, 0]
                        else:
                            t0 = ex_t[cid, j, o_side, 2]
                            t1 = ex_t[cid, j, o_side, 0]
                            t2 = ex_t[cid, j, o_side, 1]
                            y0 = ex_v[cid, j, o_side, 2]
                            y1 = ex_v[cid, j, o_side, 0]
                            y2 = ex_v[cid, j, o_side, 1]

                        w0 = ((t - t1) / (t0 - t1 + 1e-20)) * ((t - t2) / (t0 - t2 + 1e-20))
                        w1 = ((t - t0) / (t1 - t0 + 1e-20)) * ((t - t2) / (t1 - t2 + 1e-20))
                        w2 = ((t - t0) / (t2 - t0 + 1e-20)) * ((t - t1) / (t2 - t1 + 1e-20))
                        v_other = y0 * w0 + y1 * w1 + y2 * w2

            if side == 0:
                cov_CF3[cid, j] += (v - mean1) * (v_other - mean2)
            else:
                cov_CF3[cid, j] += (v - mean2) * (v_other - mean1)
            cov_w3[cid, j] += 1.0

            cov = cov_CF3[cid, j] / cov_w3[cid, j]
            ss = std1 * std2
            if ss != 0.0:
                pcc = cov / ss
            else:
                pcc = 0.0

            radius = (var1 + var2) ** 0.5
            magnitude = (mean1 * mean1 + mean2 * mean2) ** 0.5

            feats[i, base + 0] = w1
            feats[i, base + 1] = mean1
            feats[i, base + 2] = var1
            feats[i, base + 3] = radius
            feats[i, base + 4] = magnitude
            feats[i, base + 5] = cov
            feats[i, base + 6] = pcc
            base += 7

    return feats


@njit(cache=True)
def calc_jitter_values(flow_ids, timestamps, prev_ts_mem):
    n = timestamps.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        fid = flow_ids[i]
        t = timestamps[i]
        last = prev_ts_mem[fid]
        if last == 0.0:
            out[i] = 0.0
        else:
            out[i] = t - last
            if out[i] < 0.0:
                out[i] = 0.0
        prev_ts_mem[fid] = t
    return out