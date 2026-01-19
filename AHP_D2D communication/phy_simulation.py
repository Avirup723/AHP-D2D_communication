import numpy as np
from pyldpc import make_ldpc, decode, get_message
from commpy.channels import awgn
from numpy.random import default_rng
from scipy.sparse import issparse

rng = default_rng()
latency_samples = {}

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
THERM_NOISE_DBM_PER_HZ = -174.0
DEFAULT_NF_DB         = {"5G": 6.0, "LTE": 7.0, "D2D": 5.0}
TTI_SCHED_S           = {"5G": 0.25e-3, "LTE": 1.00e-3, "D2D": 0.20e-3}
HARQ_RTT_S            = {"5G": 0.0002,   "LTE": 0.0004,  "D2D": 0.00015}
BASE_QUEUE_S          = {"5G": 0.000005, "LTE": 0.00002, "D2D": 0.000004}
MAX_MOD_CAP           = {"5G": 256,     "LTE": 64,     "D2D": 256}

CENTER_FREQ           = {"5G": 6e9,   "LTE": 2.1e9,  "D2D": 2.4e9}
SCS_KHZ               = {"5G": 30.0,    "LTE": 15.0,   "D2D": 15.0}

GAMMA_COEF            = {"5G": 60.0, "LTE": 140.0, "D2D": 180.0}
K_SPEED_DB_PER_MS     = {"5G": 0.06, "LTE": 0.10,  "D2D": 0.13}

FAST_FADING_SIGMA_DB0 = {"5G": 0.8,  "LTE": 1.6,  "D2D": 2.2}

N_DIV = {"5G": 4, "LTE": 2, "D2D": 1}

SLICE_SENS_COEF = {"eMBB": 1.0, "URLLC": 1.6, "mMTC": 2.0}
# small penalty to lift ultra-low BER into a visible range
SLICE_EXTRA_PENALTY_DB = {"eMBB": 0.0, "URLLC": 0.8, "mMTC": 1.2}

def modulation_penalty_db(M: int) -> float:
    table = {2:0.0, 4:0.0, 8:1.0, 16:2.5, 32:3.5, 64:5.0, 128:6.5, 256:7.8, 512:9.0}
    return float(table.get(int(M), 7.8))

def _rsrp_dbm_to_snr_db(rsrp_dbm: float, bandwidth_hz: float, nf_db: float) -> float:
    noise_dbm = THERM_NOISE_DBM_PER_HZ + 10.0 * np.log10(bandwidth_hz) + nf_db
    return float(rsrp_dbm - noise_dbm)

def _doppler_penalty_db(speed_mps: float, tech: str) -> float:
   
    fc  = CENTER_FREQ.get(tech, 2.1e9)
    scs = SCS_KHZ.get(tech, 15.0)
    c   = 3e8
    fd  = speed_mps * fc / c                     # Hz
    Tc  = 0.423 / max(fd, 1e-6)
    Ts  = 1.0 / (scs * 1000.0)
    gamma = GAMMA_COEF.get(tech, 100.0)
    ratio = gamma * (Ts / Tc)
    coh_term_db = 10.0 * np.log10(1.0 + ratio * ratio)
    speed_term  = K_SPEED_DB_PER_MS.get(tech, 0.08) * float(speed_mps)
    return float(coh_term_db + speed_term)

def select_mcs_from_snr(snr_db: float):

    if snr_db < 0.0:
        return 4,   2.0, 24
    elif snr_db < 6.0:
        return 16,  1.6, 18
    elif snr_db < 12.0:
        return 64,  1.2, 12
    elif snr_db < 20.0:
        return 256, 1.0, 10
    else:
        return 256, 0.8, 8

def _encode_bits_with_G(G, data_bits):
  
    if issparse(G):
        cw = G.dot(data_bits)
        cw = np.asarray(cw).reshape(-1)
    else:
        cw = G @ data_bits
    return (cw % 2).astype(np.int8)


LDPC_CACHE = {}

def _get_ldpc(n=512, d_v=2, d_c=4):

    key = (int(n), int(d_v), int(d_c))
    if key in LDPC_CACHE:
        return LDPC_CACHE[key]
    H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
    LDPC_CACHE[key] = (H, G)
    return H, G


def simulate_phy_link(rsrp_dbm,
                      modulation_order=16,
                      bandwidth=20e6,
                      ldpc_gain=0.0,
                      speed=2,
                      tech="D2D",
                      user_id=0,
                      packet_size_bytes=1500,
                      noise_figure_db=None,
                      slice_type="eMBB"):
   

    nf = (DEFAULT_NF_DB.get(tech, 7.0) if noise_figure_db is None else float(noise_figure_db))
    snr_db_nominal = _rsrp_dbm_to_snr_db(rsrp_dbm, bandwidth, nf)

    pre_snr_db = snr_db_nominal + float(ldpc_gain) - _doppler_penalty_db(speed, tech) * SLICE_SENS_COEF.get(slice_type, 1.0)

    mcs_mod, extra_gain_db, mcs_iter_hint = select_mcs_from_snr(pre_snr_db)
    tech_cap  = MAX_MOD_CAP.get(tech, 64)
    mcs_mod   = min(mcs_mod, tech_cap)
    mod_order = int(min(int(modulation_order), int(mcs_mod)))
    mod_order = max(2, mod_order)
    bits_per_symbol = int(np.log2(mod_order))

    total_gain_db = float(ldpc_gain) + float(extra_gain_db)
    snr_db_eff = snr_db_nominal + total_gain_db - _doppler_penalty_db(speed, tech) * SLICE_SENS_COEF.get(slice_type, 1.0)

    n_div = N_DIV.get(tech, 1)
    snr_db_eff += 10.0 * np.log10(max(1, n_div))

    snr_db_eff -= modulation_penalty_db(mod_order)

    snr_db_eff -= SLICE_EXTRA_PENALTY_DB.get(slice_type, 0.0)

    impl_loss_db = 0.5

    # ------------ LDPC  ------------ #
    n = 512
    d_v, d_c = 2, 4
    H, G = _get_ldpc(n, d_v, d_c)
    k = G.shape[1]
    rate = float(k) / float(n)

    if snr_db_eff >= 12.0:
        maxiter = max(8, mcs_iter_hint)
    elif snr_db_eff >= 6.0:
        maxiter = max(14, mcs_iter_hint)
    else:
        maxiter = max(24, mcs_iter_hint)

    packet_bits = int(packet_size_bytes * 8)
    num_blocks = max(1, int(np.ceil(packet_bits / k)))
    num_blocks = min(num_blocks, 16)

    total_errors = 0
    total_bits = 0
    sched_accum = 0.0

    sigma_ff_db_base = FAST_FADING_SIGMA_DB0.get(tech, 1.2) * (1.0 + 0.25 * float(speed))
    sigma_ff_db = sigma_ff_db_base * SLICE_SENS_COEF.get(slice_type, 1.0)

    for _ in range(num_blocks):
        data = rng.integers(0, 2, k, dtype=np.int8)

        snr_db_blk = snr_db_eff + np.random.normal(0.0, sigma_ff_db)
        snr_db_blk_for_awgn = snr_db_blk - impl_loss_db
        snr_lin_blk = float(10.0 ** (snr_db_blk / 10.0))

        cw_bits = _encode_bits_with_G(G, data)
        symbols = 1.0 - 2.0 * cw_bits.astype(np.float32)
        noisy   = awgn(symbols, snr_db_blk_for_awgn)

        decoded = decode(H, noisy, snr=snr_lin_blk, maxiter=maxiter)
        decoded_data = get_message(G, decoded)

        bit_errors = int(np.sum(data != decoded_data))
        total_errors += bit_errors
        total_bits  += k

        sched_accum += TTI_SCHED_S.get(tech, 1.0e-3)

    ber = float(np.clip(total_errors / max(1, total_bits), 1e-9, 0.2))
    bler = float(np.clip(1.0 - np.exp(-ber * k), 0.0, 0.95))

    se = bits_per_symbol * rate * (1.0 - bler)
    throughput_bps  = max(1e-6, se * bandwidth)
    throughput_mbps = throughput_bps / 1e6

    tx_time = (packet_size_bytes * 8) / throughput_bps
    sched_delay = sched_accum
    base_dec = 0.00010 if mod_order <= 16 else 0.00018
    per_iter = 0.000008 if mod_order <= 16 else 0.000012
    snr_factor = max(0.5, 10.0 / max(2.0, snr_db_eff + 2.0))
    dec_delay = (base_dec + per_iter * maxiter) * snr_factor
    harq_delay = bler * HARQ_RTT_S.get(tech, 0.006)

    load = np.clip(throughput_bps / max(bandwidth * bits_per_symbol * rate, 1.0), 0.0, 0.98)
    q_base = BASE_QUEUE_S.get(tech, 0.0003)
    q_mob  = (1.0 + 0.06 * float(speed))
    queue_delay = q_base * (load / max(1e-6, (1.0 - load))) * q_mob
    queue_delay = min(queue_delay, 0.050)

    c = 3e8
    tx_range   = {"5G": 120.0, "LTE": 220.0, "D2D": 80.0}
    prop_delay = tx_range.get(tech, 120.0) / c

    latency_s = tx_time + sched_delay + dec_delay + harq_delay + queue_delay + prop_delay

    if user_id not in latency_samples:
        latency_samples[user_id] = []
    latency_samples[user_id].append(latency_s)
    if len(latency_samples[user_id]) > 12:
        latency_samples[user_id].pop(0)
    jitter_s = float(np.std(latency_samples[user_id])) if len(latency_samples[user_id]) > 1 else 0.0

    return ber, float(throughput_mbps), float(latency_s), float(jitter_s)
