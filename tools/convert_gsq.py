#!/usr/bin/env python3
# Converts the ISTA-DASLab GSQ checkpoint of Qwen3.6-35B-A3B (compressed-tensors "pack-quantized": 2-bit routed
# experts, int8 attention / DeltaNet / shared experts / lm_head) into a GGUF for arcinfer (arch "qwen35moe").
#
#   int8 matrices (group 128, or per channel for lm_head) -> Q8_0: the same int8 values, the scale per 32 as f16
#   2-bit experts (q in {-2,-1,0,1} x bf16 scale per 128 inputs) -> the layout of kernels/moe.cl:
#       blk.N.ffn_{gate,up,down}_exps.weight  I32  [N*K/16, n_expert]   codes (q + 2), see repack_codes()
#       blk.N.ffn_{gate,up,down}_exps.scale   BF16 [N*K/128, n_expert]  scales, see repack_scales()
#   tensor transforms as llama.cpp's convert_hf_to_gguf.py (Qwen3_5MoeTextModel): norm weights + 1 (except the
#   DeltaNet gated norm), A = -exp(A_log), DeltaNet V heads reordered from grouped to tiled order.
# The vision tower and the MTP block (which this checkpoint ships without its routed experts) are not converted.
# Tokenizer metadata is copied from a GGUF with the same tokenizer (--vocab-from) after checking it against
# tokenizer.json; the chat template comes from the checkpoint.
#
# usage: python -P tools/convert_gsq.py <snapshot dir> <out.gguf> --vocab-from <qwen3.5/3.6 gguf>
import argparse, json, os, struct, sys, time
import numpy as np

GGML_F32, GGML_Q8_0, GGML_I32, GGML_BF16 = 0, 8, 26, 30
ALIGN = 32


# ---------------------------------------------------------------- safetensors
class Checkpoint:
    def __init__(self, snap):
        self.snap = snap
        self.map = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
        self.hdr = {}
        self.files = {}

    def meta(self, name):
        fn = os.path.join(self.snap, self.map[name])
        if fn not in self.hdr:
            with open(fn, "rb") as f:
                n, = struct.unpack("<Q", f.read(8))
                self.hdr[fn] = (json.loads(f.read(n)), 8 + n)
            self.files[fn] = open(fn, "rb")
        h, base = self.hdr[fn]
        return fn, h[name], base

    def get(self, name, f32=False):
        fn, m, base = self.meta(name)
        a, b = m["data_offsets"]
        dt = {"BF16": np.uint16, "F32": np.float32, "I32": np.int32, "I64": np.int64}[m["dtype"]]
        f = self.files[fn]
        f.seek(base + a)
        arr = np.frombuffer(f.read(b - a), dtype=dt).reshape(m["shape"])
        if f32 and m["dtype"] == "BF16":
            arr = (arr.astype(np.uint32) << 16).view(np.float32)
        return arr


# ---------------------------------------------------------------- transforms
def reorder_v(x, dim, nk, nvk, hd):
    """V heads from grouped (by K head) to tiled order along `dim` (llama.cpp _reorder_v_heads)."""
    sh = list(x.shape)
    ns = sh[:dim] + [nk, nvk, hd] + sh[dim + 1:]
    perm = list(range(len(ns)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return np.ascontiguousarray(x.reshape(ns).transpose(perm)).reshape(sh)


def int8_codes(ck, p):
    """int8 matrix [N][K] (stored + 128, 4 per int32, LSB first) and its scales [N][K/g] as f32."""
    q = ck.get(p + ".weight_packed")
    q = (q.view(np.uint8).reshape(q.shape[0], -1).astype(np.int16) - 128).astype(np.int8)
    return q, ck.get(p + ".weight_scale", f32=True).astype(np.float32)


def q8_0_bytes(q, s):
    """Q8_0 blocks of an int8 matrix with group scales s [N][K/g] (g a multiple of 32)."""
    N, K = q.shape
    d = np.repeat(s, (K // 32) // s.shape[1], axis=1).astype(np.float16)
    blk = np.empty((N, K // 32), dtype=[("d", "<f2"), ("qs", "i1", 32)])
    blk["d"] = d
    blk["qs"] = q.reshape(N, K // 32, 32)
    return blk.tobytes()


def q8_0_from_f32(w):
    N, K = w.shape
    b = w.reshape(N, K // 32, 32).astype(np.float32)
    amax = np.abs(b).max(axis=2)
    d = amax / 127.0
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1), 0)
    q = np.clip(np.rint(b * inv[..., None]), -127, 127).astype(np.int8)
    blk = np.empty((N, K // 32), dtype=[("d", "<f2"), ("qs", "i1", 32)])
    blk["d"] = d.astype(np.float16)
    blk["qs"] = q
    return blk.tobytes()


# Within each 32-bit word of 16 codes, code e goes to bit 8 * (e % 4) + 2 * (e // 4): then
# (word >> 2j) & 0x03030303 holds the codes 4j .. 4j+3 in its bytes 0..3 (four consecutive inputs, one DP4a operand).
_SHIFT = np.array([8 * (e % 4) + 2 * (e // 4) for e in range(16)], dtype=np.uint32)


def repack_codes(packed):
    """2-bit codes [N][K/16] (LSB-first, 16 per word) -> words [N/16][K/128][8][16]: column group, 128-input group,
    word, lane (one output column per lane; a sub-group block read of 8 words gives each lane its column)."""
    u = packed.view(np.uint32)
    N, W = u.shape
    out = np.zeros_like(u)
    for e in range(16):
        out |= ((u >> np.uint32(2 * e)) & np.uint32(3)) << _SHIFT[e]
    return np.ascontiguousarray(out.reshape(N // 16, 16, W // 8, 8).transpose(0, 2, 3, 1))


def repack_scales(s_bf16):
    """bf16 scales [N][K/128] -> [N/16][K/128][16]."""
    N, G = s_bf16.shape
    return np.ascontiguousarray(s_bf16.reshape(N // 16, 16, G).transpose(0, 2, 1))


# ---------------------------------------------------------------- GGUF writing
def kv_str(key, s):
    kb, sb = key.encode(), s.encode()
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 8) + struct.pack("<Q", len(sb)) + sb


def kv_u32(key, v):
    kb = key.encode()
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<II", 4, v)


def kv_f32(key, v):
    kb = key.encode()
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<If", 6, v)


def kv_bool(key, v):
    kb = key.encode()
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<IB", 7, 1 if v else 0)


def kv_arr_i32(key, vals):
    kb = key.encode()
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<IIQ", 9, 5, len(vals)) + struct.pack(f"<{len(vals)}i", *vals)


def read_gguf_kvs(path):
    """Raw KV entries of a GGUF: {key: (encoded entry bytes, decoded value or None)}."""
    SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    out = {}
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        ver, = struct.unpack("<I", f.read(4))
        nt, nkv = struct.unpack("<QQ", f.read(16))

        def rstr():
            n, = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "surrogateescape")

        def rval(t):
            if t == 8:
                return rstr()
            if t == 9:
                et, = struct.unpack("<I", f.read(4))
                n, = struct.unpack("<Q", f.read(8))
                if et in FMT:
                    return list(struct.unpack(f"<{n}{FMT[et][1]}", f.read(n * SZ[et])))
                return [rval(et) for _ in range(n)]
            return struct.unpack(FMT[t], f.read(SZ[t]))[0]

        for _ in range(nkv):
            p0 = f.tell()
            k = rstr()
            t, = struct.unpack("<I", f.read(4))
            v = rval(t)
            p1 = f.tell()
            f.seek(p0)
            raw = f.read(p1 - p0)
            out[k] = (raw, v)
    return out


def check_vocab(snap, tokens):
    tj = json.load(open(os.path.join(snap, "tokenizer.json"), encoding="utf-8"))
    ref = {i: t for t, i in tj["model"]["vocab"].items()}
    for a in tj.get("added_tokens", []):
        ref[a["id"]] = a["content"]
    bad = [i for i, t in ref.items() if i >= len(tokens) or tokens[i] != t]
    return len(ref), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("out")
    ap.add_argument("--vocab-from", required=True, help="GGUF with the same tokenizer (Qwen3.5 / Qwen3.6)")
    ap.add_argument("--layers", type=int, default=0, help="convert only the first N layers (testing)")
    args = ap.parse_args()
    ck = Checkpoint(args.snapshot)
    cfg = json.load(open(os.path.join(args.snapshot, "config.json")))
    tc = cfg.get("text_config", cfg)
    E, nl = tc["hidden_size"], tc["num_hidden_layers"]
    if args.layers:
        nl = args.layers
    nexp, nused, ffe, ffs = tc["num_experts"], tc["num_experts_per_tok"], tc["moe_intermediate_size"], tc["shared_expert_intermediate_size"]
    nh, nkv, hd = tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
    lk, lv, nk, nv = tc["linear_key_head_dim"], tc["linear_value_head_dim"], tc["linear_num_key_heads"], tc["linear_num_value_heads"]
    rp = tc.get("rope_parameters", {})
    rope_theta = rp.get("rope_theta", tc.get("rope_theta", 1e7))
    prf = rp.get("partial_rotary_factor", tc.get("partial_rotary_factor", 0.25))
    sect = list(rp.get("mrope_section", [11, 11, 10])) + [0]
    vocab = tc["vocab_size"]
    interval = tc.get("full_attention_interval", 4)
    types = tc.get("layer_types") or ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(nl)]
    nvk = nv // nk

    # ---- metadata
    kv = [kv_str("general.architecture", "qwen35moe"),
          kv_str("general.name", "Qwen3.6-35B-A3B GSQ 2-bit (ISTA-DASLab), arcinfer"),
          kv_str("arcinfer.expert_format", "gsq2-v1")]
    a = "qwen35moe."
    kv += [kv_u32(a + "block_count", nl), kv_u32(a + "context_length", tc.get("max_position_embeddings", 262144)),
           kv_u32(a + "embedding_length", E), kv_u32(a + "attention.head_count", nh),
           kv_u32(a + "attention.head_count_kv", nkv), kv_arr_i32(a + "rope.dimension_sections", sect),
           kv_f32(a + "rope.freq_base", float(rope_theta)),
           kv_f32(a + "attention.layer_norm_rms_epsilon", float(tc.get("rms_norm_eps", 1e-6))),
           kv_u32(a + "expert_count", nexp), kv_u32(a + "expert_used_count", nused),
           kv_bool(a + "expert_weights_norm", True),
           kv_u32(a + "attention.key_length", hd), kv_u32(a + "attention.value_length", hd),
           kv_u32(a + "expert_feed_forward_length", ffe), kv_u32(a + "expert_shared_feed_forward_length", ffs),
           kv_u32(a + "ssm.conv_kernel", tc["linear_conv_kernel_dim"]), kv_u32(a + "ssm.state_size", lk),
           kv_u32(a + "ssm.group_count", nk), kv_u32(a + "ssm.time_step_rank", nv),
           kv_u32(a + "ssm.inner_size", lv * nv), kv_u32(a + "full_attention_interval", interval),
           kv_u32(a + "rope.dimension_count", int(hd * prf))]
    ref = read_gguf_kvs(args.vocab_from)
    toks = ref["tokenizer.ggml.tokens"][1]
    if len(toks) != vocab:
        sys.exit(f"--vocab-from has {len(toks)} tokens, the model {vocab}")
    n_checked, bad = check_vocab(args.snapshot, toks)
    if bad:
        sys.exit(f"tokenizer mismatch: {len(bad)} of {n_checked} tokens differ (first ids {bad[:5]})")
    print(f"tokenizer: {n_checked} tokens match tokenizer.json")
    for k, (raw, _) in ref.items():
        if k.startswith("tokenizer.") and k != "tokenizer.chat_template":
            kv.append(raw)
    tpl = os.path.join(args.snapshot, "chat_template.jinja")
    if os.path.exists(tpl):
        kv.append(kv_str("tokenizer.chat_template", open(tpl, encoding="utf-8").read()))
    elif "tokenizer.chat_template" in ref:
        kv.append(ref["tokenizer.chat_template"][0])

    # ---- tensor plan: (name, dims (ne0 first), type, nbytes, producer)
    plan = []
    L = "model.language_model."

    def add(name, dims, typ, nbytes, fn):
        plan.append((name, dims, typ, nbytes, fn))

    def f32(src, transform=None, plus1=False):
        def fn():
            x = ck.get(src, f32=True).astype(np.float32)
            if plus1:
                x = x + 1.0
            if transform:
                x = transform(x)
            return np.ascontiguousarray(x, dtype=np.float32).tobytes()
        return fn

    def q8(src, row_perm=None, col_blocks=None):
        def fn():
            q, s = int8_codes(ck, src)
            if row_perm is not None:
                q, s = row_perm(q), row_perm(s)
            if col_blocks is not None:
                q, s = col_blocks(q, s)
            return q8_0_bytes(q, s)
        return fn

    def q8_size(N, K):
        return N * K // 32 * 34

    add("token_embd.weight", [E, vocab], GGML_BF16, E * vocab * 2,
        lambda: np.ascontiguousarray(ck.get(L + "embed_tokens.weight")).tobytes())

    def lm_head():
        q = ck.get("lm_head.weight_packed")
        q = (q.view(np.uint8).reshape(q.shape[0], -1).astype(np.int16) - 128).astype(np.int8)
        s = ck.get("lm_head.weight_scale", f32=True).astype(np.float32)
        out = bytearray()
        for r0 in range(0, q.shape[0], 16384):
            out += q8_0_bytes(q[r0:r0 + 16384], s[r0:r0 + 16384])
        return bytes(out)

    add("output.weight", [E, vocab], GGML_Q8_0, q8_size(vocab, E), lm_head)
    add("output_norm.weight", [E], GGML_F32, E * 4, f32(L + "norm.weight", plus1=True))

    qk = 2 * lk * nk  # q and k channels of the DeltaNet projection

    def v_rows(x):  # in_proj_qkv: reorder the V rows only
        return np.concatenate([x[:qk], reorder_v(x[qk:], 0, nk, nvk, lv)], 0)

    def out_cols(q, s):  # out_proj: reorder 128-column blocks (head_v_dim == group size)
        assert lv == 128 and s.shape[1] * 128 == q.shape[1]
        return reorder_v(q, 1, nk, nvk, lv), reorder_v(s, 1, nk, nvk, 1)

    for il in range(nl):
        p, g = L + f"layers.{il}.", f"blk.{il}."
        add(g + "attn_norm.weight", [E], GGML_F32, E * 4, f32(p + "input_layernorm.weight", plus1=True))
        add(g + "post_attention_norm.weight", [E], GGML_F32, E * 4, f32(p + "post_attention_layernorm.weight", plus1=True))
        if types[il] == "linear_attention":
            la = p + "linear_attn."
            C = 2 * lk * nk + lv * nv
            add(g + "attn_qkv.weight", [E, C], GGML_Q8_0, q8_size(C, E), q8(la + "in_proj_qkv", row_perm=v_rows))
            add(g + "attn_gate.weight", [E, lv * nv], GGML_Q8_0, q8_size(lv * nv, E),
                q8(la + "in_proj_z", row_perm=lambda x: reorder_v(x, 0, nk, nvk, lv)))
            add(g + "ssm_out.weight", [lv * nv, E], GGML_Q8_0, q8_size(E, lv * nv), q8(la + "out_proj", col_blocks=out_cols))
            rv1 = lambda x: reorder_v(x, 0, nk, nvk, 1)
            add(g + "ssm_alpha.weight", [E, nv], GGML_F32, nv * E * 4, f32(la + "in_proj_a.weight", transform=rv1))
            add(g + "ssm_beta.weight", [E, nv], GGML_F32, nv * E * 4, f32(la + "in_proj_b.weight", transform=rv1))
            add(g + "ssm_a", [nv], GGML_F32, nv * 4, f32(la + "A_log", transform=lambda x: rv1(-np.exp(x))))
            add(g + "ssm_dt.bias", [nv], GGML_F32, nv * 4, f32(la + "dt_bias", transform=rv1))
            add(g + "ssm_conv1d.weight", [tc["linear_conv_kernel_dim"], C], GGML_F32, C * tc["linear_conv_kernel_dim"] * 4,
                f32(la + "conv1d.weight", transform=lambda x: v_rows(x.reshape(x.shape[0], -1))))
            add(g + "ssm_norm.weight", [lv], GGML_F32, lv * 4, f32(la + "norm.weight"))
        else:
            sa = p + "self_attn."
            add(g + "attn_q.weight", [E, 2 * nh * hd], GGML_Q8_0, q8_size(2 * nh * hd, E), q8(sa + "q_proj"))
            add(g + "attn_k.weight", [E, nkv * hd], GGML_Q8_0, q8_size(nkv * hd, E), q8(sa + "k_proj"))
            add(g + "attn_v.weight", [E, nkv * hd], GGML_Q8_0, q8_size(nkv * hd, E), q8(sa + "v_proj"))
            add(g + "attn_output.weight", [nh * hd, E], GGML_Q8_0, q8_size(E, nh * hd), q8(sa + "o_proj"))
            add(g + "attn_q_norm.weight", [hd], GGML_F32, hd * 4, f32(sa + "q_norm.weight", plus1=True))
            add(g + "attn_k_norm.weight", [hd], GGML_F32, hd * 4, f32(sa + "k_norm.weight", plus1=True))
        m = p + "mlp."
        add(g + "ffn_gate_inp.weight", [E, nexp], GGML_F32, nexp * E * 4, f32(m + "gate.weight"))
        add(g + "ffn_gate_inp_shexp.weight", [E], GGML_F32, E * 4, f32(m + "shared_expert_gate.weight",
                                                                         transform=lambda x: x.reshape(-1)))
        add(g + "ffn_gate_shexp.weight", [E, ffs], GGML_Q8_0, q8_size(ffs, E), q8(m + "shared_expert.gate_proj"))
        add(g + "ffn_up_shexp.weight", [E, ffs], GGML_Q8_0, q8_size(ffs, E), q8(m + "shared_expert.up_proj"))
        add(g + "ffn_down_shexp.weight", [ffs, E], GGML_Q8_0, q8_size(E, ffs), q8(m + "shared_expert.down_proj"))
        for w, (N, K) in (("gate", (ffe, E)), ("up", (ffe, E)), ("down", (E, ffe))):
            words, groups = N * K // 16, N * K // 128

            def codes(w=w, m=m):
                return b"".join(repack_codes(ck.get(m + f"experts.{e}.{w}_proj.weight_packed")).tobytes()
                                for e in range(nexp))

            def scales(w=w, m=m):
                return b"".join(repack_scales(ck.get(m + f"experts.{e}.{w}_proj.weight_scale")).tobytes()
                                for e in range(nexp))

            add(g + f"ffn_{w}_exps.weight", [words, nexp], GGML_I32, words * nexp * 4, codes)
            add(g + f"ffn_{w}_exps.scale", [groups, nexp], GGML_BF16, groups * nexp * 2, scales)

    # ---- write: header, KVs, tensor infos, then the data (each tensor aligned)
    infos, off = [], 0
    for name, dims, typ, nb, _ in plan:
        nbn = name.encode()
        infos.append(struct.pack("<Q", len(nbn)) + nbn + struct.pack("<I", len(dims)) +
                     struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", typ, off))
        off += (nb + ALIGN - 1) // ALIGN * ALIGN
    tmp = args.out + ".part"
    t0 = time.time()
    with open(tmp, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, len(plan), len(kv)))
        for e in kv:
            f.write(e)
        for e in infos:
            f.write(e)
        f.write(b"\0" * ((-f.tell()) % ALIGN))
        data0 = f.tell()
        for i, (name, dims, typ, nb, fn) in enumerate(plan):
            b = fn()
            if len(b) != nb:
                sys.exit(f"{name}: {len(b)} bytes, expected {nb}")
            f.write(b)
            f.write(b"\0" * ((-len(b)) % ALIGN))
            if name.endswith("ffn_down_exps.scale"):
                print(f"  layer {name.split('.')[1]} done ({f.tell() / 2**30:.2f} GiB, {time.time() - t0:.0f} s)", flush=True)
        assert f.tell() - data0 == off
    os.replace(tmp, args.out)
    print(f"wrote {args.out}: {len(plan)} tensors, {os.path.getsize(args.out) / 2**30:.2f} GiB in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
