"""Expert-cache simulation for arcinferzero from routes recorded with ARC_DUMP_ROUTES (zero/engine.h layer_moe).

Each record: int32 layer, int32 tokens, then tokens x NU int32 expert ids. Records with tokens > 1 are prompt chunks,
tokens == 1 single-token generation steps. The cache holds (layer, expert) entries; capacity is a fraction of all
of them. Hit rate and SSD bytes are counted over the generation steps (the prompt warms the cache).

usage: python cache_sim.py routes.bin [routes2.bin ...] [--layers 40 --experts 256 --mb 1.36 --ssd 3.5 --tgpu 0.0505]
"""
import argparse
import collections
import struct
import sys


def load(path, nu):
    """(prompt records, generation records): each a list of (layer, [token ids tuples])"""
    prompt, gen = [], []
    data = open(path, 'rb').read()
    off = 0
    while off + 8 <= len(data):
        layer, t = struct.unpack_from('<ii', data, off)
        off += 8
        ids = struct.unpack_from('<%di' % (t * nu), data, off)
        off += t * nu * 4
        (prompt if t > 1 else gen).append((layer, [ids[i * nu:(i + 1) * nu] for i in range(t)]))
    return prompt, gen


def token_stream(records, layers):
    """Rebuild per-token, per-layer accesses: list of tokens, each a list of (layer, expert)."""
    out = []
    # generation: records come layer by layer for each token (40 records per token)
    cur, last_layer = [], -1
    for layer, toks in records:
        if layer >= layers:
            continue  # the MTP block
        if layer <= last_layer and cur:
            out.append(cur)
            cur = []
        last_layer = layer
        for e in toks[0]:
            cur.append((layer, e))
    if cur:
        out.append(cur)
    return out


def prompt_accesses(records, layers):
    acc = []
    for layer, toks in records:
        if layer >= layers:
            continue
        for tok in toks:
            for e in tok:
                acc.append((layer, e))
    return acc


def simulate(policy, cap, warm, gen, pin_frac=0.9):
    hits = miss = 0
    if policy == 'lru':
        c = collections.OrderedDict()
        def touch(k, count):
            nonlocal hits, miss
            if k in c:
                c.move_to_end(k); hits += count; return
            miss += count
            c[k] = 1
            if len(c) > cap: c.popitem(last=False)
        for k in warm: touch(k, 0)
        for tok in gen:
            for k in tok: touch(k, 1)
    elif policy == 'lfu':
        cnt = collections.Counter(warm)
        c = set(k for k, _ in cnt.most_common(cap))
        for tok in gen:
            for k in tok:
                cnt[k] += 1
                if k in c: hits += 1; continue
                miss += 1
                victim = min(c, key=lambda x: cnt[x]) if len(c) >= cap else None
                if victim is not None and cnt[victim] < cnt[k]:
                    c.discard(victim); c.add(k)
                elif victim is None:
                    c.add(k)
    elif policy in ('pin', 'pin+lru'):
        cnt = collections.Counter(warm)
        npin = cap if policy == 'pin' else int(cap * pin_frac)
        pinned = set(k for k, _ in cnt.most_common(npin))
        dyn = collections.OrderedDict()
        dcap = cap - npin
        for tok in gen:
            for k in tok:
                if k in pinned: hits += 1; continue
                if k in dyn: dyn.move_to_end(k); hits += 1; continue
                miss += 1
                if dcap > 0:
                    dyn[k] = 1
                    if len(dyn) > dcap: dyn.popitem(last=False)
    elif policy == 'layerlru':
        # an LRU per layer with an equal share of the slots (a global LRU evicts the next layer's experts during
        # single-token decoding: it walks the layers in a cycle)
        caches = collections.defaultdict(collections.OrderedDict)
        nl = max(1, len(set(l for l, _ in warm) | set(l for tok in gen for l, _ in tok)))
        lcap = max(1, cap // nl)
        def touch(k, count):
            nonlocal hits, miss
            c = caches[k[0]]
            if k in c:
                c.move_to_end(k); hits += count; return
            miss += count
            c[k] = 1
            if len(c) > lcap: c.popitem(last=False)
        for k in warm: touch(k, 0)
        for tok in gen:
            for k in tok: touch(k, 1)
    elif policy == 'belady':
        seq = [k for tok in gen for k in tok]
        nxt = [0] * len(seq)
        last = {}
        for i in range(len(seq) - 1, -1, -1):
            nxt[i] = last.get(seq[i], 1 << 60); last[seq[i]] = i
        # warm with the most frequent prompt experts
        cnt = collections.Counter(warm)
        c = {k: 1 << 59 for k, _ in cnt.most_common(cap)}  # unknown next use until seen
        import heapq
        for i, k in enumerate(seq):
            if k in c: hits += 1; c[k] = nxt[i]; continue
            miss += 1
            if len(c) >= cap:
                victim = max(c, key=c.get); del c[victim]
            c[k] = nxt[i]
    return hits, miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--layers', type=int, default=40)
    ap.add_argument('--experts', type=int, default=256)
    ap.add_argument('--nu', type=int, default=8, help='experts per token')
    ap.add_argument('--mb', type=float, default=1.36, help='MB per expert per layer (all its matrices)')
    ap.add_argument('--ssd', type=float, default=3.5, help='SSD GB/s')
    ap.add_argument('--tgpu', type=float, default=1 / 19.8, help='seconds per token with every expert cached')
    ap.add_argument('--caps', default='0.5,0.7,0.8,0.88,0.95')
    a = ap.parse_args()
    total = a.layers * a.experts
    for path in a.files:
        prompt, gen = load(path, a.nu)
        warm = prompt_accesses(prompt, a.layers)
        toks = token_stream(gen, a.layers)
        # locality: share of a layer's experts also used by the previous token in that layer
        reuse = []
        for i in range(1, len(toks)):
            prev = collections.defaultdict(set)
            for l, e in toks[i - 1]: prev[l].add(e)
            same = sum(1 for l, e in toks[i] if e in prev[l])
            reuse.append(same / max(1, len(toks[i])))
        uniq = len(set(k for t in toks for k in t))
        print(f'\n== {path.split(chr(92))[-1]}: prompt {len(warm) // a.nu // a.layers if warm else 0} tokens, '
              f'generation {len(toks)} tokens; distinct experts in generation {uniq} of {total} '
              f'({100 * uniq / total:.0f}%); reused from the previous token {100 * sum(reuse) / max(1, len(reuse)):.0f}%')
        print(f'{"cache":>6} {"policy":>8} {"hit %":>6} {"SSD MB/tok":>10} {"tok/s serial":>12} {"tok/s overlap":>13}')
        for capf in [float(x) for x in a.caps.split(',')]:
            cap = int(total * capf)
            for pol in ('lru', 'layerlru', 'lfu', 'pin', 'pin+lru', 'belady'):
                h, m = simulate(pol, cap, warm, toks)
                n = max(1, len(toks))
                mb = m / n * a.mb
                io = mb / 1000 / a.ssd
                print(f'{capf * 100:5.0f}% {pol:>8} {100 * h / max(1, h + m):6.1f} {mb:10.1f} '
                      f'{1 / (a.tgpu + io):12.1f} {1 / max(a.tgpu, io):13.1f}')


if __name__ == '__main__':
    main()
