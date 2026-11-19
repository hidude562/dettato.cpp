"""Reference tensors of the drum stage (dettato/drums), for cpp/tools/parity_drums.cpp.

    python py/dump_refs_drums.py OUT_DIR --audio WAV [--pop AUDIO] [--fixtures mix30,mid87,song4min]

Runs the PyTorch code on the CPU in float32. Fixtures, cut from --audio (a 44.1 kHz file, the router picks the
orchestral model on all of it) and --pop (a 44.1 kHz pop song with a drum kit, read as dettato.audio.load does):

    mix30     --audio 0-30 s: T = 3000 feature frames, every model takes the plain ``embed`` path
    mid87     --audio 0-87 s: T = 8700, between the two long-input conditions: the drum-kit model (``T > 9000``) takes
              ``embed``, the experts (``T > chunk + 2 * margin = 8400``) take ``embed_long``
    song4min  all of --audio (240 s, T = 24000): every model takes ``embed_long``
    pop30     --pop 30-60.013 s: the drum-kit decode path; an odd frame count (3001, trimmed to 3000)
    tiny      --pop 40-40.257 s: T = 25 (odd, 12 output frames), the smallest graphs

Per fixture, OUT_DIR/<fixture>/:

    x44.npy                 the 44.1 kHz input, float32
    gain.npy                level_gain(x44), float64 (1,)
    x16.npy                 soxr.resample(x44 * gain, 44100, 16000), the 16 kHz signal (the parity tool can inject it)
    lo.npy hi.npy cq.npy    the three log-power blocks of features(), untrimmed, (frames, bands) float32
    feats.npy               features(x44), (T, 280) float16
    kit_*.npy               the drum-kit DrumNet: embed, tb (after the conv block), gru (after the GRU block), h (trunk
                            states), z (onset logits), v (velocity logits), all (T/2, .); stem (56, T/2, 20) for mix30
    ex{i}_*.npy             expert i: its DrumNet base (embed, tb, gru, h, z, v), cq (T/2, 128), a (T/2, 192), r1 (the first
                            GRU layer, T/2, 256), rnn (T/2, 256), att (r + ao(att), before the LayerNorm), r (after it),
                            router (1,) logit, drum (T/2, 61), timpani (T/2, 13); for mix30 also cqin (3, T, 120) and cq1
                            (24, T, 120), the cq branch's input and its first layer
    P_*.npy                 probabilities(): router (1,) float64, kit, kit_vel, drum, timpani, vel
    hits_*.npy              transcribe(): onset, offset (float64), pitch (int64), velocity (float64)
    info.json               router, model, T, the embed path of each model

Every tensor taken by calling the sub-modules is checked bit for bit against the modules' own forward.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dettato_ref  # noqa: E402,F401  (puts the reference package on sys.path)
import importlib  # noqa: E402

FE = importlib.import_module("dettato.drums.features")  # (the package exports a function of the same name)
from dettato.drums.infer import DrumTranscriber  # noqa: E402
from dettato.drums.model import N_MEL  # noqa: E402

torch.set_num_threads(int(os.environ.get("DUMP_THREADS", "16")))

FIXTURES = {"mix30": ("audio", 0.0, 30.0), "mid87": ("audio", 0.0, 87.0), "song4min": ("audio", 0.0, None),
            "pop30": ("pop", 30.0, 30.013), "tiny": ("pop", 40.0, 0.257)}


def save(d: Path, name: str, a):
    d.mkdir(parents=True, exist_ok=True)
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    np.save(d / f"{name}.npy", np.ascontiguousarray(a))


def same(a: torch.Tensor, b: torch.Tensor, what: str):
    assert a.shape == b.shape and torch.equal(a, b), f"{what}: the decomposed forward differs from the module's"


@torch.no_grad()
def drumnet(net, x, long_path: bool, d: Path, p: str, stem: bool):
    """DrumNet.embed / embed_long + trunk, stage by stage. x: (1, T, 160)."""
    c = net.embed_long(x) if long_path else net.embed(x)
    if stem:
        save(d, f"{p}_stem", net.stem(((x + 5.0) / 3.0)[:, None])[0])
    save(d, f"{p}_embed", c[0])
    h = c
    names = {"tb": "tb", "gru": "gru", "attn": "h"}
    for spec_i, b in enumerate(net.temporal):
        h = b(h)
        kind = net.cfg["temporal"][spec_i]["t"] if len(net.temporal) == len(net.cfg["temporal"]) else None
        assert kind is not None, "one block per temporal spec is assumed by this dump"
        save(d, f"{p}_{names[kind]}", h[0])
    z, v = net.on(h), net.vel(h)
    h2, z2, v2 = net.trunk(c)
    same(h, h2, p + " h")
    same(z, z2, p + " z")
    same(v, v2, p + " v")
    save(d, f"{p}_z", z[0])
    save(d, f"{p}_v", v[0])
    return c, h, z, v


@torch.no_grad()
def expert(ex, x, d: Path, p: str, small: bool):
    """OrchestralExpert.forward, stage by stage. x: (1, T, 280)."""
    chunk, margin = 8000, 200
    T = x.shape[1] // 2 * 2
    x = x[:, :T]
    xm = x[..., :N_MEL]
    long_path = T > chunk + 2 * margin
    _, h0, z0, v0 = drumnet(ex.base, xm, long_path, d, p, stem=False)
    cqb = ex.cq
    xc = x[..., N_MEL:].float()
    if small:
        xn = (xc - xc.mean(dim=(1, 2), keepdim=True)) / 4.0
        lo = xn.amin(dim=(1, 2), keepdim=True)

        def shift(k):
            return torch.cat([xn[..., k:], lo.expand(-1, xn.shape[1], k)], -1)

        z = torch.stack([xn, shift(14), shift(24)], 1)
        save(d, f"{p}_cqin", z[0])
        save(d, f"{p}_cq1", F.silu(cqb.n1(cqb.c1(z)))[0])
    cq = cqb(xc)
    save(d, f"{p}_cq", cq[0])
    n = min(h0.shape[1], cq.shape[1])
    h0, z0, v0, cq = h0[:, :n], z0[:, :n], v0[:, :n], cq[:, :n]
    a = F.silu(ex.inp(torch.cat([h0.float(), cq, torch.sigmoid(z0.float())], -1)))
    save(d, f"{p}_a", a[0])
    # the first GRU layer on its own (the same CPU kernel as inside the two-layer module)
    g1 = torch.nn.GRU(ex.rnn.input_size, ex.rnn.hidden_size, num_layers=1, batch_first=True, bidirectional=True)
    g1.load_state_dict({k: v for k, v in ex.rnn.state_dict().items() if "_l0" in k})
    save(d, f"{p}_r1", g1(a)[0][0])
    r, _ = ex.rnn(a)
    save(d, f"{p}_rnn", r[0])
    B, Tn, D = r.shape
    q, k, v = ex.qkv(r).view(B, Tn, 3, ex.heads, D // ex.heads).permute(2, 0, 3, 1, 4)
    att = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, Tn, D)
    r = r + ex.ao(att)
    save(d, f"{p}_att", r[0])
    r = ex.ln(r)
    save(d, f"{p}_r", r[0])
    router = ex.gate(r.float().mean(1))[:, 0]
    out = {"base": z0.float(), "drum": z0.float() + ex.delta(r).float(), "timpani": ex.timp(r).float(),
           "velocity": v0.float(), "router": router}
    save(d, f"{p}_router", out["router"])
    save(d, f"{p}_drum", out["drum"][0])
    save(d, f"{p}_timpani", out["timpani"][0])
    return out, long_path


def hits_of(dt: DrumTranscriber, P: dict):
    """DrumTranscriber.transcribe after probabilities(): decode and the rounding of the returned dicts."""
    on, keys, vel, which = dt.decode(P)
    return [{"onset": round(float(a), 4), "offset": round(float(a) + 0.1, 4), "pitch": int(k), "is_drum": True,
             "velocity": round(float(np.clip(v, 0.05, 1.0)), 3)} for a, k, v in zip(on, keys, vel)], which


@torch.no_grad()
def dump(dt: DrumTranscriber, x44: np.ndarray, d: Path, name: str):
    import soxr

    small = name in ("mix30", "pop30", "tiny")
    save(d, "x44", x44)
    g = FE.level_gain(x44)
    save(d, "gain", np.array([g], np.float64))
    x = (x44 * g).astype(np.float32)
    x16 = soxr.resample(x, FE.SR, 16000).astype(np.float32)
    save(d, "x16", x16)
    lo = FE._logspec(x16, 512, 160, FE._filters("lo"))
    hi = FE._logspec(x, 1411, FE.SR // 100, FE._filters("hi"))
    cq = FE._logspec(x16, FE.CQ_NFFT, 160, FE._filters("cq"))
    save(d, "lo", lo)
    save(d, "hi", hi)
    save(d, "cq", cq)
    feats = FE.features(x44)
    T0 = min(len(x16) // 160, len(lo), len(hi), len(cq))
    assert np.array_equal(feats, np.concatenate([lo[:T0], hi[:T0], cq[:T0]], 1).astype(np.float16))
    save(d, "feats", feats)

    # the networks, as probabilities() feeds them
    f32 = feats.astype(np.float32)
    T = f32.shape[0] // 2 * 2
    xt = torch.from_numpy(np.ascontiguousarray(f32[:T]))[None]
    kit_long = T > 9000
    _, _, zk, vk = drumnet(dt.kit, xt[..., :160], kit_long, d, "kit", stem=small)
    zk2, vk2 = dt.kit(xt[..., :160])
    same(zk, zk2, "kit z")
    same(vk, vk2, "kit v")
    outs, ex_long = [], None
    for i, ex in enumerate(dt.experts):
        o, ex_long = expert(ex, xt, d, f"ex{i}", small and i == 0)
        o2 = ex(xt)
        for k in o:
            same(o[k], o2[k], f"ex{i} {k}")
        outs.append(o)

    P = dt.probabilities(f32)
    for k, v in P.items():
        save(d, f"P_{k}", np.array([v], np.float64) if k == "router" else v)
    hits, which = hits_of(dt, P)
    save(d, "hits_onset", np.array([h["onset"] for h in hits], np.float64))
    save(d, "hits_offset", np.array([h["offset"] for h in hits], np.float64))
    save(d, "hits_pitch", np.array([h["pitch"] for h in hits], np.int64))
    save(d, "hits_velocity", np.array([h["velocity"] for h in hits], np.float64))
    if small:
        assert dt.transcribe(x44) == hits, "transcribe() differs from probabilities() + decode()"
        assert dt.last_info == {"router": P["router"], "model": which}
    info = {"router": P["router"], "model": which, "T": int(feats.shape[0]), "n_hits": len(hits),
            "kit_embed": "embed_long" if kit_long else "embed", "expert_embed": "embed_long" if ex_long else "embed",
            "x16_len": int(len(x16)), "x44_len": int(len(x44))}
    (d / "info.json").write_text(json.dumps(info, indent=1) + "\n")
    print(f"{name}: {len(x44) / FE.SR:.1f} s, T {feats.shape[0]}, router {P['router']:.4f} -> {which}, {len(hits)} hits "
          f"(kit {info['kit_embed']}, experts {info['expert_embed']})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--audio", required=True, help="a 44.1 kHz recording of about 4 minutes (orchestral, no drum kit)")
    ap.add_argument("--pop", help="a 44.1 kHz pop song with a drum kit (the pop30 and tiny fixtures; skipped without it)")
    ap.add_argument("--fixtures", default=",".join(FIXTURES))
    a = ap.parse_args()
    if a.pop is None:
        a.fixtures = ",".join(f for f in a.fixtures.split(",") if FIXTURES[f][0] != "pop")
    from dettato import audio as A

    dt = DrumTranscriber.from_pretrained("cpu")
    out = Path(a.out)
    loaded = {}
    for name in a.fixtures.split(","):
        src, start, sec = FIXTURES[name]
        if src not in loaded:
            au = A.load(getattr(a, src))
            assert au.rate == FE.SR, f"{getattr(a, src)}: {au.rate} Hz, the fixtures need 44.1 kHz sources"
            loaded[src] = au.samples
        x = loaded[src]
        s0 = int(round(start * FE.SR))
        xa = np.ascontiguousarray(x[s0:] if sec is None else x[s0: s0 + int(round(sec * FE.SR))])
        dump(dt, xa, out / name, name)
    (out / "REF_COMMIT").write_text(dettato_ref.ref_commit() + "\n")


if __name__ == "__main__":
    main()
