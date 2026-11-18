"""
Bitstream of AIR's quantised Gaussians (FeedForwardGaussianCodec, model/quantize.py): bytes that a decoder reads
without the network. model/codec.py writes and renders them.

AIR itself only estimates the size of its quantised Gaussians (model/compress.py: one categorical ANS stream per
channel, with the side information counted from a varint estimate); it writes no bitstream. This module writes
those same streams for real, with the side information they need, plus what the estimate leaves out: the picture
size, the patch grid and the stage masks (which candidate Gaussians of each stage are active), and the per-picture
quantiser steps and offsets that the quantisers derive from the class token of the network (the decoder has no
network). Pure numpy and constriction; no torch.

Layout (little-endian):
    magic "AIR1"
    u16 width, u16 height, u8 patch_size, u8 stages
    8 x u8 bits per channel (x, y, scale x, scale y, rotation, Y, Cr, Cb)
    8 x (f32 step, f32 offset) of the uniform quantisers: value = symbol * step + offset
    per stage: one stream over {0, 1} of length L = ceil(H / p) * ceil(W / p) (row-major patch grid): the mask
    per channel: one stream of length N = number of ones in all masks, in stage order then patch order
Stream: varint k (distinct symbols), first symbol (varint), k - 1 gaps - 1 (varints), k counts (varints),
varint w, w x u32 words of a constriction AnsCoder with the categorical model counts / sum (no words when k = 1:
constriction cannot build a categorical model of a single symbol, and the stream is then fully described by it).

Gaussian i of stage s at patch (r, c) has centre (c p + (p - 1) / 2, r p + (p - 1) / 2) clamped to the picture,
as AIR's PatchEmbedding: position = centre + dequantised offset (pixels).
"""

import struct

import numpy as np

MAGIC = b"AIR1"
CHANNELS = ("x", "y", "sx", "sy", "rot", "Y", "Cr", "Cb")


def _varint(v: int, out: bytearray):
    v = int(v)
    if v < 0:
        raise ValueError(f"varint of a negative value {v}")
    while v >= 0x80:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    out.append(v)


class _Reader:
    def __init__(self, data: bytes):
        self.data, self.pos = memoryview(data), 0

    def take(self, n):
        if self.pos + n > len(self.data):
            raise ValueError("truncated AIR bitstream")
        b = self.data[self.pos:self.pos + n]
        self.pos += n
        return b

    def unpack(self, fmt):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))

    def varint(self):
        v, shift = 0, 0
        while True:
            b = self.take(1)[0]
            v |= (b & 0x7F) << shift
            if b < 0x80:
                return v
            shift += 7


def _model(counts):
    import constriction
    p = np.asarray(counts, dtype=np.float64)
    return constriction.stream.model.Categorical(p / p.sum(), perfect=False)


def encode_stream(symbols: np.ndarray, out: bytearray):
    """One categorical ANS stream of non-negative integer symbols (its length is known to the decoder)."""
    import constriction
    symbols = np.asarray(symbols).reshape(-1).astype(np.int64, copy=False)
    if symbols.size == 0:
        _varint(0, out)
        return
    if symbols.min() < 0:
        raise ValueError("negative symbol")
    top = int(symbols.max())
    if top < (1 << 16):  # a small alphabet (AIR's symbols have at most 8 bits): a count table, no sort
        table = np.bincount(symbols, minlength=top + 1)
        unique = np.flatnonzero(table)
        counts = table[unique]
        lut = np.zeros(top + 1, np.int32)
        lut[unique] = np.arange(len(unique), dtype=np.int32)
        inverse = lut[symbols]
    else:
        unique, inverse, counts = np.unique(symbols, return_inverse=True, return_counts=True)
    _varint(len(unique), out)
    _varint(unique[0], out)
    for gap in np.diff(unique):
        _varint(int(gap) - 1, out)
    for c in counts:
        _varint(c, out)
    if len(unique) == 1:
        _varint(0, out)
        return
    enc = constriction.stream.stack.AnsCoder()
    enc.encode_reverse(inverse.astype(np.int32), _model(counts))
    words = enc.get_compressed().astype("<u4")
    _varint(len(words), out)
    out += words.tobytes()


def decode_stream(r: _Reader, length: int) -> np.ndarray:
    import constriction
    k = r.varint()
    if k == 0:
        if length:
            raise ValueError("empty stream for a non-empty field")
        return np.zeros(0, np.int64)
    unique = [r.varint()]
    for _ in range(k - 1):
        unique.append(unique[-1] + r.varint() + 1)
    counts = [r.varint() for _ in range(k)]
    if sum(counts) != length:
        raise ValueError(f"stream counts sum to {sum(counts)}, expected {length}")
    n_words = r.varint()
    unique = np.asarray(unique, np.int64)
    if k == 1:
        return np.full(length, unique[0], np.int64)
    words = np.frombuffer(r.take(4 * n_words), dtype="<u4").astype(np.uint32)
    dec = constriction.stream.stack.AnsCoder(words)
    return unique[np.asarray(dec.decode(_model(counts), length), np.int64)]


def grid(width, height, patch_size):
    """(rows, cols) of the patch grid of a picture."""
    return -(-height // patch_size), -(-width // patch_size)


def encode(width, height, patch_size, bits, steps, offsets, masks, symbols) -> bytes:
    """masks: [stages, L] bool; symbols: [N, 8] non-negative ints (N = masks.sum()); bits, steps, offsets: 8 each."""
    masks = np.asarray(masks, bool)
    symbols = np.asarray(symbols)
    rows, cols = grid(width, height, patch_size)
    if masks.ndim != 2 or masks.shape[1] != rows * cols:
        raise ValueError(f"masks {masks.shape}, expected [stages, {rows * cols}]")
    if symbols.shape != (int(masks.sum()), len(CHANNELS)):
        raise ValueError(f"symbols {symbols.shape}, expected [{int(masks.sum())}, {len(CHANNELS)}]")
    symbols = np.ascontiguousarray(symbols.T, np.int64)  # [8, N]: each channel's stream contiguous
    qmax = (1 << np.asarray(bits, np.int64)) - 1
    if symbols.size and (symbols.min() < 0 or np.any(symbols.max(axis=1) > qmax)):
        raise ValueError("symbol outside the quantiser range")
    out = bytearray(MAGIC)
    out += struct.pack("<HHBB", width, height, patch_size, masks.shape[0])
    out += bytes(int(b) for b in bits)
    for s, o in zip(steps, offsets):
        out += struct.pack("<ff", float(s), float(o))
    for m in masks:
        encode_stream(m.astype(np.int64), out)
    for c in range(len(CHANNELS)):
        encode_stream(symbols[c], out)
    return bytes(out)


def parse_header(data: bytes):
    """(width, height, patch_size, stages) without decoding the streams."""
    if data[:4] != MAGIC:
        raise ValueError("not an AIR bitstream")
    return struct.unpack_from("<HHBB", data, 4)


def decode(data: bytes) -> dict:
    """The fields of an AIR bitstream: width, height, patch_size, bits [8], steps [8] and offsets [8] (float32),
    masks [stages, L] bool, symbols [N, 8] int64."""
    r = _Reader(data)
    if bytes(r.take(4)) != MAGIC:
        raise ValueError("not an AIR bitstream")
    width, height, patch_size, stages = r.unpack("<HHBB")
    bits = list(r.take(len(CHANNELS)))
    so = np.frombuffer(r.take(8 * len(CHANNELS)), dtype="<f4").reshape(-1, 2)
    rows, cols = grid(width, height, patch_size)
    masks = np.stack([decode_stream(r, rows * cols).astype(bool) for _ in range(stages)]) if stages else np.zeros((0, rows * cols), bool)
    n = int(masks.sum())
    # [N, 8] as a view of the channel-major streams (no strided copy)
    symbols = np.stack([decode_stream(r, n) for _ in CHANNELS]).T if n else np.zeros((0, len(CHANNELS)), np.int64)
    if r.pos != len(data):
        raise ValueError(f"{len(data) - r.pos} trailing bytes in the AIR bitstream")
    return {"width": width, "height": height, "patch_size": patch_size, "bits": bits, "steps": so[:, 0].astype(np.float32),
            "offsets": so[:, 1].astype(np.float32), "masks": masks, "symbols": symbols}
