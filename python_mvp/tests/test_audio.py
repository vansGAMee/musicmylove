"""Unit and contract tests for audio representation learning and multimodal components."""
import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from python_mvp.audio_pipeline import (
    create_mel_filterbank, compute_log_mel, extract_segments,
    SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, SEGMENT_SAMPLES
)
from python_mvp.audio_network import AudioEncoder
from python_mvp.networks import MultiInterest, MultimodalRanker
from python_mvp.train_audio import info_nce_loss, cross_modal_loss


def test_mel_filterbank_properties():
    fb = create_mel_filterbank(sr=22050, n_fft=1024, n_mels=128)
    assert fb.shape == (128, 513)
    assert (fb >= 0).all()
    # Check that filterbank has non-zero energy across all mel bands
    assert (fb.sum(dim=1) > 0).all()


def test_compute_log_mel_synthetic_waveform():
    t = torch.linspace(0, 4.0, int(SAMPLE_RATE * 4.0))
    sine = 0.5 * torch.sin(2 * math.pi * 440.0 * t)
    fb = create_mel_filterbank()
    mel = compute_log_mel(sine, mel_fb=fb)
    assert mel.shape[0] == 128
    assert mel.dim() == 2
    assert not torch.isnan(mel).any()
    assert not torch.isinf(mel).any()


def test_extract_segments_deterministic_and_training():
    waveform = torch.randn(int(SAMPLE_RATE * 15.0))  # 15 seconds
    segs_eval = extract_segments(waveform, num_segments=3, is_training=False)
    assert len(segs_eval) == 3
    for s in segs_eval:
        assert len(s) == SEGMENT_SAMPLES

    # Evaluation segments must be deterministic
    segs_eval2 = extract_segments(waveform, num_segments=3, is_training=False)
    for s1, s2 in zip(segs_eval, segs_eval2):
        assert torch.equal(s1, s2)

    # Training segments with RNG
    rng = np.random.default_rng(42)
    segs_train = extract_segments(waveform, num_segments=3, is_training=True, rng=rng)
    assert len(segs_train) == 3


def test_audio_encoder_forward_backward_and_norm():
    torch.manual_seed(42)
    model = AudioEncoder(dim=128)
    # Batch of 4 segments: [B, 1, 128, 173]
    x = torch.randn(4, 1, 128, 173)
    out = model(x)
    assert out.shape == (4, 128)
    norms = torch.linalg.vector_norm(out, dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    # Verify backprop
    loss = out.sum()
    loss.backward()
    assert model.conv_in.weight.grad is not None
    assert model.proj.weight.grad is not None
    assert model.attn.weight.grad is not None


def test_audio_encoder_multisegment():
    model = AudioEncoder(dim=128)
    # Batch of 2 tracks, 3 segments each: [2, 3, 1, 128, 173]
    x = torch.randn(2, 3, 1, 128, 173)
    out = model(x)
    assert out.shape == (2, 128)
    norms = torch.linalg.vector_norm(out, dim=-1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5)


def test_contrastive_loss_optimization():
    torch.manual_seed(42)
    z1 = F.normalize(torch.randn(8, 128), dim=-1)
    # Correlated view
    z2 = F.normalize(z1 + 0.05 * torch.randn(8, 128), dim=-1)
    z_rand = F.normalize(torch.randn(8, 128), dim=-1)

    loss_paired = info_nce_loss(z1, z2)
    loss_unpaired = info_nce_loss(z1, z_rand)
    assert loss_paired.item() < loss_unpaired.item()


def test_multimodal_multi_interest_attention_gating():
    torch.manual_seed(42)
    dim = 96
    model = MultiInterest(dim=dim)
    graph_emb = torch.randn(20, dim)
    audio_emb = torch.randn(20, dim)
    audio_mask = torch.tensor([True, False, True, False, True] * 4)

    # Single-modal forward
    heads_single, masses_single, _ = model(graph_emb, [0, 1, 2, 3])
    assert heads_single.shape == (model.queries.shape[0], dim)

    # Multimodal forward with audio
    heads_multi, masses_multi, _ = model(graph_emb, [0, 1, 2, 3], audio_embeddings=audio_emb, audio_mask=audio_mask)
    assert heads_multi.shape == (model.queries.shape[0], dim)
    assert not torch.isnan(heads_multi).any()


def test_multimodal_ranker():
    torch.manual_seed(42)
    dim = 96
    ranker = MultimodalRanker(dim=dim)
    cand_graph = torch.randn(5, dim)
    cand_audio = torch.randn(5, dim)
    cand_mask = torch.tensor([[1.0], [0.0], [1.0], [0.0], [1.0]])
    head_graph = torch.randn(5, dim)
    head_audio = torch.randn(5, dim)
    evidence = torch.randn(5, 6)

    scores = ranker(cand_graph, cand_audio, cand_mask, head_graph, head_audio, evidence)
    assert scores.shape == (5,)
    assert not torch.isnan(scores).any()

    # Backprop
    scores.sum().backward()
    assert ranker.net[0].weight.grad is not None
