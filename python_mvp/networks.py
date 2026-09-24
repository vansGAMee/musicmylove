"""Own randomly initialized trainable modules. Graph and taste are separate signals."""
import math
import torch
from torch import nn
from torch.nn import functional as F
try:
    from .config import DIM, HEADS
except ImportError:
    from config import DIM, HEADS


class GraphEncoder(nn.Module):
    def __init__(self, tracks, users, sessions, dim=DIM):
        super().__init__()
        self.track = nn.Embedding(tracks, dim)
        self.user = nn.Embedding(users, dim)
        self.session = nn.Embedding(sessions, dim)
        self.mix = nn.Parameter(torch.zeros(2))
        for embedding in (self.track, self.user, self.session):
            nn.init.normal_(embedding.weight, std=.05)

    def forward(self, user_graph, session_graph):
        def propagate(context, adjacency):
            h = torch.cat((self.track.weight, context.weight))
            layers = [h]
            for _ in range(2):
                h = torch.sparse.mm(adjacency, h)
                layers.append(h)
            return torch.stack(layers).mean(0)
        long = propagate(self.user, user_graph)
        local = propagate(self.session, session_graph)
        n = self.track.num_embeddings
        mix = self.mix.softmax(0)
        tracks = F.normalize(mix[0] * long[:n] + mix[1] * local[:n], dim=-1)
        return tracks, F.normalize(long[n:], dim=-1)


class MultiInterest(nn.Module):
    def __init__(self, dim=DIM, heads=HEADS):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(heads, dim) / math.sqrt(dim))
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.audio_proj = nn.Linear(dim, dim, bias=False)
        self.audio_gate = nn.Linear(dim * 2, 1)
        self.temperature = nn.Parameter(torch.tensor(-2.))
        nn.init.eye_(self.audio_proj.weight)
        nn.init.zeros_(self.audio_gate.weight)
        nn.init.zeros_(self.audio_gate.bias)

    def forward(self, embeddings, seeds, audio_embeddings=None, audio_mask=None):
        seeds = sorted(set(int(i) for i in seeds))
        if not seeds:
            raise ValueError('No resolved trained seeds')
        x = embeddings[seeds]
        if audio_embeddings is not None and audio_mask is not None:
            a = audio_embeddings[seeds]
            mask = audio_mask[seeds].unsqueeze(-1).float()
            proj_a = self.audio_proj(a)
            gate = torch.sigmoid(self.audio_gate(torch.cat((x, proj_a), dim=-1))) * mask
            x = F.normalize(x + gate * proj_a, dim=-1)

        # Keep the graph's learned geometry; learn small residuals rather than
        # randomly rotating all music vectors before the confidence/radius gate.
        keys = F.normalize(x + .1 * self.key(x), dim=-1)
        centers = F.normalize(self.queries, dim=-1)
        temperature = self.temperature.exp().clamp(.03, .3)
        for _ in range(3):
            assignments = ((keys @ centers.T) / temperature).softmax(-1)
            centers = F.normalize(assignments.T @ keys + .1 * self.queries, dim=-1)
        mass = assignments.mean(0)
        heads = F.normalize(assignments.T @ (x + .1 * self.value(x)), dim=-1)
        return heads, mass, assignments


    @staticmethod
    def regularization(heads, masses, assignments):
        # Discourage both identical vectors and all seeds using a single head.
        gram = heads @ heads.T
        mask = ~torch.eye(len(heads), dtype=torch.bool, device=heads.device)
        separation = gram[mask].square().mean()
        balance = (masses * (masses.clamp_min(1e-8).log() + math.log(len(masses)))).sum()
        entropy = -(assignments * assignments.clamp_min(1e-8).log()).sum(-1).mean()
        return .02 * separation + .01 * balance + .005 * entropy


class NeuralRanker(nn.Module):
    """Learned fusion of graph geometry and evidence; no fixed score mixture."""
    def __init__(self, dim=DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 4 + 7, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, candidates, heads, evidence):
        # Signed log compression changes units, not musical preference weights.
        evidence = evidence.sign() * torch.log1p(evidence.abs())
        dot = (candidates * heads).sum(-1, keepdim=True)
        features = torch.cat((candidates, heads, candidates * heads,
                              (candidates - heads).abs(), dot, evidence), dim=-1)
        return self.net(features).squeeze(-1)


class MultimodalRanker(nn.Module):
    """Deep ranker fusing candidate graph representation, audio representation, and evidence."""
    def __init__(self, dim=DIM):
        super().__init__()
        # graph candidate (dim) + graph head (dim) + elem product (dim) + dot (1)
        # audio candidate (dim) + audio head (dim) + audio dot (1) + audio mask (1)
        # evidence features (6)
        in_dim = dim * 5 + 9
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, cand_graph, cand_audio, cand_mask, head_graph, head_audio, evidence):
        dot_graph = (cand_graph * head_graph).sum(-1, keepdim=True)
        masked_audio = cand_audio * cand_mask
        dot_audio = (masked_audio * head_audio).sum(-1, keepdim=True)
        features = torch.cat((
            cand_graph, head_graph, cand_graph * head_graph, dot_graph,
            masked_audio, head_audio, dot_audio, cand_mask,
            evidence
        ), dim=-1)
        return self.net(features).squeeze(-1)
