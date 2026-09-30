"""Inference-only discovery policy: supported neighborhoods, then neural choice.

No genre/artist blacklists or fabricated audio judgments. Limits are explicit
playlist policy, not learned quality estimates or a new training verdict.
"""
from collections import Counter
from dataclasses import dataclass
import numpy as np
from .discovery_engine import input_keys, song_key
from .honest_training import family_index
from .recommend import known_track_indices, match_key


@dataclass
class DiscoveryPool:
    tracks: list
    groups: dict
    familiar: set
    popular: set
    families: np.ndarray
    candidates: set
    unsupported: list
    popularity_threshold: float
    artist_popularity_threshold: float | None


def prepare_discovery(engine, seeds, excluded):
    tracks = engine.tracks
    families = engine.families if hasattr(engine, 'families') else family_index(tracks)
    known = known_track_indices(excluded, tracks) | set(seeds)
    keys = input_keys(excluded)
    known.update(i for i,t in enumerate(tracks) if
                 ({song_key(t["artist"],t["title"])} | {song_key(*a) for a in t.get("aliases",[])}) & keys)
    blocked = set(families[sorted(known)])
    familiar = {a for a, _ in input_keys(excluded)} | {match_key(tracks[i]['artist']) for i in known}
    embeddings = engine.embeddings.detach().cpu().numpy()
    embeddings = embeddings / np.maximum(np.linalg.norm(embeddings,axis=1,keepdims=True),1e-12)
    groups = {}
    for seed in sorted(set(seeds),key=lambda i:tracks[i]['id']):
        direction = match_key(tracks[seed]['artist'])
        group = groups.setdefault(direction,{})
        similarities = embeddings @ embeddings[seed]
        # Each seed has its own threshold, so a quieter direction isn't compared
        # against the size/popularity of another seed's audience.
        threshold = max(0.,float(np.quantile(similarities,.90)))
        row = engine.graph['global'].getrow(seed).multiply(engine.graph['confidence'].getrow(seed)).tocsr()
        supported = {}
        for candidate, affinity in zip(row.indices,row.data):
            candidate = int(candidate)
            if (affinity <= 0 or not np.isfinite(affinity) or families[candidate] in blocked
                    or similarities[candidate] < threshold):
                continue
            shared = float(engine.graph['users'][seed,candidate])
            if shared < 3:
                continue
            item = {'direction':direction,'seed_id':tracks[seed]['id'],
                    'seed_artist':tracks[seed]['artist'],'seed_title':tracks[seed]['title'],
                    'graph_affinity':float(affinity),'embedding_similarity':float(similarities[candidate]),
                    'shared_listeners':int(shared)}
            supported[candidate] = item
        # Compare like with like: strong same-artist edges must not suppress discovery.
        peaks = {}
        for i,item in supported.items():
            bucket = match_key(tracks[i]["artist"]) in familiar
            peaks[bucket] = max(peaks.get(bucket,0.),item["graph_affinity"])
        for candidate,item in supported.items():
            bucket = match_key(tracks[candidate]["artist"]) in familiar
            if item["graph_affinity"] < .25 * peaks[bucket]:
                continue
            previous = group.get(candidate)
            if previous is None or (item['graph_affinity'],item['embedding_similarity']) > (previous['graph_affinity'],previous['embedding_similarity']):
                group[candidate]=item
    pop=np.asarray(engine.pop)
    threshold=float(np.quantile(pop,.99))
    popular=set(np.flatnonzero((pop>=threshold)&(pop>np.median(pop))).tolist())
    artist_threshold = None
    if hasattr(engine,"artist_pop") and hasattr(engine,"artist_ids"):
        artist_pop = np.asarray(engine.artist_pop)
        artist_threshold = float(np.quantile(artist_pop,.995))
        popular.update(np.flatnonzero((artist_pop[engine.artist_ids]>=artist_threshold) &
                                     (artist_pop[engine.artist_ids]>np.median(artist_pop))).tolist())
    candidates={i for group in groups.values() for i in group}
    return DiscoveryPool(tracks,groups,familiar,popular,families,candidates,
                         sorted(a for a,g in groups.items() if not g),threshold,artist_threshold)


def select_discovery(pool, scores, size=50):
    """Balance seed directions; neural score chooses within each feasible group.

    Every prefix has <=30% familiar artists and <=20% catalog-popular tracks.
    Never weaken evidence/novelty requirements merely to fill 50 positions.
    """
    queues={a:sorted((i for i in group if np.isfinite(scores[i])),
                    key=lambda i:(-float(scores[i]),pool.tracks[i]['id'])) for a,group in pool.groups.items()}
    selected,explanations=[],{}
    artists,coverage=Counter(),Counter()
    used=set(); familiar_count=popular_count=0; previous_artist=None
    while len(selected)<size:
        position=len(selected)+1
        options=[]
        for direction,queue in sorted(queues.items()):
            for i in queue:
                artist=match_key(pool.tracks[i]['artist'])
                if (pool.families[i] in used or artists[artist]>=2 or artist==previous_artist
                        or familiar_count+int(artist in pool.familiar)>position*3//10
                        or popular_count+int(i in pool.popular)>position//5):
                    continue
                options.append((coverage[direction],-float(scores[i]),pool.tracks[i]['id'],direction,i))
                break
        if not options:
            break
        _,_,_,direction,i=min(options)
        selected.append(i); explanations[i]=pool.groups[direction][i]
        artist=match_key(pool.tracks[i]['artist'])
        used.add(pool.families[i]); artists[artist]+=1; coverage[direction]+=1
        familiar_count+=int(artist in pool.familiar); popular_count+=int(i in pool.popular)
        previous_artist=artist
    report={'policy':'graph-discovery-v1','candidate_count':len(pool.candidates),
            'familiar_artist_tracks':familiar_count,'catalog_popular_tracks':popular_count,
            'popularity_threshold_listeners':pool.popularity_threshold,
            'artist_popularity_threshold_listeners':pool.artist_popularity_threshold,
            'minimum_relative_graph_affinity':.25,
            'covered_directions':dict(sorted(coverage.items())),
            'unsupported_directions':pool.unsupported,
            'unrepresented_directions':sorted(set(pool.groups)-set(coverage)),
            'genre_or_audio_verified':False,'requested_size':size,'returned_size':len(selected)}
    return selected,explanations,report
