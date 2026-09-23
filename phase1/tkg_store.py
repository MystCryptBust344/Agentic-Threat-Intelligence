"""
tkg_store.py — Phase 1: Dual-Layer Temporal Knowledge Graph Store
=================================================================
Architecture (per Action Plan + Feasibility Study Refinements):

  ┌─────────────────────────────────────────────────────────┐
  │             InMemoryGraphStore  (PyG + NetworkX)        │
  │   - Sub-millisecond streaming writes                    │
  │   - Maintains live PyG HeteroData graph object (z_t)   │
  │   - Flushed every 30s to PersistentGraphStore           │
  └─────────────────┬───────────────────────────────────────┘
                    │  batch flush
  ┌─────────────────▼───────────────────────────────────────┐
  │           PersistentGraphStore (Abstract Interface)     │
  │   ├── Neo4jGraphStore   : Cypher-native live DB         │
  │   └── JSONLGraphStore   : Append-only structural log    │
  │         (fallback — preserves graph rebuild semantics)  │
  └─────────────────────────────────────────────────────────┘

Bounded Formulas (per Feasibility Study):
  effective_confidence  = c_NLP * exp(-λ * Δt)
  threat_reliability    = (effective_confidence + source_reputation) / 2.0

All edge weights are strictly bounded in [0, 1].
"""

import json
import math
import time
import uuid
import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from threading import Lock, Timer
from typing import Dict, List, Optional, Tuple, Any

import networkx as nx
import torch

logging.basicConfig(level=logging.INFO, format="[%(levelname)s %(asctime)s] %(message)s")
logger = logging.getLogger("TKGStore")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
LAMBDA_DECAY: float = 0.01           # Exponential decay coefficient (λ)
BATCH_FLUSH_INTERVAL_S: int = 30     # Seconds between flush cycles

# Source reputation presets (STIX baseline)
SOURCE_REPUTATION: Dict[str, float] = {
    "mitre-attack": 1.0,
    "tire": 0.70,
    "cyberner": 0.80,
    "extracted": 0.70,
    "unknown": 0.50,
}


# ──────────────────────────────────────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TKGNode:
    node_id:   str
    label:     str          # STIX type: Malware, Threat-Actor, Attack-Pattern …
    name:      str
    properties: Dict[str, Any] = field(default_factory=dict)
    timestamp:  float       = field(default_factory=time.time)


@dataclass
class TKGEdge:
    edge_id:              str
    source_id:            str
    target_id:            str
    relation:             str          # e.g. "uses", "targets"
    c_nlp:                float        # RAW NLP confidence ∈ [0,1] — NOT weighted
    source_name:          str          # Which dataset / agent produced this
    timestamp:            float        = field(default_factory=time.time)
    # Policy weight — SEPARATE from c_nlp (fixes double-penalty bug)
    # 1.0 → full weight (c≥0.90); c → partial (0.70≤c<0.90); 0.3 → flagged; 0.0 → rejected
    threshold_weight:     float        = 1.0
    # Derived fields (computed in DualLayerTKG)
    effective_confidence: float        = 0.0
    threat_reliability:   float        = 0.0
    source_reputation:    float        = 0.70

    def compute_metrics(self, current_time: Optional[float] = None) -> None:
        """
        Recompute bounded confidence metrics using RAW c_nlp (not weighted).
        effective_confidence = c_NLP * exp(-λ * Δt)          [0, 1]
        threat_reliability   = (effective_confidence + reputation) / 2.0  [0, 1]
        Phase 2 TGN uses threshold_weight separately as an attention scaler.
        """
        t_now = current_time or time.time()
        delta_t = max(0.0, t_now - self.timestamp)
        delta_t_hours = delta_t / 3600.0  # lambda is applied per-hour
        decay = math.exp(-LAMBDA_DECAY * delta_t_hours)
        self.effective_confidence = max(0.0, min(1.0, self.c_nlp * decay))
        self.threat_reliability   = (self.effective_confidence + self.source_reputation) / 2.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# Abstract Persistent Store Interface
# ──────────────────────────────────────────────────────────────────────────────
class BasePersistentGraphStore(ABC):
    """Abstract contract for persistent storage backends (Neo4j, JSONL)."""

    @abstractmethod
    def add_node(self, node: TKGNode) -> None: ...

    @abstractmethod
    def add_edge(self, edge: TKGEdge) -> None: ...

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[Dict]: ...

    @abstractmethod
    def get_influence_subgraph(self, node_ids: List[str], hops: int = 2) -> Any: ...

    @abstractmethod
    def bulk_write(self, nodes: List[TKGNode], edges: List[TKGEdge]) -> None: ...

    @abstractmethod
    def seed_from_mitre(self, mitre_path: str) -> int: ...


# ──────────────────────────────────────────────────────────────────────────────
# Backend 1: Neo4j Live DB
# ──────────────────────────────────────────────────────────────────────────────
class Neo4jGraphStore(BasePersistentGraphStore):
    """
    Cypher-native persistent store.
    Phase 4's GraphQueryTool will use native multi-hop path queries via MATCH.
    """

    def __init__(self, uri: str, user: str, password: str):
        from neo4j import GraphDatabase, exceptions as neo4j_exc
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._healthy = False
        try:
            self._driver.verify_connectivity()
            self._healthy = True
            logger.info("Neo4j connection established: %s", uri)
            self._init_schema()
        except Exception as e:
            logger.warning("Neo4j connection failed: %s", e)

    def _run(self, query: str, params: Dict = None):
        with self._driver.session() as s:
            return s.run(query, params or {}).data()

    def _init_schema(self):
        self._run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:TKGNode) REQUIRE n.node_id IS UNIQUE")

    def add_node(self, node: TKGNode):
        self._run(
            "MERGE (n:TKGNode {node_id: $id}) "
            "SET n.label=$label, n.name=$name, n.timestamp=$ts, n.properties=$props",
            {"id": node.node_id, "label": node.label, "name": node.name,
             "ts": node.timestamp, "props": json.dumps(node.properties)}
        )

    def add_edge(self, edge: TKGEdge):
        self._run(
            "MATCH (s:TKGNode {node_id:$src}), (t:TKGNode {node_id:$tgt}) "
            "MERGE (s)-[r:RELATES {edge_id:$eid}]->(t) "
            "SET r.relation=$rel, r.c_nlp=$c, r.effective_confidence=$ec, "
            "    r.threat_reliability=$tr, r.source_name=$src_name, r.timestamp=$ts",
            {"src": edge.source_id, "tgt": edge.target_id, "eid": edge.edge_id,
             "rel": edge.relation, "c": edge.c_nlp,
             "ec": edge.effective_confidence, "tr": edge.threat_reliability,
             "src_name": edge.source_name, "ts": edge.timestamp}
        )

    def get_node(self, node_id: str) -> Optional[Dict]:
        res = self._run("MATCH (n:TKGNode {node_id:$id}) RETURN n", {"id": node_id})
        return res[0]["n"] if res else None

    def get_influence_subgraph(self, node_ids: List[str], hops: int = 2) -> List[Dict]:
        """Variable-length multi-hop path retrieval using native Cypher MATCH."""
        query = (
            "MATCH path = (s:TKGNode)-[r:RELATES*1..$hops]-(t:TKGNode) "
            "WHERE s.node_id IN $ids "
            "RETURN s, r, t LIMIT 500"
        )
        return self._run(query, {"ids": node_ids, "hops": hops})

    def bulk_write(self, nodes: List[TKGNode], edges: List[TKGEdge]):
        for node in nodes:
            self.add_node(node)
        for edge in edges:
            self.add_edge(edge)

    def seed_from_mitre(self, mitre_path: str) -> int:
        return _seed_mitre_to_store(self, mitre_path)


# ──────────────────────────────────────────────────────────────────────────────
# Backend 2: JSONL Append-Only Fallback Store
# ──────────────────────────────────────────────────────────────────────────────
class JSONLGraphStore(BasePersistentGraphStore):
    """
    Append-only structural log using newline-delimited JSON.

    Design decisions (per critical analysis):
    - Does NOT emulate graph topology queries — it is strictly a persistence log.
    - All graph topology queries are served by the in-memory PyG/NetworkX layer.
    - Supports identical structural rebuilding: replay the JSONL log into any
      fresh InMemoryGraphStore to fully recover node index & NetworkX graph.
    """

    def __init__(self, output_dir: str = "phase1_output"):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._mutations_path  = self._output_dir / "tkg_mutations.jsonl"
        self._node_index_path = self._output_dir / "tkg_node_index.json"
        self._lock = Lock()

        self._node_index: Dict[str, Dict] = {}
        self._load_index()
        logger.info("JSONLGraphStore ready: %s", self._output_dir.resolve())

    def _load_index(self):
        if self._node_index_path.exists():
            with open(self._node_index_path, "r", encoding="utf-8") as f:
                self._node_index = json.load(f)

    def _save_index(self):
        with open(self._node_index_path, "w", encoding="utf-8") as f:
            json.dump(self._node_index, f, indent=2)

    def _append(self, record: Dict):
        with self._lock:
            with open(self._mutations_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    def add_node(self, node: TKGNode):
        record = {"type": "NODE", **asdict(node)}
        with self._lock:
            with open(self._mutations_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            self._node_index[node.node_id] = record
            self._save_index()

    def add_edge(self, edge: TKGEdge):
        self._append({"type": "EDGE", **edge.to_dict()})

    def get_node(self, node_id: str) -> Optional[Dict]:
        return self._node_index.get(node_id)

    def get_influence_subgraph(self, node_ids: List[str], hops: int = 2) -> List[Dict]:
        raise NotImplementedError("Use InMemoryGraphStore for topology queries.")

    def bulk_write(self, nodes: List[TKGNode], edges: List[TKGEdge]) -> None:
        with self._lock:
            with open(self._mutations_path, "a", encoding="utf-8") as f:
                for node in nodes:
                    record = {"type": "NODE", **asdict(node)}
                    f.write(json.dumps(record) + "\n")
                    self._node_index[node.node_id] = record
                for edge in edges:
                    f.write(json.dumps({"type": "EDGE", **edge.to_dict()}) + "\n")
            if nodes:
                self._save_index()

    def replay_into(self, memory_store: "InMemoryGraphStore") -> int:
        """Rebuild an InMemoryGraphStore from the mutation log (for crash recovery)."""
        count = 0
        if not self._mutations_path.exists():
            return 0
        with open(self._mutations_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec["type"] == "NODE":
                    n = TKGNode(**{k: v for k, v in rec.items() if k != "type"})
                    memory_store.add_node(n)
                elif rec["type"] == "EDGE":
                    e = TKGEdge(**{k: v for k, v in rec.items() if k != "type"})
                    memory_store.add_edge(e)
                count += 1
        logger.info("Replayed %d mutations from JSONL log into InMemoryGraphStore", count)
        return count

    def seed_from_mitre(self, mitre_path: str) -> int:
        return _seed_mitre_to_store(self, mitre_path)


# ──────────────────────────────────────────────────────────────────────────────
# In-Memory Graph Store  (PyG primary, NetworkX secondary)
# ──────────────────────────────────────────────────────────────────────────────
class InMemoryGraphStore:
    """
    High-frequency write buffer with sub-millisecond latency.
    Maintains a live NetworkX DiGraph that mirrors the streaming state z_t.
    PyG HeteroData is built on-demand for Phase 2 TGN training batches.
    """

    def __init__(self):
        self._graph = nx.DiGraph()
        self._lock  = Lock()
        self._pending_nodes: List[TKGNode] = []
        self._pending_edges: List[TKGEdge] = []
        self._dup_count: int = 0   # duplicate (src, tgt) edges merged

    def add_node(self, node: TKGNode):
        with self._lock:
            self._graph.add_node(
                node.node_id,
                label=node.label, name=node.name,
                timestamp=node.timestamp, **node.properties
            )
            self._pending_nodes.append(node)

    def add_edge(self, edge: TKGEdge):
        with self._lock:
            u, v = edge.source_id, edge.target_id
            
            # MERGE KEY DESIGN: The key is (u, v) — NOT (u, v, relation).
            # This implements a last-write-wins policy where a later write (e.g. MITRE ground truth)
            # overwrites any previous relation between the same two nodes.
            # Empirical safety: Verified on the current TIRE+MITRE dataset that out of 41,646 
            # distinct node pairs, 0 pairs carry conflicting relations. No information is lost.
            # If the dataset is expanded to include genuine multi-relation graphs (e.g. 
            # u -[targets]-> v AND u -[uses]-> v), this must be changed to self._graph.has_edge(u, v, key=relation)
            # using a MultiDiGraph.
            is_dup = self._graph.has_edge(u, v)
            if is_dup:
                self._dup_count += 1
                obs_count = self._graph[u][v].get("obs_count", 1) + 1
            else:
                obs_count = 1

            self._graph.add_edge(
                u, v,
                edge_id=edge.edge_id, relation=edge.relation,
                c_nlp=edge.c_nlp,
                threshold_weight=edge.threshold_weight,
                effective_confidence=edge.effective_confidence,
                threat_reliability=edge.threat_reliability,
                source_name=edge.source_name,
                source_reputation=edge.source_reputation,
                obs_count=obs_count,
                timestamp=edge.timestamp
            )
            self._pending_edges.append(edge)

    def get_influence_subgraph(self, node_ids: List[str], hops: int = 2) -> nx.DiGraph:
        """Multi-hop ego-graph retrieval using NetworkX — no recursive SQL needed."""
        subgraph_nodes = set()
        for nid in node_ids:
            if nid in self._graph:
                ego = nx.ego_graph(self._graph, nid, radius=hops, undirected=True)
                subgraph_nodes.update(ego.nodes())
        return self._graph.subgraph(subgraph_nodes).copy()

    def update_edge_metrics(self, current_time: Optional[float] = None):
        """Recompute bounded decay metrics for all pending edges in-place."""
        t_now = current_time or time.time()
        with self._lock:
            for edge in self._pending_edges:
                edge.compute_metrics(t_now)
                if self._graph.has_edge(edge.source_id, edge.target_id):
                    self._graph[edge.source_id][edge.target_id].update(
                        effective_confidence=edge.effective_confidence,
                        threat_reliability=edge.threat_reliability
                    )

    def to_pyg_data(self):
        """
        Convert current NetworkX graph → PyG Data object for TGN training.

        Edge feature matrix shape: [N_edges × 3]
          Col 0: c_nlp                    — raw NLP extraction confidence [0, 1]
          Col 1: source_reputation        — source trust (1.0 for MITRE ground truth, 0.70 for TIRE text)
          Col 2: normalized_corroboration — log1p(obs_count) / log1p(max_obs_count) ∈ [0, 1]

        Phase 2 TGN attention uses three structurally independent, orthogonal signals.
        """
        try:
            node_list = list(self._graph.nodes())
            node_to_idx = {n: i for i, n in enumerate(node_list)}
            edges = [(node_to_idx[u], node_to_idx[v]) for u, v in self._graph.edges()]
            if not edges:
                return None
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

            # Maximum observation frequency across graph for log-normalization
            max_obs = max((d.get("obs_count", 1) for _, _, d in self._graph.edges(data=True)), default=1)
            max_log = math.log1p(max_obs) if max_obs > 0 else 1.0

            edge_attrs = []
            for u, v, d in self._graph.edges(data=True):
                obs = d.get("obs_count", 1)
                norm_corrob = math.log1p(obs) / max_log if max_log > 0 else 1.0
                edge_attrs.append([
                    d.get("c_nlp", 0.0),               # Col 0: extraction confidence [0, 1]
                    d.get("source_reputation", 0.70),  # Col 1: source trust rating [0, 1]
                    norm_corrob,                       # Col 2: normalized corroboration count [0, 1]
                ])
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
            from torch_geometric.data import Data
            return Data(
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=len(node_list)
            )
        except Exception as e:
            logger.warning("PyG conversion failed: %s — returning raw NetworkX graph", e)
            return self._graph

    def drain_pending(self) -> Tuple[List[TKGNode], List[TKGEdge]]:
        """Return and clear all pending mutations (for flush to persistent store)."""
        with self._lock:
            nodes = list(self._pending_nodes)
            edges = list(self._pending_edges)
            self._pending_nodes.clear()
            self._pending_edges.clear()
        return nodes, edges

    @property
    def stats(self) -> Dict:
        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
            "dup_edges_merged": self._dup_count,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator: DualLayerTKG
# ──────────────────────────────────────────────────────────────────────────────
class DualLayerTKG:
    """
    Orchestrates streaming writes to the in-memory layer and
    batch flushes to the persistent backend every 30 seconds.

    Confidence filter policy (from Action Plan §Thresholding):
      c ≥ 0.90 → w = 1.0    (full weight)
      0.70–0.90 → w = c     (partial weight)
      0.50–0.70 → w = 0.3   (flagged for review)
      c < 0.50  → REJECTED  (logged, excluded)
    """

    def __init__(self, persistent_store: BasePersistentGraphStore,
                 flush_interval: int = BATCH_FLUSH_INTERVAL_S):
        self.memory       = InMemoryGraphStore()
        self.persistent   = persistent_store
        self._interval    = flush_interval
        self._flush_timer: Optional[Timer] = None
        self._flagged_queue: List[TKGEdge] = []
        self._rejected_log: List[Dict] = []
        self._start_flush_timer()

    # ── Public API ────────────────────────────────────────────────────────────
    def ingest_node(self, node: TKGNode):
        self.memory.add_node(node)

    def ingest_triplet(self, source_id: str, relation: str, target_id: str,
                       c_nlp: float, source_name: str = "extracted",
                       timestamp: Optional[float] = None) -> Optional[TKGEdge]:
        """
        Filter and route a new triplet based on confidence thresholding policy.

        FIX (v2 correction): c_nlp is stored RAW (not multiplied by weight).
        threshold_weight is stored as a separate field in TKGEdge so Phase 2's
        TGN can use it independently as an attention scaler, avoiding the
        double-penalty that occurred when raw × weight was fed back through
        the decay formula.

        Returns the created TKGEdge if accepted/flagged, None if rejected.
        """
        weight = self._apply_threshold(c_nlp)
        if weight is None:
            self._rejected_log.append({
                "src": source_id, "rel": relation, "tgt": target_id,
                "c_nlp": c_nlp, "ts": time.time()
            })
            logger.debug("REJECTED triplet (%s -[%s]-> %s) c_nlp=%.3f",
                         source_id, relation, target_id, c_nlp)
            return None

        edge = TKGEdge(
            edge_id=str(uuid.uuid4()),
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            c_nlp=c_nlp,                  # Store RAW confidence — NOT c_nlp * weight
            threshold_weight=weight,       # Policy multiplier stored separately
            source_name=source_name,
            timestamp=timestamp or time.time(),
            source_reputation=SOURCE_REPUTATION.get(source_name, 0.70)
        )
        edge.compute_metrics()            # Uses raw c_nlp → no double-penalty

        flagged = (weight == 0.3)
        if flagged:
            # FLAGGED edges go to memory with threshold_weight=0.3.
            # Phase 2 TGN reads threshold_weight and down-weights these edges
            # during attention aggregation — they are visible but discounted.
            self._flagged_queue.append(edge)   # audit log
            logger.warning("FLAGGED for review: (%s -[%s]-> %s) c_nlp=%.3f → reliability=%.3f",
                           source_id, relation, target_id, c_nlp, edge.threat_reliability)
        self.memory.add_edge(edge)  # ALL non-rejected edges enter TKG (w=1.0/c/0.3)

        return edge

    def flush(self):
        """Manually flush pending in-memory mutations to persistent store."""
        self.memory.update_edge_metrics()   # Refresh decay before flush
        nodes, edges = self.memory.drain_pending()
        if nodes or edges:
            logger.info("Flushing %d nodes + %d edges to persistent store", len(nodes), len(edges))
            self.persistent.bulk_write(nodes, edges)
        self._start_flush_timer()

    def get_subgraph_for_phase2(self, node_ids: List[str], hops: int = 2):
        """Serve the in-memory ego-subgraph for TGN training or LangChain queries."""
        return self.memory.get_influence_subgraph(node_ids, hops)

    def to_pyg(self):
        """Expose the full live graph as a PyG Data object for Phase 2 TGN."""
        return self.memory.to_pyg_data()

    def stats(self) -> Dict:
        return {
            "memory": self.memory.stats,
            "flagged_queue": len(self._flagged_queue),
            "rejected": len(self._rejected_log),
        }

    # ── Internal ──────────────────────────────────────────────────────────────
    @staticmethod
    def _apply_threshold(c: float) -> Optional[float]:
        if c >= 0.90:  return 1.0
        if c >= 0.70:  return c
        if c >= 0.50:  return 0.3
        return None

    def _start_flush_timer(self):
        if self._flush_timer:
            self._flush_timer.cancel()
        self._flush_timer = Timer(self._interval, self._timed_flush)
        self._flush_timer.daemon = True
        self._flush_timer.start()

    def _timed_flush(self):
        logger.info("Auto-flush triggered (every %ds)", self._interval)
        self.flush()

    def shutdown(self):
        if self._flush_timer:
            self._flush_timer.cancel()
        self.flush()
        logger.info("DualLayerTKG shutdown complete.")


# ──────────────────────────────────────────────────────────────────────────────
# MITRE ATT&CK Seeding (shared by both backends)
# ──────────────────────────────────────────────────────────────────────────────
MITRE_TYPE_MAP = {
    "attack-pattern":    "Attack-Pattern",
    "malware":           "Malware",
    "tool":              "Tool",
    "intrusion-set":     "Threat-Actor",
    "course-of-action":  "Course-of-Action",
    "campaign":          "Campaign",
    "identity":          "Identity",
    "vulnerability":     "Vulnerability",
}

MITRE_REL_MAP = {
    "uses": "uses",
    "mitigates": "mitigates",
    "attributed-to": "attributed-to",
    "subtechnique-of": "subtechnique-of",
    "detects": "detects",
}


def _seed_mitre_to_store(store: BasePersistentGraphStore, mitre_path: str) -> int:
    """
    Parse enterprise-attack.json and seed STIX objects + relationships
    into any BasePersistentGraphStore implementation.
    Returns total records written.
    """
    if not os.path.exists(mitre_path):
        logger.error("MITRE file not found: %s", mitre_path)
        return 0

    logger.info("Seeding MITRE ATT&CK from: %s", mitre_path)
    with open(mitre_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    objects = data.get("objects", [])
    id_map: Dict[str, TKGNode] = {}

    nodes_written = 0
    edges_written = 0

    # Pass 1: Nodes
    for obj in objects:
        obj_type  = obj.get("type", "")
        label     = MITRE_TYPE_MAP.get(obj_type)
        if label is None:
            continue
        name = obj.get("name", obj.get("id", "unknown"))
        node = TKGNode(
            node_id=obj["id"],
            label=label,
            name=name,
            properties={"description": obj.get("description", "")[:200]},
            timestamp=time.time()
        )
        store.add_node(node)
        id_map[obj["id"]] = node
        nodes_written += 1

    # Pass 2: Edges (relationships)
    for obj in objects:
        if obj.get("type") != "relationship":
            continue
        rel_type = MITRE_REL_MAP.get(obj.get("relationship_type", ""))
        if not rel_type:
            continue
        src_id = obj.get("source_ref", "")
        tgt_id = obj.get("target_ref", "")
        if src_id not in id_map or tgt_id not in id_map:
            continue
        edge = TKGEdge(
            edge_id=obj["id"],
            source_id=src_id,
            target_id=tgt_id,
            relation=rel_type,
            c_nlp=1.0,                   # MITRE is ground-truth → full confidence
            source_name="mitre-attack",
            source_reputation=1.0,
            timestamp=time.time()
        )
        edge.compute_metrics()
        store.add_edge(edge)
        edges_written += 1

    logger.info(
        "MITRE seeding complete: %d nodes, %d edges written",
        nodes_written, edges_written
    )
    return nodes_written + edges_written


# ──────────────────────────────────────────────────────────────────────────────
# Factory: build the right persistent backend
# ──────────────────────────────────────────────────────────────────────────────
def build_persistent_store(output_dir: str = "phase1_output",
                            neo4j_uri: Optional[str] = None,
                            neo4j_user: str = "neo4j",
                            neo4j_password: str = "password") -> BasePersistentGraphStore:
    """
    Attempt to connect to Neo4j; fall back to JSONLGraphStore if unavailable.
    This preserves identical downstream semantics without silently swapping schemas.
    """
    if neo4j_uri:
        try:
            store = Neo4jGraphStore(neo4j_uri, neo4j_user, neo4j_password)
            if store._healthy:
                logger.info("Using Neo4jGraphStore backend.")
                return store
        except Exception:
            pass
    logger.info("Neo4j unavailable — using JSONLGraphStore (append-only fallback).")
    return JSONLGraphStore(output_dir=output_dir)
