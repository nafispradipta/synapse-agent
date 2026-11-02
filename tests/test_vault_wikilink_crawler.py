"""
Unit tests for Bidirectional Vault Wikilink Crawler in Synapse Retriever.
Verifies:
1. Physical markdown [[wikilinks]] scanning and parsing (aliases, headings, extensions).
2. Lightweight mtime-based in-memory caching to avoid redundant disk I/O.
3. 1-hop Spreading Activation expansion via forward wikilinks (A -> B).
4. Bidirectional backlink expansion (B <- A) when querying referenced note.
5. Proportional relevance decay weighting (hop_decay).
6. Disabling crawling via crawl_vault_wikilinks=False or max_hops=0.
"""

import time
from pathlib import Path
import pytest

from synapse.storage import SynapseStorage
from synapse.retriever import (
    SynapseRetriever,
    VaultWikilinkCache,
    extract_wikilinks_from_markdown,
)


@pytest.fixture
def temp_vault(tmp_path):
    """Fixture providing a temporary vault directory with markdown files."""
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir(parents=True, exist_ok=True)
    return vault_dir


@pytest.fixture
def storage():
    """In-memory SynapseStorage."""
    s = SynapseStorage(db_path=":memory:")
    yield s
    s.close()


def test_extract_wikilinks_from_markdown():
    text = """
    # System Guide
    List of related modules:
    - [[sop-deploy-k8s]]
    - [[database-architecture|Database Specification]]
    - [[infra/network-topology#firewall-rules]]
    - [[observability-grafana.md]]
    - [[sop-deploy-k8s]] (duplicate should be ignored)
    """
    links = extract_wikilinks_from_markdown(text)
    assert links == [
        "sop-deploy-k8s",
        "database-architecture",
        "infra/network-topology",
        "observability-grafana",
    ]


def test_vault_wikilink_cache_mtime_and_hit_tracking(temp_vault):
    cache = VaultWikilinkCache()
    doc_path = temp_vault / "note1.md"
    doc_path.write_text("# Note 1\nReferences [[target-alpha]] and [[target-beta]].\n", encoding="utf-8")

    # Initial scan: cache miss
    links_1 = cache.get_wikilinks(doc_path)
    assert links_1 == ["target-alpha", "target-beta"]
    stats_1 = cache.get_stats()
    assert stats_1["hits"] == 0
    assert stats_1["misses"] == 1
    assert stats_1["entries"] == 1

    # Second call: cache hit (no disk read)
    links_2 = cache.get_wikilinks(doc_path)
    assert links_2 == ["target-alpha", "target-beta"]
    stats_2 = cache.get_stats()
    assert stats_2["hits"] == 1
    assert stats_2["misses"] == 1

    # Modify file: cache should invalidate on mtime change
    time.sleep(0.05)  # ensure mtime timestamp differs
    doc_path.write_text("# Note 1 Modified\nReferences [[target-gamma]].\n", encoding="utf-8")

    links_3 = cache.get_wikilinks(doc_path)
    assert links_3 == ["target-gamma"]
    stats_3 = cache.get_stats()
    assert stats_3["hits"] == 1
    assert stats_3["misses"] == 2


def test_forward_wikilink_expansion_1_hop(storage, temp_vault):
    """
    When Note A (Lexical match) references [[note_b]], Spreading Activation
    must expand to Note B even if Note B has NO lexical match and NO graph edges.
    """
    doc_a = temp_vault / "sop-deploy-kubernetes.md"
    doc_b = temp_vault / "talos-cluster-architecture.md"

    doc_a.write_text(
        "# SOP Deploy Kubernetes\n"
        "Deployment guide for cluster pods and ingress controllers.\n"
        "Node architecture references [[talos-cluster-architecture]].\n",
        encoding="utf-8",
    )
    doc_b.write_text(
        "# Talos Cluster Architecture\n"
        "Immutable kernel configuration and OS security baseline.\n",
        encoding="utf-8",
    )

    node_a_id = storage.add_node(
        uri="synapse://sop/sop-deploy-kubernetes",
        l0_abstract="SOP Deploy Kubernetes cluster with ingress controller.",
        l1_overview="Deployment manifest, helm charts, and rollback protocols.",
        l2_target_path=str(doc_a.resolve()),
        category="sop",
        importance=0.8,
        strength=1.0,
        status="active",
    )

    node_b_id = storage.add_node(
        uri="synapse://architecture/talos-cluster-architecture",
        l0_abstract="Immutable OS Talos Linux Architecture.",
        l1_overview="Kernel security hardening, no-ssh policy, and declarative specs.",
        l2_target_path=str(doc_b.resolve()),
        category="architecture",
        importance=0.7,
        strength=1.0,
        status="active",
    )

    # Ensure NO edges in synapse_edges
    assert storage.get_edges(node_a_id) == []
    assert storage.get_edges(node_b_id) == []

    retriever = SynapseRetriever(storage=storage, vault_path=temp_vault, wikilink_weight=0.8)

    # Query strictly targets Note A
    results = retriever.retrieve("SOP Deploy Kubernetes", limit=5, max_hops=1, crawl_vault_wikilinks=True)
    result_ids = [r["id"] for r in results]

    # Node A is primary lexical seed
    assert node_a_id in result_ids
    top = results[0]
    assert top["id"] == node_a_id
    assert top["bm25_score"] == 1.0

    # Node B MUST be activated via 1-hop wikilink expansion!
    assert node_b_id in result_ids
    node_b_res = next(r for r in results if r["id"] == node_b_id)
    assert node_b_res["activation_energy"] > 0.0
    assert node_b_res["bm25_score"] == 0.0
    assert node_b_res["composite_score"] > 0.0


def test_backward_wikilink_expansion_backlink(storage, temp_vault):
    """
    Bidirectional crawler test:
    Note A links to Note B via [[note_b]]. Note B has NO explicit link to Note A.
    When querying Note B, Note A should be activated via reverse backlink!
    """
    doc_a = temp_vault / "sop-backup-postgres.md"
    doc_b = temp_vault / "s3-storage-architecture.md"

    doc_a.write_text(
        "# SOP Backup PostgreSQL\n"
        "Daily pg_dump execution and snapshot shipping to [[s3-storage-architecture]].\n",
        encoding="utf-8",
    )
    doc_b.write_text(
        "# S3 Storage Architecture\n"
        "Distributed cold storage with AES-256 encryption.\n",
        encoding="utf-8",
    )

    node_a_id = storage.add_node(
        uri="synapse://sop/sop-backup-postgres",
        l0_abstract="SOP Backup PostgreSQL to Cloud Storage.",
        l1_overview="pg_dump automated script, wal archiving, and encryption verification.",
        l2_target_path=str(doc_a.resolve()),
        category="sop",
        importance=0.8,
        status="active",
    )

    node_b_id = storage.add_node(
        uri="synapse://architecture/arsitektur-storage-s3",
        l0_abstract="Arsitektur Storage S3 Cloud MinIO.",
        l1_overview="Bucket lifecycle policies, IAM access key rotation, and replication.",
        l2_target_path=str(doc_b.resolve()),
        category="architecture",
        importance=0.7,
        status="active",
    )

    retriever = SynapseRetriever(storage=storage, vault_path=temp_vault, wikilink_weight=0.8)

    # Query strictly targets Note B
    results = retriever.retrieve("Arsitektur Storage S3 MinIO", limit=5, max_hops=1, crawl_vault_wikilinks=True)
    result_ids = [r["id"] for r in results]

    # Node B is primary seed
    assert node_b_id in result_ids
    assert results[0]["id"] == node_b_id

    # Node A MUST be activated via reverse backlink expansion!
    assert node_a_id in result_ids
    node_a_res = next(r for r in results if r["id"] == node_a_id)
    assert node_a_res["activation_energy"] > 0.0


def test_crawl_vault_wikilinks_disabled_flag(storage, temp_vault):
    """When crawl_vault_wikilinks=False, unlinked neighbor is NOT retrieved."""
    doc_a = temp_vault / "note_x.md"
    doc_b = temp_vault / "note_y.md"

    doc_a.write_text("UniqueTopicAlpha mentions [[note_y]].", encoding="utf-8")
    doc_b.write_text("UniqueTopicBeta isolated content.", encoding="utf-8")

    n_a = storage.add_node(
        uri="vault:note_x",
        l0_abstract="UniqueTopicAlpha abstract.",
        l2_target_path=str(doc_a.resolve()),
        category="general",
        status="active",
    )
    n_b = storage.add_node(
        uri="vault:note_y",
        l0_abstract="UniqueTopicBeta abstract.",
        l2_target_path=str(doc_b.resolve()),
        category="general",
        status="active",
    )

    retriever = SynapseRetriever(storage=storage, vault_path=temp_vault)

    # Disabled crawling
    results = retriever.retrieve("UniqueTopicAlpha", limit=5, max_hops=1, crawl_vault_wikilinks=False)
    ids = [r["id"] for r in results]

    assert n_a in ids
    assert n_b not in ids


def test_max_hops_zero_skips_spreading_activation(storage, temp_vault):
    """When max_hops=0, no hop traversal happens regardless of wikilinks."""
    doc_a = temp_vault / "doc_alpha.md"
    doc_b = temp_vault / "doc_beta.md"

    doc_a.write_text("OmegaKeyword refers to [[doc_beta]].", encoding="utf-8")
    doc_b.write_text("ZetaKeyword isolated detail.", encoding="utf-8")

    n_a = storage.add_node(
        uri="vault:doc_alpha",
        l0_abstract="OmegaKeyword abstract.",
        l2_target_path=str(doc_a.resolve()),
        status="active",
    )
    n_b = storage.add_node(
        uri="vault:doc_beta",
        l0_abstract="ZetaKeyword abstract.",
        l2_target_path=str(doc_b.resolve()),
        status="active",
    )

    retriever = SynapseRetriever(storage=storage, vault_path=temp_vault)
    results = retriever.retrieve("OmegaKeyword", limit=5, max_hops=0, crawl_vault_wikilinks=True)
    ids = [r["id"] for r in results]

    assert n_a in ids
    assert n_b not in ids


def test_hop_decay_proportional_weighting(storage, temp_vault):
    """Higher hop_decay produces higher propagated activation_energy."""
    doc_a = temp_vault / "parent.md"
    doc_b = temp_vault / "child.md"

    doc_a.write_text("SearchParentTopic links to [[child]].", encoding="utf-8")
    doc_b.write_text("ChildDetailTopic content.", encoding="utf-8")

    n_a = storage.add_node(
        uri="vault:parent",
        l0_abstract="SearchParentTopic info.",
        l2_target_path=str(doc_a.resolve()),
        status="active",
    )
    n_b = storage.add_node(
        uri="vault:child",
        l0_abstract="ChildDetailTopic info.",
        l2_target_path=str(doc_b.resolve()),
        status="active",
    )

    retriever = SynapseRetriever(storage=storage, vault_path=temp_vault, wikilink_weight=0.8)

    # High decay (0.90)
    res_high = retriever.retrieve("SearchParentTopic", limit=5, max_hops=1, hop_decay=0.90)
    # Low decay (0.20)
    res_low = retriever.retrieve("SearchParentTopic", limit=5, max_hops=1, hop_decay=0.20)

    n_b_high = next(r for r in res_high if r["id"] == n_b)
    n_b_low = next(r for r in res_low if r["id"] == n_b)

    assert n_b_high["activation_energy"] > n_b_low["activation_energy"]


def test_get_vault_wikilink_neighbors_api(storage, temp_vault):
    """Verify retriever.get_vault_wikilink_neighbors direct API directions."""
    doc_1 = temp_vault / "guide-one.md"
    doc_2 = temp_vault / "guide-two.md"

    doc_1.write_text("Guide One references [[guide-two|Custom Label]].", encoding="utf-8")
    doc_2.write_text("Guide Two terminal target.", encoding="utf-8")

    n1 = storage.add_node(
        uri="vault:guide-one",
        l0_abstract="Guide One",
        l2_target_path=str(doc_1.resolve()),
        status="active",
    )
    n2 = storage.add_node(
        uri="vault:guide-two",
        l0_abstract="Guide Two",
        l2_target_path=str(doc_2.resolve()),
        status="active",
    )

    retriever = SynapseRetriever(storage=storage, vault_path=temp_vault)

    # Outgoing from Node 1 -> Node 2
    out_1 = retriever.get_vault_wikilink_neighbors(n1, direction="out")
    assert len(out_1) == 1
    assert out_1[0]["id"] == n2

    # Incoming to Node 1 -> None
    in_1 = retriever.get_vault_wikilink_neighbors(n1, direction="in")
    assert len(in_1) == 0

    # Incoming to Node 2 -> Node 1 (Backlink)
    in_2 = retriever.get_vault_wikilink_neighbors(n2, direction="in")
    assert len(in_2) == 1
    assert in_2[0]["id"] == n1

    # Both for Node 2 -> Node 1
    both_2 = retriever.get_vault_wikilink_neighbors(n2, direction="both")
    assert len(both_2) == 1
    assert both_2[0]["id"] == n1


def test_disk_io_lightweight_caching_on_repeated_retrievals(storage, temp_vault):
    """Subsequent retrieve() calls use cached wikilinks without repeated disk reads."""
    doc_1 = temp_vault / "cached_hub.md"
    doc_2 = temp_vault / "cached_leaf.md"

    doc_1.write_text("FrequentQueryTarget links to [[cached_leaf]].", encoding="utf-8")
    doc_2.write_text("CachedLeaf content.", encoding="utf-8")

    storage.add_node(
        uri="vault:cached_hub",
        l0_abstract="FrequentQueryTarget overview.",
        l2_target_path=str(doc_1.resolve()),
        status="active",
    )
    storage.add_node(
        uri="vault:cached_leaf",
        l0_abstract="CachedLeaf overview.",
        l2_target_path=str(doc_2.resolve()),
        status="active",
    )

    retriever = SynapseRetriever(storage=storage, vault_path=temp_vault)

    # First retrieval: initial cold file reads
    retriever.retrieve("FrequentQueryTarget", limit=5, max_hops=1)
    stats_cold = retriever.vault_cache.get_stats()
    entries_count = stats_cold["entries"]
    assert entries_count >= 2

    # Run 5 more queries
    for _ in range(5):
        retriever.retrieve("FrequentQueryTarget", limit=5, max_hops=1)

    stats_warm = retriever.vault_cache.get_stats()
    # Cache hits should have increased significantly, while entries stay constant
    assert stats_warm["hits"] > stats_cold["hits"]
    assert stats_warm["entries"] == entries_count
