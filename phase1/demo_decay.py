"""
demo_decay.py
=============
Demonstrates the temporal decay mechanism of the TKG by simulating two edges
with the same initial confidence but different timestamps (current vs. 48 hours ago).
Provides empirical evidence for the claim that the graph "automatically prioritizes
newer threat intelligence".
"""

import time
import math
from tkg_store import TKGEdge

def run_decay_demo():
    print("=" * 60)
    print("  Phase 1 UQ Temporal Decay Demo")
    print("=" * 60)
    print("Claim: Graph automatically prioritizes newer threat intelligence.\n")
    
    t_now = time.time()
    # 48 hours ago
    t_past = t_now - (48 * 3600)
    
    # Create an edge reported right now
    edge_new = TKGEdge(
        edge_id="edge-1-new",
        source_id="APT::APT29",
        target_id="MAL::CobaltStrike",
        relation="uses",
        c_nlp=0.92,
        threshold_weight=1.0,
        source_name="tire",
        timestamp=t_now
    )
    
    # Create an identical edge reported 48 hours ago
    edge_old = TKGEdge(
        edge_id="edge-2-old",
        source_id="APT::APT29",
        target_id="MAL::CobaltStrike",
        relation="uses",
        c_nlp=0.92,
        threshold_weight=1.0,
        source_name="tire",
        timestamp=t_past
    )
    
    # Compute metrics (simulating what DualLayerTKG does upon flush)
    edge_new.compute_metrics()
    edge_old.compute_metrics()
    
    print(f"Base NLP Confidence (c_NLP) for both edges: 0.92")
    print(f"Decay rate (lambda): 0.01 per hour\n")
    
    print("Edge 1 (Reported just now):")
    print(f"  Timestamp offset: 0 seconds")
    print(f"  Effective Confidence: {edge_new.effective_confidence:.4f}")
    print(f"  Threat Reliability  : {edge_new.threat_reliability:.4f}")
    
    print("\nEdge 2 (Reported 48 hours ago):")
    print(f"  Timestamp offset: {48 * 3600} seconds")
    print(f"  Effective Confidence: {edge_old.effective_confidence:.4f}")
    print(f"  Threat Reliability  : {edge_old.threat_reliability:.4f}")
    
    print("\nConclusion:")
    if edge_new.effective_confidence > edge_old.effective_confidence:
        print("  ✓ SUCCESS: The older edge decayed correctly, proving that the TKG")
        print("             computation penalizes stale information while maintaining")
        print("             high confidence for recent intelligence.")
    else:
        print("  ✗ FAILURE: Decay did not apply.")
        
if __name__ == "__main__":
    run_decay_demo()
