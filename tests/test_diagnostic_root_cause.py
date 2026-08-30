from daguandan_bridge.diagnostic_root_cause import diagnose_root_cause


def test_resource_hash_mismatch_is_proven_before_matcher_guess():
    report = diagnose_root_cause(
        support_verified=True,
        opening_evidence={
            "resource_identity": {"sha256": "remote"},
            "frames": [{"capture": {"backend": "printwindow"}}],
        },
        outcomes=[],
        truth=None,
        local_resource_identity={"sha256": "local"},
        support_build_id="same",
        runner_build_id="same",
    )

    assert report["primary_layer"] == "resource_hash"
    assert report["confidence"] == "PROVEN"


def test_truth_candidate_below_threshold_is_high_confidence_matcher_cause():
    report = diagnose_root_cause(
        support_verified=True,
        opening_evidence={
            "resource_identity": {"sha256": "same"},
            "frames": [{"capture": {"backend": "printwindow"}}],
        },
        outcomes=[
            {
                "output_fingerprint": "one",
                "candidate_vector": [
                    {"field": "level_rank", "label": "7", "score": 0.55, "threshold": 0.60},
                    {"field": "level_rank", "label": "2", "score": 0.72, "threshold": 0.60},
                ],
            }
        ],
        truth={"expected_level": "7"},
        local_resource_identity={"sha256": "same"},
        support_build_id="same",
        runner_build_id="same",
    )

    assert report["primary_layer"] == "matcher_threshold_margin"
    assert report["confidence"] == "HIGH_CONFIDENCE"
