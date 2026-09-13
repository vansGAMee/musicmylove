import json

from ml.cohort import choose_active_users, choose_users, count_user_listens, extract_usernames


def test_extracts_unique_usernames_and_skips_malformed_rows():
    rows = [
        json.dumps({"user_name": "alice"}).encode(),
        b"not-json",
        json.dumps({"user_name": "bob"}).encode(),
        json.dumps({"user_name": "alice"}).encode(),
        json.dumps({"user_name": ""}).encode(),
        json.dumps({"payload": {"user_name": "nested"}}).encode(),
    ]
    assert extract_usernames(rows) == ["alice", "bob"]


def test_user_selection_is_seeded_and_independent_of_input_order():
    names = [f"user-{index}" for index in range(20)]
    assert choose_users(names, 8, 73) == choose_users(list(reversed(names)), 8, 73)
    assert len(choose_users(names, 8, 73)) == 8


def test_selects_randomly_only_among_users_with_enough_incremental_activity():
    rows = [json.dumps({"user_name": name}).encode() for name, count in {"quiet": 2, "alice": 4, "bob": 3}.items() for _ in range(count)]
    counts = count_user_listens(rows)
    assert counts == {"alice": 4, "bob": 3, "quiet": 2}
    assert set(choose_active_users(counts, count=10, minimum_listens=3, seed=41)) == {"alice", "bob"}
