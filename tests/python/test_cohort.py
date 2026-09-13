import json

from ml.cohort import choose_users, extract_usernames


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
