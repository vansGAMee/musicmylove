from scripts.compare_models import build_evaluation_dataset


def test_local_training_dataset_is_user_disjoint_bucketed_and_leaves_test_histories_untouched():
    tracks = [{"mbid": str(index), "artist": f"Artist {index}", "title": f"Track {index}", "popularityPercentile": index / 1000}
              for index in range(1000)]
    histories = [list(range(user, user + 80)) for user in range(560)]

    dataset = build_evaluation_dataset({"tracks": tracks, "histories": histories})

    assert {row["partition"] for row in dataset["episodes"]} == {"train", "validation"}
    assert {row["user_key"] for row in dataset["episodes"] if row["partition"] == "train"}.isdisjoint(
        {row["user_key"] for row in dataset["episodes"] if row["partition"] == "validation"}
    )
    assert {len(row["seed_ids"]) for row in dataset["episodes"]} == {5, 20, 60}
    assert all(set(row["seed_ids"]).isdisjoint({row["positive_id"], row["negative_id"]}) for row in dataset["episodes"])
