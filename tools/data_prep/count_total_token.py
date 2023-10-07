import json

def get_total_samples_from_index(file_path: str) -> int:
    """指定されたindex.jsonファイルから全サンプル数を取得する"""
    with open(file_path, "r") as file:
        data = json.load(file)
    return sum(shard["samples"] for shard in data["shards"])

# merge後のindex.jsonにする
file_path = "index.json"
total_samples = get_total_samples_from_index(file_path)

print(f"Total number of samples: {total_samples}")

