import json


__all__ = [
    "serialize_json",
    "deserialize_json",
]


serialize_json = lambda obj: json.dumps(
    obj,
    ensure_ascii = False,
)
deserialize_json = json.loads