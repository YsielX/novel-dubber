from pathlib import Path

from novel_dubber.config import load_config
from novel_dubber.merge import merge_asr_diarization
from novel_dubber.text_mode import add_pause_rules, parse_text_segments
from novel_dubber.utils import append_jsonl, write_json


def test_parse_text_segments_dialogue():
    text = 'He said, "Hello." Then she replied \u300c\u3053\u3093\u306b\u3061\u306f\u300d.'
    segments = parse_text_segments(text)
    role_types = [s["role_type"] for s in segments]
    assert "DIALOGUE" in role_types
    assert "NARRATION" in role_types


def test_merge_overlap(tmp_path: Path):
    cfg = load_config(Path("config.yaml"))
    asr_path = tmp_path / "asr.jsonl"
    diar_path = tmp_path / "diar.json"
    out_path = tmp_path / "segments.jsonl"

    append_jsonl(
        asr_path,
        [
            {"start": 0.0, "end": 2.0, "text": "Hello", "confidence": 0.9},
            {"start": 5.0, "end": 7.0, "text": "World", "confidence": 0.9},
        ],
    )
    write_json(
        diar_path,
        [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "overlap": False},
            {"start": 5.0, "end": 7.0, "speaker": "SPEAKER_01", "overlap": False},
        ],
    )
    merge_asr_diarization(asr_path, diar_path, out_path, cfg)
    merged = [line for line in out_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(merged) == 2


def test_pause_rules():
    cfg = load_config(Path("config.yaml"))
    segments = [
        {"text": "Hello."},
        {"text": "Wait?"},
        {"text": "Ok"},
    ]
    add_pause_rules(segments, cfg)
    assert segments[0]["pause_after_sec"] > 0
    assert segments[1]["pause_after_sec"] > 0
    assert segments[2]["pause_after_sec"] == cfg.text_dub.default_pause_sec
