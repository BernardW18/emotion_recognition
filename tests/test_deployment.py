"""Exported default selection, compatibility and missing-default fallback."""

import json

import pytest
import torch

from inference import infer_utils
from tools import export_model
from utils.constants import CLASS_NAMES
from utils.model_spec import ModelSpec, build_model_from_spec, file_sha256


def test_default_export_roundtrip_and_later_export_preserves_default(tmp_path, monkeypatch):
    saved = tmp_path / "saved_models"
    manifest_path = saved / "export_manifest.json"
    monkeypatch.setattr(export_model, "SAVED_DIR", saved)
    monkeypatch.setattr(export_model, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(infer_utils, "SAVED_MODELS_DIR", saved)
    monkeypatch.setattr(infer_utils, "MANIFEST_PATH", manifest_path)

    spec = ModelSpec(
        model_name="micro_resnet", spec_version=2,
        class_names=list(CLASS_NAMES), activation="gelu", dropout=0.3, use_se=False,
        input_channels=1, image_size=48, normalize="x/255.0",
        blocks=(2, 2), channels=(64, 128), pool_order="after_stage1",
    )
    source = tmp_path / "source.pth"
    torch.save({
        "model_name": "micro_resnet", "run_id": "deployment-test", "format_version": 2,
        "model_spec": spec.to_dict(),
        "model_state_dict": build_model_from_spec(spec).state_dict(),
    }, source)
    first = export_model.export_checkpoint(source, name="z_recommended", make_default=True)
    second = export_model.export_checkpoint(source, name="a_other")
    manifest = infer_utils.read_export_manifest()
    assert manifest["default_checkpoint"] == first["file"]
    assert manifest["exports"] == [first, second]
    assert first["exported_sha256"] == file_sha256(source)
    items = infer_utils.list_available_checkpoints()
    assert items[0]["file"] == second["file"]
    recommended = [item for item in items if item["is_default"]]
    assert len(recommended) == 1 and recommended[0]["file"] == first["file"]
    assert recommended[0]["label"].startswith("[推荐]")
    model, _, meta = infer_utils.load_model(recommended[0]["path"], device="cpu")
    assert meta["model_spec"]["pool_order"] == "after_stage1"
    assert sum(p.numel() for p in model.parameters()) == 753991
    with pytest.raises(FileExistsError):
        export_model.export_checkpoint(source, name="z_recommended", make_default=True)
    assert infer_utils.read_export_manifest() == manifest


@pytest.mark.parametrize("default", [None, "missing.pth", "orphan.pth"])
def test_missing_or_unregistered_default_keeps_legacy_available(tmp_path, monkeypatch, default):
    monkeypatch.setattr(infer_utils, "SAVED_MODELS_DIR", tmp_path)
    monkeypatch.setattr(infer_utils, "MANIFEST_PATH", tmp_path / "export_manifest.json")
    for name in ("legacy.pth", "orphan.pth"):
        (tmp_path / name).write_bytes(b"listing does not load weights")
    (tmp_path / "export_manifest.json").write_text(json.dumps({
        "exports": [{"file": "legacy.pth", "model_name": "micro_resnet", "legacy": True}],
        "default_checkpoint": default,
    }), encoding="utf-8")
    items = infer_utils.list_available_checkpoints()
    assert [item["file"] for item in items] == ["legacy.pth", "orphan.pth"]
    assert all(item["legacy"] and not item["is_default"] for item in items)
