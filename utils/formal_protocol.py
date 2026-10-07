"""Executable frozen plans: exact settings/data binding and experiment completion."""
import copy
import hashlib
import json
import re
import subprocess
from datetime import date
from functools import lru_cache
from pathlib import Path

from utils.model_spec import file_sha256

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRS = ("models", "data", "training", "inference", "utils", "tools")


@lru_cache(maxsize=32)
def archived_code_fingerprint(commit: str, *, root: Path = PROJECT_ROOT) -> str:
    """Verify historical code against actual Git blobs, never against a claimed hash alone."""
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("历史代码校验需要完整 Git commit SHA")

    def git(*args: str) -> bytes:
        try:
            return subprocess.run(
                ["git", "-C", str(root), *args], check=True, capture_output=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError(f"无法读取冻结提交 {commit} 的代码快照") from exc

    paths = git("ls-tree", "-rz", "--name-only", commit).decode("utf-8").split("\0")
    selected = [p for p in paths if (
        p.endswith(".py") and p.split("/", 1)[0] in SOURCE_DIRS
        or p.startswith("configs/") and p.count("/") == 1 and p.endswith(".yaml")
    )]
    if not selected:
        raise ValueError("冻结提交中没有可验证源码")
    digest = hashlib.sha256()
    for path in sorted(selected):
        digest.update(path.encode("utf-8"))
        digest.update(hashlib.sha256(git("show", f"{commit}:{path}")).digest())
    return digest.hexdigest()


def code_fingerprint() -> str:
    """Bind executed source/config bytes; documentation or commit-only changes are harmless."""
    root = PROJECT_ROOT
    files: list[Path] = []
    for directory in SOURCE_DIRS:
        files.extend((root / directory).rglob("*.py"))
    files.extend((root / "configs").glob("*.yaml"))
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def normalized_protocol(protocol):
    result = copy.deepcopy(protocol)
    result["config"].pop("seed", None)  # Seeds are the only plan-level variable.
    return result


def load_frozen_protocol(path, *, historical: bool = False) -> dict:
    path = Path(path)
    if not path.exists():
        raise ValueError(f"冻结协议文件不存在: {path}")
    body = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict) or body.get("schema_version") != 1:
        raise ValueError("冻结协议缺少可执行 schema_version=1 清单")
    for key in ("protocol_id", "frozen_at", "git_commit"):
        if not isinstance(body.get(key), str) or not body[key].strip():
            raise ValueError(f"冻结协议缺少 {key}")
    date.fromisoformat(body["frozen_at"])
    if body.get("code_sha256") != code_fingerprint() and (
        not historical or body.get("code_sha256") != archived_code_fingerprint(body["git_commit"])
    ):
        raise ValueError("实际源码/配置或原提交与冻结代码指纹不一致")
    plans = body.get("plans")
    if not isinstance(plans, list) or not plans:
        raise ValueError("冻结协议缺少实际 plans")
    identities = set()
    for plan in plans:
        if not isinstance(plan, dict):
            raise ValueError("冻结 plan 非法")
        identity = (plan.get("model_name"), plan.get("arm"))
        if not all(isinstance(x, str) and x for x in identity) or identity in identities:
            raise ValueError("冻结模型/臂缺失或重复")
        identities.add(identity)
        seeds = plan.get("seeds")
        if (not isinstance(seeds, list) or len(seeds) < 3 or len(set(seeds)) != len(seeds)
                or any(type(seed) is not int or seed < 0 for seed in seeds)):
            raise ValueError("冻结 seeds 非法")
        protocol = plan.get("training_protocol")
        if not isinstance(protocol, dict) or not protocol.get("data_fingerprint"):
            raise ValueError("冻结方案缺完整 training_protocol/数据指纹")
        config = protocol["config"]
        model_config = config["models"][plan["model_name"]]
        budget = model_config.get("num_epochs", config["training"]["num_epochs"])
        if type(plan.get("max_epochs")) is not int or plan["max_epochs"] < 1:
            raise ValueError("冻结 max_epochs 非法")
        if plan["max_epochs"] != budget:
            raise ValueError("冻结预算与实际配置不一致")
        if not isinstance(plan.get("model_spec"), dict):
            raise ValueError("冻结方案缺 model_spec")
        if protocol["runtime"]["save_best"] is not True:
            raise ValueError("正式比较要求保存 best")
        if type(plan.get("allow_early_stop")) is not bool:
            raise ValueError("须明确冻结 allow_early_stop")
        runtime = protocol["runtime"]
        if not plan["allow_early_stop"] and (
            runtime["patience"] > 0 or runtime["val_loss_patience"] > 0
        ):
            raise ValueError("固定预算方案必须关闭两种验证早停")
        if plan.get("lr_floor") != 1e-7:
            raise ValueError("lr_floor 必须与当前 Trainer 的 1e-7 规则一致")
    return {"protocol_id": body["protocol_id"], "file_sha256": file_sha256(path),
            "path": str(path.resolve()), "manifest": body}


def validate_formal_plan(record, model_name, model_spec, protocol, *, historical: bool = False):
    if not isinstance(record, dict) or not record.get("path"):
        raise ValueError("缺少可执行冻结协议绑定")
    current = load_frozen_protocol(record["path"], historical=historical)
    if (record.get("protocol_id") != current["protocol_id"]
            or record.get("file_sha256") != current["file_sha256"]):
        raise ValueError("冻结协议绑定不一致（协议 id / file_sha256）")
    if record.get("manifest") != current["manifest"]:
        raise ValueError("冻结实际内容与绑定不一致")
    actual = normalized_protocol(protocol)
    seed = protocol["config"]["seed"]
    candidates = [
        p for p in current["manifest"]["plans"]
        if p["model_name"] == model_name and p["model_spec"] == model_spec
        and seed in p["seeds"] and normalized_protocol(p["training_protocol"]) == actual
    ]
    if len(candidates) != 1:
        raise ValueError("实际配置/seed/模型/数据/增强/臂与冻结方案不一致或不唯一")
    return candidates[0]


def completion_reason(plan, checkpoint):
    epoch = checkpoint["epoch"]
    if epoch > plan["max_epochs"]:
        raise ValueError("实际轮数超过冻结预算")
    if epoch == plan["max_epochs"]:
        return "budget"
    if not plan["allow_early_stop"] or epoch == 0:
        return None
    runtime = checkpoint["training_protocol"]["runtime"]
    state = checkpoint["early_stop_state"]
    if runtime["patience"] > 0 and state["acc_patience_counter"] >= runtime["patience"]:
        return "val_acc"
    if (runtime["val_loss_patience"] > 0
            and state["loss_worse_counter"] >= runtime["val_loss_patience"]):
        return "val_loss"
    if checkpoint["history"]["lr"][-1] < plan["lr_floor"]:
        return "lr_floor"
    return None

