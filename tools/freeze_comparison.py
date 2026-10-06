"""Create an executable frozen comparison manifest from actual model/config/data pipelines."""
import argparse
import copy
import json
import sys
import tempfile
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataloader import create_dataloaders
from training.checkpoint import collect_git_info, write_json_atomic
from training.trainer import Trainer, load_config
from utils.formal_protocol import code_fingerprint, load_frozen_protocol
from utils.model_spec import build_model_from_spec, make_spec_from_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="append", nargs=3, required=True,
                        metavar=("MODEL", "ARM", "CONFIG"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("冻结输出已存在；请使用新协议 id 和新文件名")
    git = collect_git_info()
    if git["commit"] == "unknown" or git["dirty"]:
        raise SystemExit("请先提交实现/配置的定稿，再冻结可追溯的干净代码基准")
    plans = []
    with tempfile.TemporaryDirectory(prefix="fer_freeze_") as directory:
        for index, (model_name, arm, config_path) in enumerate(args.plan):
            config = load_config(config_path)
            config["seed"] = args.seeds[0]
            spec = make_spec_from_config(config, model_name)
            train, val, _, _ = create_dataloaders(config, model_name, include_test=False)
            trainer = Trainer(
                build_model_from_spec(spec), train, val, None, config, model_name,
                class_counts=train.dataset.class_counts, run_dir=Path(directory) / str(index),
                run_purpose="smoke",
            )
            plans.append({
                "model_name": model_name, "arm": arm, "seeds": args.seeds,
                "model_spec": spec.to_dict(),
                "training_protocol": copy.deepcopy(trainer.get_training_protocol()),
                "max_epochs": trainer.scheduler_num_epochs,
                "allow_early_stop": True, "lr_floor": 1e-7,
            })
        body = {
            "schema_version": 1, "protocol_id": args.protocol_id,
            "frozen_at": date.today().isoformat(), "git_commit": git["commit"],
            "code_sha256": code_fingerprint(), "plans": plans,
        }
        # Validate before publishing; never leave a partially valid frozen file.
        draft = Path(directory) / "manifest.json"
        draft.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
        load_frozen_protocol(draft)
        write_json_atomic(args.output, body)
    print(f"冻结清单已创建：{args.output}（{len(plans)} 个模型/臂，seeds={args.seeds}）")


if __name__ == "__main__":
    main()

