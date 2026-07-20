"""
单元测试 — 模型、数据集、Trainer 核心功能

运行方式:
    pytest tests/ -v
"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch
import numpy as np
import pandas as pd


# ============================================================
# 测试激活函数工厂
# ============================================================
class TestActivationFactory:
    def test_get_activation_relu(self):
        from utils.activations import get_activation
        act = get_activation("relu")
        assert isinstance(act, torch.nn.ReLU)

    def test_get_activation_leaky_relu(self):
        from utils.activations import get_activation
        act = get_activation("leaky_relu", negative_slope=0.01)
        assert isinstance(act, torch.nn.LeakyReLU)

    def test_get_activation_gelu(self):
        from utils.activations import get_activation
        act = get_activation("gelu")
        assert isinstance(act, torch.nn.GELU)

    def test_get_activation_case_insensitive(self):
        from utils.activations import get_activation
        act = get_activation("RELU")
        assert isinstance(act, torch.nn.ReLU)

    def test_get_activation_invalid(self):
        from utils.activations import get_activation
        with pytest.raises(ValueError, match="不支持的激活函数"):
            get_activation("swish")

    def test_registry_completeness(self):
        from utils.activations import ACTIVATION_REGISTRY
        assert set(ACTIVATION_REGISTRY.keys()) == {"relu", "leaky_relu", "elu", "gelu"}


# ============================================================
# 测试模型 Forward Pass
# ============================================================
class TestModels:
    @pytest.fixture(params=["mini_cnn", "vgg_lite", "micro_resnet"])
    def model_name(self, request):
        return request.param

    @pytest.fixture
    def model_cls(self, model_name):
        """按 model_name 返回对应的模型类"""
        from models import MiniCNN, VGGLite, MicroResNet
        return {"mini_cnn": MiniCNN, "vgg_lite": VGGLite, "micro_resnet": MicroResNet}[model_name]

    @pytest.fixture
    def dummy_input(self):
        return torch.randn(2, 1, 48, 48)

    @pytest.fixture
    def single_input(self):
        return torch.randn(1, 1, 48, 48)

    def test_forward_shape(self, model_cls, dummy_input):
        """模型 forward 应返回 (batch_size, 7) 的输出"""
        model = model_cls(num_classes=7)
        output = model(dummy_input)
        assert output.shape == (2, 7)

    def test_gradient_flow(self, model_cls):
        """反向传播梯度应正常流动（无 None 梯度）"""
        model = model_cls(num_classes=7)
        output = model(torch.randn(2, 1, 48, 48))
        output.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{model_cls.__name__}: {name} 梯度为 None"

    def test_activation_config(self, model_cls, single_input):
        """通过 activation 参数指定不同激活函数"""
        model = model_cls(num_classes=7, activation="elu")
        output = model(single_input)
        assert output.shape == (1, 7)

    def test_num_classes_override(self, model_cls, single_input):
        """num_classes 参数应正确控制输出维度"""
        model = model_cls(num_classes=10)
        output = model(single_input)
        assert output.shape == (1, 10)


# ============================================================
# 测试 FER2013Dataset
# ============================================================
class TestFER2013Dataset:
    @pytest.fixture
    def sample_dataframe(self):
        return pd.DataFrame({
            "emotion": [0, 1, 2, 3, 4, 5, 6],
            "pixels": [
                " ".join(["128"] * 48 * 48) for _ in range(7)
            ],
        })

    def test_len(self, sample_dataframe):
        from data.dataloader import FER2013Dataset
        dataset = FER2013Dataset(sample_dataframe)
        assert len(dataset) == 7

    def test_getitem_shape(self, sample_dataframe):
        from data.dataloader import FER2013Dataset
        dataset = FER2013Dataset(sample_dataframe)
        image, label = dataset[0]
        assert image.shape == (1, 48, 48), f"期望 (1,48,48)，得到 {image.shape}"
        assert isinstance(label, (int, np.integer))

    def test_getitem_normalized(self, sample_dataframe):
        """像素值应归一化到 [0, 1]"""
        from data.dataloader import FER2013Dataset
        dataset = FER2013Dataset(sample_dataframe)
        image, _ = dataset[0]
        assert image.min() >= 0.0
        assert image.max() <= 1.0

    def test_no_dataframe_retention(self, sample_dataframe):
        """Dataset 不应持有 DataFrame 引用（避免多进程 pickle 膨胀）"""
        from data.dataloader import FER2013Dataset
        dataset = FER2013Dataset(sample_dataframe)
        assert not hasattr(dataset, "data"), "Dataset 不应保留 DataFrame 引用"
        assert hasattr(dataset, "_pixels") and hasattr(dataset, "_labels")

    def test_transform_applied(self, sample_dataframe):
        """指定 transform 时应被应用"""
        from data.dataloader import FER2013Dataset
        from torchvision import transforms

        transform = transforms.Compose([transforms.Normalize([0.5], [0.5])])
        dataset = FER2013Dataset(sample_dataframe, transform=transform)
        image, _ = dataset[0]
        # Normalize 后值域可能超出 [0,1]
        assert image.shape == (1, 48, 48)


# ============================================================
# 测试 Trainer 核心功能
# ============================================================
class TestTrainer:
    @pytest.fixture
    def dummy_config(self):
        return {
            "training": {
                "num_epochs": 2, "patience": 7, "batch_size": 4,
                "learning_rate": 0.001, "weight_decay": 1e-4,
                "optimizer": "adam", "scheduler": "none", "amp": False,
                "gradient_accumulation_steps": 1, "max_grad_norm": 1.0,
            },
            "models": {"mini_cnn": {}},
            "augmentation": {"enabled": False},
            "checkpoint": {"save_every_n_epochs": 5},
            "data": {"class_names": ["A", "B", "C"]},
        }

    @pytest.fixture
    def dummy_loader(self):
        from torch.utils.data import DataLoader, TensorDataset
        X = torch.randn(16, 1, 48, 48)
        y = torch.randint(0, 3, (16,))
        return DataLoader(TensorDataset(X, y), batch_size=4)

    # Helper: 构建测试用 Trainer 实例（减少重复代码）
    def _make_trainer(self, dummy_config, dummy_loader, num_classes=3):
        from training.trainer import Trainer
        from models.mini_cnn import MiniCNN
        model = MiniCNN(num_classes=num_classes)
        return Trainer(
            model=model, train_loader=dummy_loader, val_loader=dummy_loader,
            test_loader=dummy_loader, config=dummy_config,
            model_name="mini_cnn", device=torch.device("cpu"),
        )

    def test_trainer_init(self, dummy_config, dummy_loader):
        trainer = self._make_trainer(dummy_config, dummy_loader)
        assert trainer.scheduler_num_epochs == 2
        assert trainer.use_amp is False
        assert trainer.grad_accum_steps == 1
        assert trainer.max_grad_norm == 1.0

    def test_evaluate_returns_three_values(self, dummy_config, dummy_loader):
        trainer = self._make_trainer(dummy_config, dummy_loader)
        result = trainer.evaluate()
        assert len(result) == 3, f"evaluate 应返回三元组，得到 {len(result)} 项"
        val_loss, top1_acc, top5_acc = result
        assert 0 <= top1_acc <= 1
        assert 0 <= top5_acc <= 1

    def test_save_and_load_checkpoint(self, dummy_config, dummy_loader, tmp_path):
        trainer = self._make_trainer(dummy_config, dummy_loader)
        # 模拟 fit() 的行为：设置 _current_epoch 后训练 + 评估，追加 history
        trainer._current_epoch = 1
        train_loss, train_acc = trainer.train_one_epoch()
        val_loss, val_acc, val_top5 = trainer.evaluate()
        for k, v in [("train_loss", train_loss), ("train_acc", train_acc),
                      ("val_loss", val_loss), ("val_acc", val_acc),
                      ("val_top5_acc", val_top5)]:
            trainer.history[k].append(v)
        trainer.history["lr"].append(0.001)

        # 保存（epoch 应为 1，因为 history 有 1 条记录）
        ckpt_path = tmp_path / "test.pth"
        saved_path = trainer.save_checkpoint(ckpt_path)
        assert saved_path.exists()

        # 验证保存的 epoch 值
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert ckpt["epoch"] == 1

        # 新 Trainer 加载（start_epoch 应为 epoch + 1 = 2）
        trainer2 = self._make_trainer(dummy_config, dummy_loader)
        trainer2.load_checkpoint(ckpt_path)
        assert trainer2.start_epoch == 2
        assert len(trainer2.history["val_acc"]) == 1

    def test_load_checkpoint_wrong_model_raises(self, dummy_config, dummy_loader, tmp_path):
        """加载不匹配的模型名 checkpoint 应报错"""
        # 创建一个模型名不匹配的 checkpoint
        trainer = self._make_trainer(dummy_config, dummy_loader)
        trainer._current_epoch = 1
        ckpt_path = tmp_path / "test.pth"
        trainer.save_checkpoint(ckpt_path)

        # 手动修改 checkpoint 中的模型名
        ckpt = torch.load(ckpt_path, weights_only=False)
        ckpt["model_name"] = "wrong_model"
        torch.save(ckpt, ckpt_path)

        # 尝试加载 — 应抛出 RuntimeError
        trainer2 = self._make_trainer(dummy_config, dummy_loader)
        with pytest.raises(RuntimeError, match="模型名不匹配"):
            trainer2.load_checkpoint(ckpt_path)


# ============================================================
# 测试 CB Focal Loss
# ============================================================
class TestCBFocalLoss:
    def test_cb_focal_forward(self):
        from utils.losses import CBFocalLoss
        class_counts = [100, 10, 50]
        criterion = CBFocalLoss(gamma=2.0, beta=0.999, class_counts=class_counts)
        inputs = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 1])
        loss = criterion(inputs, targets)
        assert loss.item() > 0, "loss 应为正数"
        assert not torch.isnan(loss), "loss 不应为 NaN"

    def test_cb_focal_beta_range(self):
        from utils.losses import CBFocalLoss
        with pytest.raises(ValueError, match="beta"):
            CBFocalLoss(gamma=2.0, beta=1.0, class_counts=[100, 10])

    def test_cb_focal_weights_ordering(self):
        """少样本类应获得更高的 CB 权重"""
        from utils.losses import CBFocalLoss
        criterion = CBFocalLoss(gamma=2.0, beta=0.9, class_counts=[1000, 100, 10])
        w0 = criterion._cb_weights[0].item()
        w1 = criterion._cb_weights[1].item()
        w2 = criterion._cb_weights[2].item()
        assert w0 < w1 < w2, f"少样本类权重应更高: {w0:.4f} < {w1:.4f} < {w2:.4f}"


# ============================================================
# 测试 MixUp
# ============================================================
class TestMixUp:
    def test_mixup_data_shapes(self):
        from training.trainer import mixup_data
        import numpy as np
        x = torch.randn(8, 1, 48, 48)
        y = torch.randint(0, 7, (8,))
        mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2, device="cpu")
        assert mixed_x.shape == x.shape, "混合后图像形状应一致"
        assert y_a.shape == y.shape, "标签 A 形状应一致"
        assert y_b.shape == y.shape, "标签 B 形状应一致"
        assert 0 <= lam <= 1, "lambda 应在 [0,1] 区间"

    def test_mixup_criterion(self):
        from training.trainer import mixup_criterion
        from utils.losses import FocalLoss
        criterion = FocalLoss(gamma=2.0)
        pred = torch.randn(8, 7)
        y_a = torch.randint(0, 7, (8,))
        y_b = torch.randint(0, 7, (8,))
        loss = mixup_criterion(criterion, pred, y_a, y_b, lam=0.4)
        assert loss.item() > 0, "混叠损失应为正数"


# ============================================================
# 测试配置加载
# ============================================================
class TestConfig:
    def test_load_config(self):
        from training.trainer import load_config
        config = load_config()
        assert "training" in config
        assert "models" in config
        assert "augmentation" in config
        assert "dataloader" in config

    def test_config_dataloader_section(self):
        from training.trainer import load_config
        config = load_config()
        dl = config["dataloader"]
        assert "num_workers" in dl
        assert "pin_memory" in dl
        assert "persistent_workers" in dl
        assert "class_balanced_sampling" in dl

    def test_config_training_amp(self):
        from training.trainer import load_config
        config = load_config()
        assert "amp" in config["training"]

    def test_config_no_dead_code(self):
        """配置中不应存在不被代码读取的键"""
        from training.trainer import load_config
        config = load_config()
        # 这些键已被清理
        assert "class_balance" not in config, "class_balance 是死代码，应已清理"
        assert "save_dir" not in config.get("checkpoint", {}), "checkpoint.save_dir 是死代码"
