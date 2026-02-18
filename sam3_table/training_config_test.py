"""Golden tests for SAM3LoRAConfig parsed from testSamples/full_lora_config.yaml."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from training_config import (
    SAM3LoRAConfig,
    DataConfig,
    DatasetSplit,
    Device,
    EvalMetric,
    LRScheduler,
    MixedPrecision,
)

SAMPLE_YAML = Path(__file__).parent / "testSamples" / "full_lora_config.yaml"


@pytest.fixture
def config() -> SAM3LoRAConfig:
    return SAM3LoRAConfig.from_yaml(SAMPLE_YAML)


# ── Model section ───────────────────────────────────────────────────────────

class TestModelConfig:
    def test_name(self, config: SAM3LoRAConfig):
        assert config.model.name == "facebook/sam3"

    def test_cache_dir_is_none(self, config: SAM3LoRAConfig):
        assert config.model.cache_dir is None


# ── LoRA section ────────────────────────────────────────────────────────────

class TestLoRAConfig:
    def test_rank(self, config: SAM3LoRAConfig):
        assert config.lora.rank == 32

    def test_alpha(self, config: SAM3LoRAConfig):
        assert config.lora.alpha == 64

    def test_dropout(self, config: SAM3LoRAConfig):
        assert config.lora.dropout == pytest.approx(0.1)

    def test_target_modules(self, config: SAM3LoRAConfig):
        expected = [
            "q_proj", "k_proj", "v_proj", "out_proj",
            "qkv", "proj", "fc1", "fc2",
            "c_fc", "c_proj",
            "linear1", "linear2",
        ]
        assert config.lora.target_modules == expected

    def test_all_component_flags_true(self, config: SAM3LoRAConfig):
        lora = config.lora
        assert lora.apply_to_vision_encoder is True
        assert lora.apply_to_text_encoder is True
        assert lora.apply_to_geometry_encoder is True
        assert lora.apply_to_detr_encoder is True
        assert lora.apply_to_detr_decoder is True
        assert lora.apply_to_mask_decoder is True


# ── Training section ────────────────────────────────────────────────────────

class TestTrainingConfig:
    def test_data_train_split(self, config: SAM3LoRAConfig):
        train_split = config.training.data.train
        assert train_split.image_dir == Path("/workspace/data/train")
        assert train_split.annotation_file == Path(
            "/workspace/data/train/_annotations.coco.json"
        )

    def test_data_valid_split(self, config: SAM3LoRAConfig):
        valid_split = config.training.data.valid
        assert valid_split is not None
        assert valid_split.image_dir == Path("/workspace/data/valid")
        assert valid_split.annotation_file == Path(
            "/workspace/data/valid/_annotations.coco.json"
        )

    def test_batch_size(self, config: SAM3LoRAConfig):
        assert config.training.batch_size == 4

    def test_num_workers(self, config: SAM3LoRAConfig):
        assert config.training.num_workers == 2

    def test_learning_rate(self, config: SAM3LoRAConfig):
        assert config.training.learning_rate == pytest.approx(5e-5)

    def test_weight_decay(self, config: SAM3LoRAConfig):
        assert config.training.weight_decay == pytest.approx(0.01)

    def test_adam_betas(self, config: SAM3LoRAConfig):
        assert config.training.adam_beta1 == pytest.approx(0.9)
        assert config.training.adam_beta2 == pytest.approx(0.999)

    def test_adam_epsilon(self, config: SAM3LoRAConfig):
        assert config.training.adam_epsilon == pytest.approx(1e-8)

    def test_max_grad_norm(self, config: SAM3LoRAConfig):
        assert config.training.max_grad_norm == pytest.approx(1.0)

    def test_num_epochs(self, config: SAM3LoRAConfig):
        assert config.training.num_epochs == 100

    def test_warmup_steps(self, config: SAM3LoRAConfig):
        assert config.training.warmup_steps == 200

    def test_lr_scheduler(self, config: SAM3LoRAConfig):
        assert config.training.lr_scheduler == LRScheduler.COSINE

    def test_logging_steps(self, config: SAM3LoRAConfig):
        assert config.training.logging_steps == 10

    def test_eval_and_save_steps(self, config: SAM3LoRAConfig):
        assert config.training.eval_steps == 100
        assert config.training.save_steps == 100
        assert config.training.save_total_limit == 5

    def test_mixed_precision(self, config: SAM3LoRAConfig):
        assert config.training.mixed_precision == MixedPrecision.BF16

    def test_seed(self, config: SAM3LoRAConfig):
        assert config.training.seed == 42

    def test_gradient_accumulation(self, config: SAM3LoRAConfig):
        assert config.training.gradient_accumulation_steps == 8

    def test_effective_batch_size(self, config: SAM3LoRAConfig):
        assert config.training.effective_batch_size == 4 * 8


# ── Output section ──────────────────────────────────────────────────────────

class TestOutputConfig:
    def test_output_dir(self, config: SAM3LoRAConfig):
        assert config.output.output_dir == "outputs/sam3_lora_full"

    def test_logging_dir(self, config: SAM3LoRAConfig):
        assert config.output.logging_dir == "logs"

    def test_save_lora_only(self, config: SAM3LoRAConfig):
        assert config.output.save_lora_only is True

    def test_push_to_hub_disabled(self, config: SAM3LoRAConfig):
        assert config.output.push_to_hub is False
        assert config.output.hub_model_id is None


# ── Evaluation section ──────────────────────────────────────────────────────

class TestEvaluationConfig:
    def test_metric(self, config: SAM3LoRAConfig):
        assert config.evaluation.metric == EvalMetric.IOU

    def test_save_predictions(self, config: SAM3LoRAConfig):
        assert config.evaluation.save_predictions is False

    def test_compute_metrics_during_training(self, config: SAM3LoRAConfig):
        assert config.evaluation.compute_metrics_during_training is True


# ── Hardware section ────────────────────────────────────────────────────────

class TestHardwareConfig:
    def test_device(self, config: SAM3LoRAConfig):
        assert config.hardware.device == Device.CUDA

    def test_pin_memory(self, config: SAM3LoRAConfig):
        assert config.hardware.dataloader_pin_memory is True

    def test_compile_disabled(self, config: SAM3LoRAConfig):
        assert config.hardware.use_compile is False


# ── Round-trip serialization ────────────────────────────────────────────────

class TestRoundTrip:
    def test_yaml_round_trip(self, config: SAM3LoRAConfig, tmp_path: Path):
        out_path = tmp_path / "roundtrip.yaml"
        config.to_yaml(out_path)
        reloaded = SAM3LoRAConfig.from_yaml(out_path)
        assert reloaded == config

    def test_json_round_trip(self, config: SAM3LoRAConfig):
        json_str = config.model_dump_json()
        reloaded = SAM3LoRAConfig.model_validate_json(json_str)
        assert reloaded == config


# ── Validation edge cases ───────────────────────────────────────────────────

class TestValidation:
    def test_lora_rank_must_be_positive(self):
        with pytest.raises(ValidationError):
            SAM3LoRAConfig.model_validate({
                "lora": {"rank": 0},
                "training": {"data": {"train": {
                    "image_dir": "/tmp",
                    "annotation_file": "/tmp/a.json",
                }}},
            })

    def test_annotation_file_must_be_json(self):
        with pytest.raises(ValidationError, match="annotation_file must be a .json file"):
            DatasetSplit(
                image_dir=Path("/tmp"),
                annotation_file=Path("/tmp/data.csv"),
            )

    def test_hub_id_required_when_pushing(self):
        with pytest.raises(ValidationError, match="hub_model_id is required"):
            SAM3LoRAConfig.model_validate({
                "output": {"push_to_hub": True, "hub_model_id": None},
                "training": {"data": {"train": {
                    "image_dir": "/tmp",
                    "annotation_file": "/tmp/a.json",
                }}},
            })

    def test_valid_split_is_optional(self):
        cfg = SAM3LoRAConfig.model_validate({
            "training": {"data": {"train": {
                "image_dir": "/tmp",
                "annotation_file": "/tmp/a.json",
            }}},
        })
        assert cfg.training.data.valid is None
