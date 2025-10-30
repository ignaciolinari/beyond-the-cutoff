from pathlib import Path

from beyond_the_cutoff.config import ProjectConfig, load_config


def test_load_default_config() -> None:
    cfg = load_config()
    assert isinstance(cfg, ProjectConfig)
    # Paths resolved to absolute
    assert Path(cfg.paths.raw_data).is_absolute()
    assert cfg.retrieval.chunk_size > 0
    assert cfg.retrieval.top_k >= 1
    assert cfg.fine_tuning.base_model.endswith("SmolLM2-135M")
    assert cfg.inference.provider == "transformers"
    assert cfg.inference.model == "HuggingFaceTB/SmolLM2-135M"
