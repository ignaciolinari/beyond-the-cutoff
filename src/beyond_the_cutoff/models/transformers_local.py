"""Local inference client built on top of Hugging Face Transformers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

torch: Any
AutoModelForCausalLM: Any
AutoTokenizer: Any
_TORCH_IMPORT_ERROR: ModuleNotFoundError | None
_TRANSFORMERS_IMPORT_ERROR: ModuleNotFoundError | None

try:  # pragma: no cover - optional dependency
    torch = import_module("torch")
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

try:  # pragma: no cover - optional dependency
    _transformers = import_module("transformers")
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None
    AutoTokenizer = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    AutoModelForCausalLM = _transformers.AutoModelForCausalLM
    AutoTokenizer = _transformers.AutoTokenizer
    _TRANSFORMERS_IMPORT_ERROR = None

_OPTION_KEYS = {"max_new_tokens", "temperature", "top_p", "repetition_penalty"}


@dataclass
class TransformersClient:
    """Thin wrapper around ``AutoModelForCausalLM.generate`` suitable for local tests."""

    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "auto"
    torch_dtype: str | None = "auto"
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.95
    repetition_penalty: float = 1.05
    stop_sequences: Iterable[str] = field(default_factory=list)

    _tokenizer: Any = field(init=False, repr=False)
    _model: Any = field(init=False, repr=False)
    _device: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if torch is None:
            raise RuntimeError(
                "PyTorch is required to use the Transformers generation backend."
            ) from _TORCH_IMPORT_ERROR
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise RuntimeError(
                "transformers is required to use the Transformers generation backend."
            ) from _TRANSFORMERS_IMPORT_ERROR
        self._device = self._resolve_device(self.device)
        dtype = self._resolve_dtype(self.torch_dtype)
        if self._device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
            dtype = torch.float32

        model_kwargs: dict[str, Any] = {}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        if self._tokenizer.pad_token is None:
            raise ValueError("Tokenizer must define either a pad token or an eos token")

        self._model = AutoModelForCausalLM.from_pretrained(self.model, **model_kwargs)
        self._model.eval()
        self._model.to(self._device)

    def generate(
        self,
        prompt: str,
        *,
        stream: bool = False,
        options: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        if stream:
            raise NotImplementedError("Streaming is not supported for the transformers backend")

        generation_kwargs = self._compose_generation_kwargs(options)
        encoded = self._tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=self._tokenizer.pad_token_id,
                **generation_kwargs,
            )

        generated = outputs[0][input_ids.shape[-1] :].cpu()
        text = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not text:
            text = self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        text = self._apply_stop_sequences(text, generation_kwargs.get("stop_sequences", []))
        return {"response": text}

    def _compose_generation_kwargs(self, options: Mapping[str, Any] | None) -> dict[str, Any]:
        values: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
        }

        if options:
            for key, value in options.items():
                if key in _OPTION_KEYS:
                    values[key] = value

        do_sample = float(values.get("temperature", 0.0)) > 0.0
        values["do_sample"] = do_sample
        stop_sequences = list(self.stop_sequences)
        if options and "stop_sequences" in options:
            stop_sequences.extend(self._coerce_sequence(options["stop_sequences"]))
        values["stop_sequences"] = stop_sequences
        return values

    @staticmethod
    def _apply_stop_sequences(text: str, stops: Iterable[str]) -> str:
        truncated = text
        for stop in stops:
            if not stop:
                continue
            idx = truncated.find(stop)
            if idx != -1:
                truncated = truncated[:idx].rstrip()
        return truncated

    @staticmethod
    def _coerce_sequence(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, Sequence):
            return [str(item) for item in value]
        return [str(value)]

    @staticmethod
    def _resolve_device(device: str) -> Any:
        if torch is None:  # pragma: no cover - defensive guard
            raise RuntimeError(
                "PyTorch is required to resolve devices for the Transformers backend."
            ) from _TORCH_IMPORT_ERROR
        normalized = (device or "auto").lower()
        if normalized == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(normalized)

    @staticmethod
    def _resolve_dtype(dtype: str | None) -> Any:
        if torch is None:  # pragma: no cover - defensive guard
            raise RuntimeError(
                "PyTorch is required to resolve dtypes for the Transformers backend."
            ) from _TORCH_IMPORT_ERROR
        if not dtype or dtype == "auto":
            return None
        normalized = dtype.lower()
        if normalized in {"float32", "fp32"}:
            return torch.float32
        if normalized in {"float16", "fp16"}:
            return torch.float16
        if normalized in {"bfloat16", "bf16"}:
            return torch.bfloat16
        raise ValueError(f"Unsupported torch dtype: {dtype}")
