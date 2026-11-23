"""Cấu hình chuẩn cho QLoRA Fine-tuning trên Qwen-2.5-3B/7B-Instruct."""

from pydantic import BaseModel, Field

LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class QLoRATrainingConfig(BaseModel):
    """Siêu tham số chuẩn hóa cho SFT + QLoRA (NF4) + Loss Masking."""

    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    lora_r: int = Field(default=16, ge=4, le=128)
    lora_alpha: int = Field(default=32, ge=8, le=256)
    lora_dropout: float = Field(default=0.0, ge=0.0, le=0.2)
    target_modules: list[str] = Field(default_factory=lambda: list(LORA_TARGET_MODULES))
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    mask_user_prompt_loss: bool = True  # Chỉ tính Loss trên phần Assistant (SQL)
