"""Script chuẩn bị tập Train JSONL & Huấn luyện QLoRA trên GPU (Colab / Linux)."""

import argparse
import logging
import sys
from pathlib import Path

from finetuning.data_prep import TextToSQLSample, deduplicate_and_prevent_leakage, export_jsonl
from finetuning.qlora_config import QLoRATrainingConfig

logger = logging.getLogger(__name__)

SAMPLE_SCHEMA = (
    "companies(ticker, company_name, exchange, sector)\n"
    "quarterly_financials(ticker, year, quarter, revenue_billion_vnd, net_profit_billion_vnd, roe_pct)\n"
    "daily_stock_prices(ticker, trade_date, close_price_vnd, pe_ratio, pb_ratio)"
)


def build_seed_dataset(output_path: Path) -> int:
    """Sinh tập dữ liệu mẫu Text-to-SQL tài chính Việt Nam chuẩn ChatML."""
    raw_samples = [
        TextToSQLSample(
            question="Doanh thu và lợi nhuận ròng quý 1 năm 2025 của FPT là bao nhiêu?",
            db_schema=SAMPLE_SCHEMA,
            sql_query=(
                "SELECT revenue_billion_vnd, net_profit_billion_vnd "
                "FROM quarterly_financials WHERE ticker = 'FPT' AND year = 2025 AND quarter = 1 LIMIT 100"
            ),
        ),
        TextToSQLSample(
            question="Top 3 doanh nghiệp có ROE cao nhất năm 2025?",
            db_schema=SAMPLE_SCHEMA,
            sql_query=(
                "SELECT ticker, roe_pct FROM quarterly_financials "
                "WHERE year = 2025 ORDER BY roe_pct DESC LIMIT 3"
            ),
        ),
        TextToSQLSample(
            question="Chỉ số P/E gần nhất của Hòa Phát (HPG) là bao nhiêu?",
            db_schema=SAMPLE_SCHEMA,
            sql_query=(
                "SELECT pe_ratio, trade_date FROM daily_stock_prices "
                "WHERE ticker = 'HPG' ORDER BY trade_date DESC LIMIT 1"
            ),
        ),
    ]
    clean = deduplicate_and_prevent_leakage(raw_samples, test_questions=set())
    return export_jsonl(clean, output_path)


def run_training(train_jsonl: Path, output_dir: Path) -> None:
    """Khởi chạy QLoRA SFTTrainer (chạy trên máy có GPU CUDA + peft/trl/bitsandbytes)."""
    try:
        import torch  # type: ignore[import-untyped]
        from datasets import load_dataset  # type: ignore[import-untyped]
        from peft import LoraConfig  # type: ignore[import-untyped]
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore[import-untyped]
        from trl import SFTConfig, SFTTrainer  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "Thiếu thư viện GPU (torch, transformers, peft, trl, bitsandbytes, datasets). "
            "Chạy trên Google Colab GPU hoặc server Linux có CUDA."
        ) from exc

    cfg = QLoRATrainingConfig()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, quantization_config=bnb_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    dataset = load_dataset("json", data_files=str(train_jsonl), split="train")

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        peft_config=lora_config,
        args=SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=cfg.num_train_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            max_seq_length=cfg.max_seq_length,
            logging_steps=10,
        ),
    )
    trainer.train()
    trainer.save_model(str(output_dir))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-only", action="store_true", help="Chỉ xuất tập train_sample.jsonl.")
    parser.add_argument("--data-path", type=Path, default=Path("data/finetuning/train_sample.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/vnfin-qwen2.5-3b-lora"))
    args = parser.parse_args()

    count = build_seed_dataset(args.data_path)
    print(f"Đã xuất {count} mẫu ChatML vào {args.data_path}")
    if not args.export_only:
        run_training(args.data_path, args.output_dir)
