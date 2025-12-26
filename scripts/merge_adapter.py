import argparse
import os
import sys

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_model_id() -> str:
    candidate = os.environ.get("BASE_MODEL_ID", "").strip()
    if not candidate:
        print("Error: BASE_MODEL_ID environment variable is required", file=sys.stderr)
        sys.exit(1)
    return candidate


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("adapter_path", help="Path to the LoRA adapter checkpoint")
    parser.add_argument(
        "-o", "--output", default="merged_model", help="Output directory for merged model"
    )
    args = parser.parse_args()

    base = resolve_model_id()
    adapter = args.adapter_path
    output_dir = args.output

    print(f"Base model: {base}")
    print(f"Adapter: {adapter}")
    print(f"Output: {output_dir}")

    base_model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto")
    print("Loaded base model")

    model = PeftModel.from_pretrained(base_model, adapter, torch_dtype="auto")
    print("Applied adapter weights")

    model = model.merge_and_unload()
    print("Merged adapter into base weights")
    # Persist merged weights/config to disk for downstream conversion.
    model.save_pretrained(output_dir)
    # Copy tokenizer artifacts to the same folder to keep the model package complete.
    AutoTokenizer.from_pretrained(base).save_pretrained(output_dir)
    print(f"Merged model saved to {output_dir}")


if __name__ == "__main__":
    main()
