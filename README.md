# Notion-Qwen2.5 Training

This repository contains the training notebook for fine-tuning Qwen2.5-1.5B-Instruct model to generate structured Notion templates. The fine-tuned model is hosted on Hugging Face at [gamithasam/notion-qwen2.5-1.5B](https://huggingface.co/gamithasam/notion-qwen2.5-1.5B).

## Repository Contents

- `finetune_qwen2_5_1_5B.ipynb`: Jupyter notebook containing the complete fine-tuning process

## Training Data

The model was fine-tuned using the dataset from [sbhatti2009/NotionGPT](https://huggingface.co/spaces/sbhatti2009/NotionGPT). The training data can be found at:
- [Download training data](https://huggingface.co/spaces/sbhatti2009/NotionGPT/resolve/main/data/finetuning_data_cot_v12.jsonl)

## Model Description

The fine-tuned model specializes in generating structured Notion templates, understanding and generating complex JSON blueprints that represent detailed, organized, and highly functional Notion templates.

### Base Model
- **Name**: [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- **Type**: Instruction-tuned language model
- **Size**: 1.5B parameters

## Training Details

The notebook implements the following training configuration:

- **Method**: LoRA (Low-Rank Adaptation)
- **Parameters**:
  - Learning rate: 2e-4
  - Epochs: 3
  - Batch size: 1
  - Gradient accumulation steps: 4
  - LoRA rank: 16
  - LoRA alpha: 32
  - Target modules: q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj
  - Training precision: fp16

## Usage

The fine-tuned model can be accessed directly from Hugging Face:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "gamithasam/notion-qwen2.5-1.5B",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("gamithasam/notion-qwen2.5-1.5B")

# Create pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Example prompt
prompt = """<|im_start|>user
Generate me a Notion template for tracking my daily tasks and habits.
<|im_end|>
<|im_start|>assistant
"""

result = pipe(
    prompt,
    max_new_tokens=1000,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id
)
```

## Limitations

- The model requires specific formatting with `<|im_start|>` and `<|im_end|>` tokens
- Generated JSON should be validated before use in production
- May require adjustments for very complex template structures
- Performance depends on available GPU memory
- Training dataset size was limited to 41 examples

## Intended Uses

This model is designed for:
- Generating structured Notion templates
- Creating documentation layouts
- Organizing information in a hierarchical format
- Planning project structures
- Creating knowledge bases

## Links

- Fine-tuned model: [gamithasam/notion-qwen2.5-1.5B](https://huggingface.co/gamithasam/notion-qwen2.5-1.5B)
- Base model: [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- Training data source: [sbhatti2009/NotionGPT](https://huggingface.co/spaces/sbhatti2009/NotionGPT)

## License

This repository inherits the license of the base Qwen2.5-1.5B-Instruct model (Apache License Version 2.0).
