from objectclear.pipelines import ObjectClearPipeline
import torch

# This triggers automatic download and caching
pipe = ObjectClearPipeline.from_pretrained_with_custom_modules(
    "jixin0101/ObjectClear",
    torch_dtype=torch.float16,
    apply_attention_guided_fusion=True
)