from huggingface_hub import login, HfApi
from transformers import AutoModelForImageTextToText,AutoProcessor
from peft import PeftModel
login(token="hf_xxxxxxxxxxxxxxxx")
#Base model on your local filesystem
base_model_dir = "unsloth/Qwen2.5-VL-7B-Instruct"
base_model = AutoModelForImageTextToText.from_pretrained(base_model_dir)

#Adaptor directory on your local filesystem
## SET PATHS
adaptor_dir = "./vlm_finetuned_full_context_48_2"
merged_model = PeftModel.from_pretrained(base_model,adaptor_dir)
processor = AutoProcessor.from_pretrained(adaptor_dir, use_fast=False)

merged_model = merged_model.merge_and_unload()
merged_model.save_pretrained("./vlm_finetuned_full_context-Merged-Model_2/")
processor.save_pretrained("./vlm_finetuned_full_context-Merged-Model_2/")