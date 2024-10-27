# COMP9444 Group Assignment

## Group Member
- Breeze Chen
- Yolanda Song
- Skyler Gu
- Joffrey Ji
- Yixin Kang

## Project 15: Fine-tune Multi-modal LLaVA Vision and Language Models


## Timeline:
base model tested


Expanding inputs for image tokens in LLaVa-NeXT should be done in processing. Please add `patch_size` and `vision_feature_select_strategy` to the model's processing config or set directly with `processor.patch_size = {{patch_size}}` and processor.vision_feature_select_strategy = 
{{vision_feature_select_strategy}}`. Using processors without these attributes in the config is deprecated and will throw an error in v4.47.Setting `pad_token_id` to `eos_token_id`:None for open-end generation.