import os
import sys
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_prompt(model, num_prompts, rating, aspect_ratio, length, first_tag):
    prompt = f"<copyright></copyright><character></character>{rating}{aspect_ratio}{length}<general>{first_tag}"
    prompts = [prompt] * num_prompts
    inputs = tokenizer(prompts, return_tensors="pt").input_ids
    inputs = inputs.to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=100,
            max_new_tokens=128,
            num_beams=1,
        )
    decoded = []
    for i in range(num_prompts):
        output = outputs[i].cpu()
        tags = tokenizer.batch_decode(output, skip_special_tokens=True)
        prompt = ", ".join([tag for tag in tags if tag.strip() != ""])
        decoded.append(prompt)
    return decoded

def generate_prompts(model, output_file_path, NUM_PROMPTS_PER_VARIATION,  BATCH_SIZE):
    random.seed(42)
    prompts = []
    # 設定：寸法、アスペクト比、評価など
    DIMENSIONS = [(1024, 1024), (1152, 896), (896, 1152), (1216, 832), (832, 1216), (1344, 768), (768, 1344), (1536, 640), (640, 1536)]
    ASPECT_RATIO_TAGS = [
        "<|aspect_ratio:square|>",
        "<|aspect_ratio:wide|>",
        "<|aspect_ratio:tall|>",
        "<|aspect_ratio:wide|>",
        "<|aspect_ratio:tall|>",
        "<|aspect_ratio:wide|>",
        "<|aspect_ratio:tall|>",
        "<|aspect_ratio:ultra_wide|>",
        "<|aspect_ratio:ultra_tall|>",
    ]
    RATING_MODIFIERS = ["safe", "sensitive"]
    RATING_TAGS = ["<|rating:general|>", "<|rating:sensitive|>"]
    FIRST_TAGS = ["no humans", "1girl", "2girls", "1boy", "2boys", "1other", "2others"]
    YEAR_MODIFIERS = [None, "newest", "recent", "mid"]
    LENGTH_TAGS = ["<|length:very_short|>", "<|length:short|>", "<|length:medium|>", "<|length:long|>", "<|length:very_long|>"]
    QUALITY_MODIFIERS_AND_AESTHETIC = ["masterpiece", "best quality", "very aesthetic", "absurdres"]
    NEGATIVE_PROMPT = "nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]"

    # プロンプトの生成ループ
    for rating_modifier, rating_tag in zip(RATING_MODIFIERS, RATING_TAGS):
        negative_prompt = NEGATIVE_PROMPT
        if "nsfw" in rating_modifier:
            negative_prompt = negative_prompt.replace("nsfw, ", "")

        for dimension, aspect_ratio_tag in zip(DIMENSIONS, ASPECT_RATIO_TAGS):
            for first_tag in FIRST_TAGS:
                dart_prompts = []
                for i in range(0, NUM_PROMPTS_PER_VARIATION * len(YEAR_MODIFIERS), BATCH_SIZE):
                    length = random.choice(LENGTH_TAGS)
                    dart_prompts += get_prompt(model, BATCH_SIZE, rating_tag, aspect_ratio_tag, length, first_tag)

                num_prompts_for_each_year_modifier = NUM_PROMPTS_PER_VARIATION
                for j, year_modifier in enumerate(YEAR_MODIFIERS):
                    for prompt in dart_prompts[j * num_prompts_for_each_year_modifier : (j + 1) * num_prompts_for_each_year_modifier]:
                        prompt = prompt.replace("(", "\\(").replace(")", "\\)")
                        quality_modifiers = random.sample(QUALITY_MODIFIERS_AND_AESTHETIC, random.randint(0, 4))
                        quality_modifiers = ", ".join(quality_modifiers)
                        qm = f"{quality_modifiers}, " if quality_modifiers else ""
                        ym = f", {year_modifier}" if year_modifier else ""
                        image_index = len(prompts)
                        width, height = dimension
                        rm_filename = rating_modifier.replace(", ", "_")
                        ym_filename = year_modifier if year_modifier else "none"
                        ft_filename = first_tag.replace(" ", "")
                        image_filename = f"{image_index:08d}_{rm_filename}_{width:04d}x{height:04d}_{ym_filename}_{ft_filename}.webp"
                        final_prompt = f"{qm}{prompt}, {rating_modifier}{ym} --n {negative_prompt} --w {width} --h {height} --f {image_filename}"
                        prompts.append(final_prompt)

    # ファイルに出力
    with open(output_file_path, "w") as f:
        f.write("\n".join(prompts))

    print(f"完了しました。{len(prompts)}個のプロンプトが{output_file_path}に書き込まれました。")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python script.py <model_name> <output_file_path>")
    else:
        MODEL_NAME = sys.argv[1]
        output_file_path = sys.argv[2]
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
        model.to("cuda")
        NUM_PROMPTS_PER_VARIATION = 60
        BATCH_SIZE = 8
        generate_prompts(model, output_file_path, NUM_PROMPTS_PER_VARIATION,  BATCH_SIZE)
