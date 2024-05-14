import torch
import zipfile
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import AutoPipelineForText2Image
from huggingface_hub import HfApi, HfFolder, Repository

def get_prompt(model):
    prompt = (
        f"<|bos|>"
        f"<copyright></copyright>"
        f"<character></character>"
        f"<|rating:general|><|aspect_ratio:tall|><|length:long|>"
        f"<general>1girl"
    )
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
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
    return ", ".join([tag for tag in tokenizer.batch_decode(outputs[0], skip_special_tokens=True) if tag.strip() != ""])

def make_image(pipe, prompt):
    negative_prompt = "lowres, error, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, blurry"
    image = pipe(prompt=prompt, negative_prompt=negative_prompt).images[0]
    return image

def save_files(image, prompt, index, image_dir, caption_dir):
    image.save(f"{image_dir}/image_{index}.png")
    with open(f"{caption_dir}/caption_{index}.txt", mode='w') as f:
        f.write(prompt)
    print(f"Saved image and caption for index {index}")

def create_zip(folder_path, zip_name):
    with zipfile.ZipFile(zip_name, 'w') as z:
        for file in os.listdir(folder_path):
            z.write(os.path.join(folder_path, file), file)
    print(f"Created zip file {zip_name}")

def upload_to_hf(zip_file, repo_name):
    api = HfApi()
    token = HfFolder.get_token()
    repo_url = api.create_repo(repo_name, private=False, exist_ok=True, token=token)
    repo = Repository(repo_name, clone_from=repo_url, use_auth_token=token)
    repo.git_pull()
    repo.git_add(zip_file)
    repo.git_commit("Add new images")
    repo.git_push()
    print(f"Uploaded {zip_file} to Hf repository {repo_name}")

if __name__ == '__main__':
    MODEL_NAME = "p1atdev/dart-v2-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    
    pipe = AutoPipelineForText2Image.from_pretrained(
        "cagliostrolab/animagine-xl-3.1", 
        torch_dtype=torch.float16, 
        use_safetensors=True
    ).to("cuda")

    image_dir = "./data/image"
    caption_dir = "./data/caption"
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(caption_dir, exist_ok=True)
    
    for idx in range(30000):
        prompt = get_prompt(model, tokenizer)
        image = make_image(pipe, prompt)
        save_files(image, prompt, idx, image_dir, caption_dir)
        
        if (idx + 1) % 1000 == 0:
            zip_name = f"./data/images_{idx // 1000 + 1}.zip"
            create_zip("./data", zip_name)
            upload_to_hf(zip_name, "your_hf_organization/your_model_name")
