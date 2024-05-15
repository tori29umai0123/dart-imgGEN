import torch
import zipfile
import os
import shutil
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import AutoPipelineForText2Image
from huggingface_hub import HfApi, Repository, upload_file, create_repo


# 画像生成用のプロンプトを生成する関数
def get_prompt(model, tokenizer):
    random_list = ["no humans, scenery", "1girl", "1boy", ""]
    random_choice = random.choice(random_list)
    prompt = (
        f"<|bos|>"
        "<copyright></copyright>"
        "<character></character>"
        "<|rating:general|><|aspect_ratio:square|><|length:short|>"
        f"<general>{random_choice}"
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
    negative_prompt = "nsfw, text, logo, bad composition, lowres, low quality, worst quality, low effort, watermark, signature, ugly, poorly drawn"
    image = pipe(prompt=prompt, negative_prompt=negative_prompt).images[0]
    return image

def save_files(image, prompt, index, image_dir, caption_dir):
    formatted_index = str(index + 1).zfill(5)
    image_path = os.path.abspath(f"{image_dir}/{formatted_index}.png")
    image.save(image_path)
    caption_path = os.path.abspath(f"{caption_dir}/{formatted_index}.txt")
    with open(caption_path, mode='w') as f:
        f.write(prompt)
    print(f"Saved image and caption for index {formatted_index}")
    return image_path, caption_path

def create_zip(image_dir, caption_dir, zip_name):
    zip_path = os.path.abspath(zip_name)
    with zipfile.ZipFile(zip_path, 'w') as z:
        # Save images
        for file in os.listdir(image_dir):
            file_path = os.path.join(image_dir, file)
            if os.path.isfile(file_path):  # Check if it's a file
                z.write(file_path, arcname=os.path.join('images', file))
                os.remove(file_path)
        # Save captions
        for file in os.listdir(caption_dir):
            file_path = os.path.join(caption_dir, file)
            if os.path.isfile(file_path):  # Check if it's a file
                z.write(file_path, arcname=os.path.join('captions', file))
                os.remove(file_path)
    print(f"Created zip file {zip_path}")
    return zip_path

def check_repository_access(repo_name, token):
    api = HfApi()
    try:
        repo_info = api.repo_info(repo_name, token=token, repo_type='dataset')
        print("Repository access verified.")
        return True
    except Exception as e:
        print(f"Unable to access repository: {e}")
        return False

def upload_to_hf(zip_file, repo_name, token):
    try:
        create_repo(repo_name, token=token, repo_type='dataset', exist_ok=True)
        repo_local_path = os.path.join(os.getcwd(), repo_name.split('/')[-1])
        
        # Clone the repo if it doesn't exist locally
        if not os.path.exists(repo_local_path):
            repo_local_path = os.path.abspath(repo_name.split('/')[-1])
            Repository(repo_local_path, clone_from=f"https://huggingface.co/datasets/{repo_name}", use_auth_token=token)
        
        # Move the zip file to the repository directory
        shutil.copy(zip_file, repo_local_path)
        repo_zip_path = os.path.join(repo_local_path, os.path.basename(zip_file))
        
        # Upload the file
        upload_file(path_or_fileobj=repo_zip_path, path_in_repo=os.path.basename(repo_zip_path), repo_id=repo_name, token=token, repo_type='dataset')
        os.remove(zip_file)
        print(f"Uploaded {zip_file} to Hf dataset repository {repo_name}")
    except Exception as e:
        print(f"Error during upload: {e}")
        exit(1)

def get_last_processed_index(repo_local_path):
    zip_files = [f for f in os.listdir(repo_local_path) if f.startswith('images_') and f.endswith('.zip')]
    if not zip_files:
        return -1
    zip_files.sort()
    last_zip = zip_files[-1]
    last_index = int(last_zip.split('_')[-1].split('.')[0]) * 1000 - 1
    return last_index

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

    repo_name = input("Please enter your Hugging Face repository name: ")
    token = input("Please enter your Hugging Face API token: ")

    if not check_repository_access(repo_name, token):
        print("Access denied or invalid repository. Stopping execution.")
        exit(1)

    repo_local_path = os.path.join(os.getcwd(), repo_name.split('/')[-1])
    try:
        # Try to clone the repository
        if not os.path.exists(repo_local_path):
            print(f"Cloning repository {repo_name}...")
            Repository(repo_local_path, clone_from=f"https://huggingface.co/datasets/{repo_name}", use_auth_token=token)
    except Exception as e:
        print(f"Repository not found, creating new repository {repo_name}...")
        create_repo(repo_name, token=token, repo_type='dataset', exist_ok=True)
        Repository(repo_local_path, clone_from=f"https://huggingface.co/datasets/{repo_name}", use_auth_token=token)
    
    last_index = get_last_processed_index(repo_local_path)

    for idx in range(last_index + 1, 30000):
        prompt = get_prompt(model, tokenizer)
        image = make_image(pipe, prompt)
        image_path, caption_path = save_files(image, prompt, idx, image_dir, caption_dir)

        if (idx + 1) % 1000 == 0:
            zip_name = f"./data/images_{idx // 1000 + 1}.zip"
            zip_path = create_zip(image_dir, caption_dir, zip_name)
            upload_to_hf(zip_path, repo_name, token)
