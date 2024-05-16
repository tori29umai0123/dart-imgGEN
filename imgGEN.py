import torch
import zipfile
import os
import shutil
from diffusers import AutoPipelineForText2Image
from huggingface_hub import HfApi, Repository, upload_file, create_repo

# 画像生成機能
def make_image(pipe, prompt, negative_prompt, width, height):
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, width=width, height=height).images[0]
    return image

# 画像とキャプションのファイルを保存
def save_files(image, prompt, index, image_dir, caption_dir, image_filename):
    image_path = os.path.abspath(f"{image_dir}/{image_filename}")
    image.save(image_path)
    caption_path = os.path.abspath(f"{caption_dir}/{index+1}.txt")
    with open(caption_path, mode='w') as f:
        f.write(prompt)
    print(f"保存された画像とキャプションのインデックス: {index+1}")

# Hugging Faceリポジトリへのアクセスを確認
def check_repository_access(repo_name, token):
    api = HfApi()
    try:
        repo_info = api.repo_info(repo_name, token=token, repo_type='dataset')
        print("Repository access verified.")
        return True
    except Exception as e:
        print(f"Unable to access repository: {e}")
        return False

# 画像とキャプションをZIPファイルに圧縮して保存
def create_zip(image_dir, caption_dir, zip_name):
    zip_path = os.path.abspath(zip_name)
    with zipfile.ZipFile(zip_path, 'w') as z:
        for file in os.listdir(image_dir):
            file_path = os.path.join(image_dir, file)
            if os.path.isfile(file_path):
                z.write(file_path, arcname=os.path.join('images', file))
                os.remove(file_path)
        for file in os.listdir(caption_dir):
            file_path = os.path.join(caption_dir, file)
            if os.path.isfile(file_path):
                z.write(file_path, arcname=os.path.join('captions', file))
                os.remove(file_path)
    print(f"作成されたZIPファイル {zip_path}")
    return zip_path

# Hugging Faceのリポジトリにアップロード
def upload_to_hf(zip_file, repo_name, token):
    create_repo(repo_name, token=token, repo_type='dataset', exist_ok=True)
    repo_local_path = os.path.join(os.getcwd(), repo_name.split('/')[-1])
    if not os.path.exists(repo_local_path):
        Repository(repo_local_path, clone_from=f"https://huggingface.co/datasets/{repo_name}", use_auth_token=token)
    shutil.copy(zip_file, repo_local_path)
    repo_zip_path = os.path.join(repo_local_path, os.path.basename(zip_file))
    upload_file(path_or_fileobj=repo_zip_path, path_in_repo=os.path.basename(repo_zip_path), repo_id=repo_name, token=token, repo_type='dataset')
    os.remove(zip_file)
    print(f"Hfデータセットリポジトリ {repo_name} に {zip_file} をアップロードしました。")

# 最後に処理されたインデックスを取得
def get_last_processed_index(repo_local_path):
    zip_files = [f for f in os.listdir(repo_local_path) if f.startswith('images_') and f.endswith('.zip')]
    if not zip_files:
        return -1
    zip_files.sort()
    last_zip = zip_files[-1]
    last_index = int(last_zip.split('_')[-1].split('.')[0]) * 1000 - 1
    return last_index

# プロンプトファイルを処理して画像生成
def process_prompts(prompt_txt_path, pipe, image_dir, caption_dir, repo_name, token):
    with open(prompt_txt_path, 'r') as file:
        prompts = file.readlines()

    # 最後に処理されたインデックスを取得
    repo_local_path = os.path.join(os.getcwd(), repo_name.split('/')[-1])
    last_index = get_last_processed_index(repo_local_path)
    print(f"Starting from index: {last_index + 1}")

    total_prompts = len(prompts)
    for index, prompt in enumerate(prompts[last_index + 1:], start=last_index + 1):
        prompt_details = prompt.strip().split(' --')
        image = make_image(pipe, prompt_details[0], 'Negative Prompt', int(prompt_details[2][3:]), int(prompt_details[3][3:]))
        save_files(image, prompt_details[0], index, image_dir, caption_dir, prompt_details[4][3:])

        # 1000ごとまたは最後のプロンプトの場合ZIPにまとめる
        if (index + 1) % 1000 == 0 or index == total_prompts - 1:
            zip_name = f"./data/images_{index // 1000 + 1}.zip"
            zip_path = create_zip(image_dir, caption_dir, zip_name)
            upload_to_hf(zip_path, repo_name, token)

if __name__ == '__main__':
    pipe = AutoPipelineForText2Image.from_pretrained(
        "cagliostrolab/animagine-xl-3.1",
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")
    image_dir = "./data/image"
    caption_dir = "./data/caption"
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(caption_dir, exist_ok=True)
    prompt_txt_path = "dart_prompts.txt"
    repo_name = input("Hugging Faceリポジトリ名を入力してください: ")
    token = input("Hugging Face APIトークンを入力してください: ")
    if not check_repository_access(repo_name, token):
        print("リポジトリへのアクセスが拒否されたか、無効です。実行を停止します。")
        exit(1)
    process_prompts(prompt_txt_path, pipe, image_dir, caption_dir, repo_name, token)
