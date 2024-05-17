import torch
import zipfile
import os
import shutil
from diffusers import StableDiffusionXLPipeline
from huggingface_hub import HfApi, Repository, upload_file, create_repo

# 画像生成機能を定義します。
def make_image(pipe, prompt, negative_prompt, width, height):
    # デバッグ出力
    print("Width:", width)
    print("Height:", height)
    print("Prompt:", prompt)
    print("Negative Prompt:", negative_prompt)
    # 画像を生成します。
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, width=width, height=height).images[0]
    return image

# 画像ファイルとキャプションファイルを保存します。
def save_files(image, prompt, image_filename, image_dir, caption_dir):
    # 画像ファイルのパスを設定します。
    image_path = os.path.abspath(f"{image_dir}/{image_filename}")
    # 画像を保存します。
    image.save(image_path)
    # キャプションファイルの名前を画像ファイルの名前から設定します。
    caption_filename = f"{os.path.splitext(image_filename)[0]}.txt"
    # キャプションファイルのパスを設定します。
    caption_path = os.path.abspath(f"{caption_dir}/{caption_filename}")
    # キャプションをファイルに書き込みます。
    with open(caption_path, mode='w') as f:
        f.write(prompt)
    print(f"画像を{image_path}に、キャプションを{caption_path}に保存しました。")

# Hugging Faceのリポジトリアクセスを確認します。
def check_repository_access(repo_name, token):
    api = HfApi()
    try:
        # リポジトリの情報を取得します。
        repo_info = api.repo_info(repo_name, token=token, repo_type='dataset')
        print("リポジトリアクセスを確認しました。")
        return True
    except Exception as e:
        print(f"リポジトリへのアクセスに失敗しました: {e}")
        return False

# 画像とキャプションをZIPファイルに圧縮して保存します。
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
    print(f"ZIPファイルを作成しました: {zip_path}")
    return zip_path

# Hugging FaceのリポジトリにZIPファイルをアップロードします。
def upload_to_hf(zip_file, repo_name, token):
    repo_local_path = os.path.join(os.getcwd(), repo_name.split('/')[-1])
    if not os.path.exists(repo_local_path):
        Repository(repo_local_path, clone_from=f"https://huggingface.co/datasets/{repo_name}", use_auth_token=token)
    shutil.copy(zip_file, repo_local_path)
    repo_zip_path = os.path.join(repo_local_path, os.path.basename(zip_file))
    upload_file(path_or_fileobj=repo_zip_path, path_in_repo=os.path.basename(repo_zip_path), repo_id=repo_name, token=token, repo_type='dataset')
    os.remove(zip_file)
    print(f"{zip_file}をHugging Faceのデータセットリポジトリ{repo_name}にアップロードしました。")

# プロンプトファイルを処理して画像を生成します。
def process_prompts(prompt_txt_path, pipe, image_dir, caption_dir, repo_name, token):
    with open(prompt_txt_path, 'r') as file:
        prompts = file.readlines()

    total_prompts = len(prompts)
    for index, prompt in enumerate(prompts):
        parts = prompt.strip().split(' --')
        main_prompt = parts[0]
        negative_prompt = parts[1].split(' ', 1)[1]
        width = int(parts[2].split(' ')[1])
        height = int(parts[3].split(' ')[1])
        image_filename = parts[4].split(' ')[1]
        
        image = make_image(pipe, main_prompt, negative_prompt, width, height)
        save_files(image, main_prompt, image_filename, image_dir, caption_dir)

        # プロンプトの終わりに達したか、または5000個ごとにZIPファイルを作成し、アップロードします。
        if index % 5000 == 9 or index == total_prompts - 1:
            zip_name = f"./data/images_{index // 5000 + 1}.zip"
            zip_path = create_zip(image_dir, caption_dir, zip_name)
            upload_to_hf(zip_path, repo_name, token)

# 実行ブロック
if __name__ == '__main__':
    # StableDiffusionXLPipelineを初期化します。
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "cagliostrolab/animagine-xl-3.1",
        torch_dtype=torch.float16,
    ).to("cuda")
    # 画像とキャプションの保存先ディレクトリを作成します。
    image_dir = "./data/image"
    caption_dir = "./data/caption"
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(caption_dir, exist_ok=True)
    # プロンプトファイルのパスを設定します。
    prompt_txt_path = "dart_prompts.txt"
    # ユーザーからリポジトリ名とAPIトークンを入力してもらいます。
    repo_name = input("Hugging Faceのリポジトリ名を入力してください: ")
    token = input("Hugging FaceのAPIトークンを入力してください: ")
    # リポジトリアクセスを確認します。
    if not check_repository_access(repo_name, token):
        print("アクセスが拒否されたか、無効なリポジトリです。実行を停止します。")
        exit(1)
    # プロンプトを処理して画像生成を実行します。
    process_prompts(prompt_txt_path, pipe, image_dir, caption_dir, repo_name, token)
