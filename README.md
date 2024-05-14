# dart-imgGEN
`dart-imgGEN`は、テキストプロンプトを使用して画像を生成するPythonスクリプトです。このスクリプトは、Hugging Faceの`transformers`と`diffusers`ライブラリを活用しており、GPU環境で最適に動作します。

## 環境構築

### 1. GPUサーバーのレンタル
適切なGPUサーバー（例：Vast.ai）をレンタルしてください。

### 2. Docker環境の構築
以下のDockerイメージを使用して、GPUサーバー上に環境を構築します。
https://hub.docker.com/layers/pytorch/pytorch/2.2.0-cuda12.1-cudnn8-devel/images/sha256-6353d21b66b5c00ae0f2683e447a0ba392157761f6dd0386a0dd648e0e5cc2de

### 3. 必要なライブラリのインストール
```
git clone https://github.com/tori29umai0123/dart-imgGEN.git
cd dart-imgGEN
pip install -r requirements.txt
```
### 4. Hugging Face アカウント設定

####1. アカウント作成
Hugging Faceにアクセスしてアカウントを作成します。

####2. データセットリポジトリの作成
Hugging Faceのアカウントで新しいデータセットのリポジトリを作成し、APIトークンを取得します。

### 5. スクリプトの実行
imgGEN.pyスクリプトをサーバー上で実行し、画像生成を開始します。出力された画像は指定されたHugging Faceに1000枚ごとにzipとしてアップロードされます
```
python imgGEN.py
```
一番最初に各種モデルのDLがはじまる。
DL後、アップロードするリポジトリ名と、アクセストークンを聞かれるので入力。
あとはひたすら画像生成を待つだけです。
