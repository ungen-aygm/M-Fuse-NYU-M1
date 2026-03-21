#!/usr/bin/env zsh
echo "--- NYU v2 Dataset Preparation ---"
echo "Note: 実行時はvenv環境で行ってください。"
# データファイルは約2.8GBあるので、サーバ負荷も考え何度も実行しない
NYU_V2_URL="http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
MAT_40_URL="https://github.com/ankurhanda/nyuv2-meta-data/raw/refs/heads/master/classMapping40.mat"
MAT_13_URL="https://github.com/ankurhanda/nyuv2-meta-data/raw/refs/heads/master/class13Mapping.mat"
DATASETS="./datasets"
mkdir -p "$DATASETS"
# ダウンロード
TARGET_MAT="$DATASETS/nyu_depth_v2_labeled.mat"
TARGET_MAT_40="$DATASETS/classMapping40.mat"
TARGET_MAT_13="$DATASETS/class13Mapping.mat"
[ ! -f "$TARGET_MAT" ] && curl -L -o "$TARGET_MAT" "$NYU_V2_URL"
[ ! -f "$TARGET_MAT_40" ] && curl -L -o "$TARGET_MAT_40" "$MAT_40_URL"
[ ! -f "$TARGET_MAT_13" ] && curl -L -o "$TARGET_MAT_13" "$MAT_13_URL"
sleep 0.1

# データ準備
.venv/bin/python test/prepare.py
# データ変換(numpy)
.venv/bin/python test/preprocess.py
sleep 0.1

# あれば、データ削除
rm -rf "$DATASETS"/nyuv2/{train,test}/{image,depth,label}
echo "--- finished, Prepare Dataset. ---"
echo "--- check train.py: is_load_model=False"
echo "	Run Train:$ python3 train.py"
sleep 0.1