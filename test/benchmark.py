import torch
from models.late_fusion import LateFusion
from models.vit import semantic_encoder
from models.unet import UNet as unet
from config import Config
import time
from tqdm import tqdm
from utils.functions import *

def main():
    config = Config()
    # 1. デバイス設定 (M1 GPUを使用)
    device = torch.device(config.device)
    set_seed(config.SEED)
    
    # 2. モデル準備
    model = LateFusion(semantic_encoder, unet).to(device) # 重みロード済みのインスタンス

    # 推論モード
    model.eval()
    # 3. ダミーデータの作成 (入力サイズを合わせる)
    dummy_rgb = torch.randn(1, 3, config.TARGET_SIZE[0], config.TARGET_SIZE[1]).to(device)
    dummy_depth = torch.randn(1, 1, config.TARGET_SIZE[0], config.TARGET_SIZE[1]).to(device)

    # 4. ウォームアップ (最初の数回はGPUの初期化で遅いため)
    print("モデルの初期化中...")
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_rgb, dummy_depth)
    torch.mps.synchronize() # M1 GPUの処理完了を待機

    # 5. 本番計測
    print("推論時間測定中...")
    num_samples = 44
    start_time = time.time()

    loop = tqdm(range(num_samples), desc="Measuring inference time")
    for _ in enumerate(loop):
        with torch.no_grad():
            _ = model(dummy_rgb, dummy_depth)
        torch.mps.synchronize()

    end_time = time.time()

    # 結果表示
    avg_time = (end_time - start_time) / num_samples
    fps = 1.0 / avg_time

    print(f"------------------------------")
    print(f"平均推論時間: {avg_time*1000:.2f} ms")
    print(f"推定スループット/データ処理: {fps:.2f} FPS")
    print(f"------------------------------")
if __name__ == "__main__":
    main()