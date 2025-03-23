import torch
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# ===== 引入模型定义和scheduler =====
from ddm.net import UNet, DiffusionModel, offset_cosine_diffusion_schedule

# ===== 推理配置 =====
IMAGE_SIZE = 64
NUM_SAMPLES = 8
DIFFUSION_STEPS = 20
CKPT_PATH = "./checkpoint/checkpoint.ckpt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_inference():
    # 初始化模型
    ddm = DiffusionModel(UNet, offset_cosine_diffusion_schedule, EMA=0.999).to(DEVICE)

    # 加载权重
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"模型权重文件未找到: {CKPT_PATH}")
    ddm.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    ddm.eval()

    # 执行图像生成
    with torch.no_grad():
        ddm.train() # FIXME eval mode does not work
        generated_images = ddm.generate(
            num_images=NUM_SAMPLES,
            diffusion_steps=DIFFUSION_STEPS
        )

    # 可视化（可选）
    fig = plt.figure(figsize=(12, 4))
    for i in range(NUM_SAMPLES):
        ax = fig.add_subplot(2, NUM_SAMPLES // 2, i + 1)
        ax.axis("off")
        ax.imshow(generated_images[i].permute(1, 2, 0).cpu().numpy())
    plt.tight_layout()
    plt.show()

    # 保存到本地
    os.makedirs("output", exist_ok=True)
    out_path = "./output/generated_samples.png"
    save_image(generated_images, out_path, nrow=4)
    print(f"图像已保存到: {out_path}")


if __name__ == "__main__":
    run_inference()
