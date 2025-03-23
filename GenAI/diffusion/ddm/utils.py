import matplotlib.pyplot as plt
import torch


# run training and plot generated images periodically
def display_imgs(
    images, n=10, size=(20, 3), cmap="gray_r", as_type="float32", save_to=None
):
    """
    Displays n random images from each one of the supplied arrays.
    """
    if images.max() > 1.0:
        images = images / 255.0
    elif images.min() < 0.0:
        images = (images + 1.0) / 2.0

    plt.figure(figsize=size)
    for i in range(n):
        _ = plt.subplot(1, n, i + 1)
        plt.imshow(images[i].astype(as_type), cmap=cmap)
        plt.axis("off")

    if save_to:
        plt.savefig(save_to)
        print(f"\nSaved to {save_to}")

    # plt.show()

class ImageGenerator:
    
    def __init__(self, model, num_img, 
                 plot_diffusion_steps=20,
                 save_to="./output/generated_img_%03d.png"):
        self.model = model
        self.num_img = num_img
        self.plot_diffusion_steps = plot_diffusion_steps
        self.save_to = save_to
        
    def on_epoch_end(self, epoch):
        with torch.no_grad():
            generated_images = self.model.generate(
                num_images=self.num_img,
                diffusion_steps=self.plot_diffusion_steps,
            ).cpu().permute(0,2,3,1).numpy()
            print(generated_images.shape)
            display_imgs(
                generated_images,
                save_to=self.save_to % (epoch),
            )
