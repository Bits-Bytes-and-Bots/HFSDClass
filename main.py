# This is a sample Python script.
import PIL.ImageShow


class SDClassUnit1:

    def __init__(self):
        import matplotlib.pyplot as plt
        self.setup_env()
        self.login()

        # get the functions from part 1
        self.show_images, self.make_grid, self.device = self.part1()

        # self.potato_image = self.part2()
        # plt.imshow(self.potato_image)
        # plt.show()

        self.batch_size = 64
        self.samples = 8
        #self.image_size = 128 + 32
        self.image_size = 32

        # self.part3_image = self.part3()
        # plt.imshow(self.part3_image)
        # plt.show()

        self.part4_image, self.part4_dataloader, self.part4_train_dataloader = self.part4(image_size=self.image_size,
                                                                                          batch_size=self.batch_size,
                                                                                          samples=self.samples)
        PIL.ImageShow.show(self.part4_image)

        self.part5_image, self.part5_noisey_dataloader, self.part5_timesteps = self.part5(image_size=self.image_size,
                                                                                          batch_size=self.batch_size,
                                                                                          samples=self.samples)
        PIL.ImageShow.show(self.part5_image)

        self.part6_model, self.part6_model_prediction = self.part6(image_size=self.image_size,
                                                                   batch_size=self.batch_size, samples=self.samples)
        self.part7_model, self.part7_model_prediction = self.part7(image_size=self.image_size,
                                                                   batch_size=self.batch_size, samples=self.samples)

    @staticmethod
    def setup_env():
        import os
        # execute the command "pip install -qq -U diffusers datasets transformers accelerate ftfy"
        # to install the required packages
        os.system("pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117")
        os.system("pip install -qq -U diffusers datasets transformers accelerate ftfy torchvision matplotlib")

    def login(self):
        from huggingface_hub import login
        # notebook_login()
        # execute the command "huggingface-cli login" to login
        # os.system("huggingface-cli login")
        login("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", True)

    def part1(self):
        print("**PART 1**")
        import numpy as np
        try:
            import torch
            import torch.nn.functional as F
        except ModuleNotFoundError as e:
            # raise the error with message explaining that installing torch apparently did not work
            raise ModuleNotFoundError("Installing torch apparently did not work") from e
        from matplotlib import pyplot as plt
        from PIL import Image

        def show_images(x):
            import torchvision
            """Given a batch of images x, make a grid and convert to PIL"""
            x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
            grid = torchvision.utils.make_grid(x)
            grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
            grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
            return grid_im

        def make_grid(images, size=64):
            """Given a list of PIL images, stack them together into a line for easy viewing"""
            output_im = Image.new("RGB", (size * len(images), size))
            for i, im in enumerate(images):
                output_im.paste(im.resize((size, size)), (i * size, 0))
            return output_im

        # Mac users may need device = 'mps' (untested)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return show_images, make_grid, device

    def part2(self):
        print("**PART 2**")
        from diffusers import StableDiffusionPipeline
        import torch

        # Check out https://huggingface.co/sd-dreambooth-library for loads of models from the community
        model_id = "sd-dreambooth-library/mr-potato-head"

        # Load the pipeline
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(self.device)

        prompt = "an abstract oil painting of sks mr potato head being eaten by a (dragon:1.5), style of fantasy art"
        image = pipe(prompt, num_inference_steps=75, guidance_scale=7.5).images[0]
        return image

    def part3(self):
        print("**PART 3**")
        from diffusers import DDPMPipeline

        print("Load the butterfly pipeline")
        butterfly_pipeline = DDPMPipeline.from_pretrained(
            "johnowhitaker/ddpm-butterflies-32px"
        ).to(self.device)

        print("Generate a batch of images")
        images = butterfly_pipeline(batch_size=self.batch_size, num_inference_steps=1000).images

        return self.make_grid(images)

    def part4(self, image_size=32, batch_size=64, samples=32):
        print("**PART 4**")
        import torchvision, torch
        import PIL
        from datasets import load_dataset
        from torchvision import transforms

        dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")

        # Define data augmentations
        preprocess = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),  # Resize
                transforms.RandomHorizontalFlip(),  # Randomly flip (data augmentation)
                transforms.RandomAutocontrast(),  # Randomly adjust contrast (data augmentation)
                transforms.ToTensor(),  # Convert to tensor (0, 1)
                transforms.Normalize([0.5], [0.5]),  # Map to (-1, 1)
            ]
        )

        def transform(examples):
            images = [preprocess(image.convert("RGB")) for image in examples["image"]]
            return {"images": images}

        dataset.set_transform(transform)

        # Create a dataloader from the dataset to serve up the transformed images in batches
        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        xb = next(iter(train_dataloader))["images"].to(self.device)[:batch_size]
        print("X shape:", xb.shape)
        return self.show_images(xb).resize((8 * image_size, round(samples / 8) * image_size)
                                           , resample=PIL.Image.Resampling.NEAREST), xb, train_dataloader

    def part5(self, image_size, batch_size, samples):
        from diffusers import DDPMScheduler
        from matplotlib import pyplot as plt
        import torch
        import PIL

        noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

        plt.plot(noise_scheduler.alphas_cumprod.cpu() ** 0.5, label="sqrt_alpha_prod")
        plt.plot(
            (1 - noise_scheduler.alphas_cumprod.cpu()) ** 0.5, label="sqrt_one_minus_alpha_prod"
        )
        plt.legend()
        plt.show()

        timesteps = torch.linspace(0, 999, batch_size).long().to(self.device)
        noise = torch.randn_like(self.part4_dataloader)
        noisy_xb = noise_scheduler.add_noise(self.part4_dataloader, noise, timesteps)
        print("Noisy X shape", noisy_xb.shape)
        return self.show_images(noisy_xb).resize((8 * image_size, round(samples / 8) * image_size)
                                                 , resample=PIL.Image.Resampling.NEAREST), noisy_xb, timesteps

    def part6(self, image_size, batch_size, samples):
        from diffusers import UNet2DModel
        import torch
        """
        UNet2DModel is a generative model that takes in a batch of images and returns a batch of images.
        DownBlock and UpBlock are the building blocks of the UNet2DModel.
        DownBlock is a convolutional block that downsamples the image by a factor of 2.
        UpBlock is a convolutional block that upsamples the image by a factor of 2.
        AttnDownBlock and AttnUpBlock are the building blocks of the UNet2DModel with attention.
        AttnDownBlock is a convolutional block that downsamples the image by a factor of 2 with attention.
        AttnUpBlock is a convolutional block that upsamples the image by a factor of 2 with attention.        
        """

        # Create a model
        model = UNet2DModel(
            sample_size=image_size,  # the target image resolution
            in_channels=3,  # the number of input channels, 3 for RGB images
            out_channels=3,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(64, 128, 128, 256),  # More channels -> more parameters
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",  # a regular ResNet upsampling block
            ),
        )
        model.to(self.device)

        with torch.no_grad():
            model_prediction = model(self.part5_noisey_dataloader, self.part5_timesteps).sample

        print(model_prediction.shape)
        return model, model_prediction

    def part7(self, image_size, batch_size, samples):
        """training loop"""
        import torch
        # import the functional interface
        import torch.nn.functional as F
        import numpy as np

        from diffusers import UNet2DModel, DDPMScheduler

        # Set the noise scheduler
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
        )
        optimizer = torch.optim.AdamW(self.part6_model.parameters(), lr=4e-4)

        losses = []

        for epoch in range(50):
            print(f'Epoch {epoch}')
            for step, batch in enumerate(self.part4_train_dataloader):
                print(f'Step {step}')
                clean_images = batch["images"].to(self.device)
                # Sample noise to add to the images
                noise = torch.randn(clean_images.shape).to(clean_images.device)
                bs = clean_images.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device
                ).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

                # Get the model prediction
                noise_pred = self.part6_model(noisy_images, timesteps, return_dict=False)[0]

                # Calculate the loss
                loss = F.mse_loss(noise_pred, noise)
                loss.backward(loss)
                losses.append(loss.item())

                # Update the model parameters with the optimizer
                optimizer.step()
                optimizer.zero_grad()

            if (epoch + 1) % 1 == 0 or epoch == 0:
                loss_last_epoch = sum(losses[-len(self.part4_train_dataloader):]) / len(self.part4_train_dataloader)
                print(f"Epoch:{epoch + 1}, loss: {loss_last_epoch}")

                import matplotlib.pyplot as plt
                noisy_images = noisy_images.detach().cpu().numpy()
                clean_images = clean_images.detach().cpu().numpy()
                noise_pred = noise_pred.detach().cpu().numpy()
                # reshape the images to (batch_size, height, width, channels)
                noisy_images = noisy_images.transpose(0, 2, 3, 1)
                clean_images = clean_images.transpose(0, 2, 3, 1)
                noise_pred = noise_pred.transpose(0, 2, 3, 1)
                # normalize the images to [0, 1]
                noisy_images = (noisy_images - noisy_images.min()) / (noisy_images.max() - noisy_images.min())
                clean_images = (clean_images - clean_images.min()) / (clean_images.max() - clean_images.min())
                noise_pred = (noise_pred - noise_pred.min()) / (noise_pred.max() - noise_pred.min())

                # plot the images
                fig, axs = plt.subplots(3, 3, figsize=(10, 10))
                for i in range(3):
                    axs[i, 0].imshow(noisy_images[i])
                    axs[i, 1].imshow(clean_images[i])
                    axs[i, 2].imshow(noise_pred[i])
                plt.show()

                fig, axs = plt.subplots(2, 1, figsize=(12, 4))
                # plot losses and its trend
                axs[0].plot(losses[-1000:] or losses)
                axs[0].set_title("Loss")
                axs[0].set_xlabel("Step")
                axs[0].set_ylabel("Loss")
                axs[0].plot(np.convolve(losses[-1000:] or losses, np.ones(batch_size*2) / (batch_size*2), mode="valid"))

                axs[1].plot(np.log(losses[-1000:] or losses))
                axs[1].set_title("Log Loss")
                axs[1].set_xlabel("Step")
                axs[1].set_ylabel("Log Loss")
                axs[1].plot(np.convolve(np.log(losses[-1000:] or losses), np.ones(batch_size*2) / (batch_size*2), mode="valid"))
                plt.show()

        return

    def part8(self, image_size, batch_size, samples):
        """Generating images via pipeline"""

        from diffusers import DDPMPipeline, DDPMScheduler
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
        )
        image_pipe = DDPMPipeline(unet=self.part6_model, scheduler=noise_scheduler)
        pipeline_output = image_pipe()
        self.show_images(pipeline_output.images[0])

    def part9(self,image_size, batch_size, samples):
        """Generating images via sampling loop"""
        import torch
        from diffusers import DDPMPipeline, DDPMScheduler
        sample = torch.randn(8, 3, image_size, image_size).to(self.device)
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
        )
        for i, t in enumerate(noise_scheduler.timesteps):
            # Get model pred
            with torch.no_grad():
                residual = self.part6_model(sample, t).sample
            # Update sample with step
            sample = noise_scheduler.step(residual, t, sample).prev_sample
        self.show_images(sample)


if __name__ == '__main__':
    my_sd_class = SDClassUnit1()
