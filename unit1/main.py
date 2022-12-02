# This is a sample Python script.
import os

import PIL.ImageShow


class SDClassUnit1:

    def __init__(self):
        # provide some class variables for storing configuration, data, models and results
        # these are created here and will be modified by the different parts of the unit exercises

        self.batch_size = 64
        self.samples = 8
        self.image_size = 32
        self.xbatch = None
        self.noise_batch = None
        self.noisy_xbatch = None
        self.dataloader = None

        # set to true to skip sampling in slow inference etc.
        self.fast = True
        self.training_epochs = 1

        self.step1()
        self.step2()
        self.step3()
        self.step4()  # define the model
        self.step5()  # train the model
        self.step6()  # sample from the model

    def step1(self):
        import torch
        from PIL import Image
        import numpy as np

        def show_images(x):
            import torchvision
            """Given a batch of images x, make a grid and convert to PIL"""
            x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
            grid = torchvision.utils.make_grid(x)
            grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
            grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
            return grid_im

        self.show_images = show_images

        def make_grid(images, size=64):
            """Given a list of PIL images, stack them together into a line for easy viewing"""
            output_im = Image.new("RGB", (size * len(images), size))
            for i, im in enumerate(images):
                output_im.paste(im.resize((size, size)), (i * size, 0))
            return output_im

        self.make_grid = make_grid

        def setup_env():
            import os
            # execute the command "pip install -qq -U diffusers datasets transformers accelerate ftfy"
            # to install the required packages
            os.system(
                "pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117")
            os.system("pip install -qq -U diffusers datasets transformers accelerate ftfy torchvision matplotlib")

        def login():
            from huggingface_hub import login
            token = os.getenv("HF_TOKEN")
            login(token, True)

        def setup_git_lfs():
            print("git LFS is required for this project please install it via" + \
                  "https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage?platform=windows")

        def dreambooth():
            print("**Dreambooth: A Sneak Peak at What's to Come**")
            from diffusers import StableDiffusionPipeline
            import torch

            # Check out https://huggingface.co/sd-dreambooth-library for loads of models from the community
            model_id = "sd-dreambooth-library/mr-potato-head"

            # Load the pipeline
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(self.device)

            prompt = "an abstract oil painting of sks mr potato head being eaten by a (dragon:1.5), style of fantasy art"
            image = pipe(prompt, num_inference_steps=25, guidance_scale=7.5).images[0]
            image.show()

        def MVP():
            from diffusers import DDPMPipeline

            # Load the butterfly pipeline
            butterfly_pipeline = DDPMPipeline.from_pretrained(
                "johnowhitaker/ddpm-butterflies-32px"
            ).to(self.device)

            # Create images
            images = butterfly_pipeline(batch_size=3, num_inference_steps=50 if self.fast else 1000).images

            # View the result
            self.make_grid(images).show("butterfly pipeline")

        setup_env()
        login()
        print("""Next, head over to https://huggingface.co/settings/tokens and create an access token with write
         permission if you don't already have one:""")
        setup_git_lfs()

        # Mac users may need device = 'mps' (untested)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dreambooth()
        MVP()
        return

    def step2(self):
        def download_dataset():
            print("**Download a training dataset**")
            import torchvision, torch
            import PIL
            from datasets import load_dataset
            from torchvision import transforms

            dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")

            # Define data augmentations
            preprocess = transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),  # Resize
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
            self.dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True
            )

        def show_batch_from_dataloader():
            self.xbatch = next(iter(self.dataloader))["images"].to(self.device)[:self.batch_size]
            self.show_images(self.xbatch).show()

        download_dataset()
        show_batch_from_dataloader()

    def step3(self):
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

        # One with too little noise added:
        # noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.001, beta_end=0.004)
        # The 'cosine' schedule, which may be better for small image sizes:
        # noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')

        timesteps = torch.linspace(0, 999, self.batch_size).long().to(self.device)
        self.noise_batch = torch.randn_like(self.xbatch)

        self.noisy_xbatch = noise_scheduler.add_noise(self.xbatch, self.noise_batch, timesteps)
        print("Noisy X shape", self.noisy_xbatch.shape)
        self.show_images(self.noisy_xbatch).show()

        self.timesteps = timesteps

        return

    def step4(self):
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
        self.model = UNet2DModel(
            sample_size=self.image_size,  # the target image resolution
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
        self.model.to(self.device)

        with torch.no_grad():
            model_prediction = self.model(self.noisy_xbatch, self.timesteps).sample

        print(model_prediction.shape)
        self.show_images(model_prediction).show()
        return

    def step5(self):
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
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=4e-4)

        losses = []

        for epoch in range(self.training_epochs):
            print(f'Epoch {epoch}')
            for step, batch in enumerate(self.dataloader):
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
                noise_pred = self.model(noisy_images, timesteps, return_dict=False)[0]

                # Calculate the loss
                loss = F.mse_loss(noise_pred, noise)
                loss.backward(loss)
                losses.append(loss.item())

                # Update the model parameters with the optimizer
                optimizer.step()
                optimizer.zero_grad()

            if (epoch + 1) % 1 == 0 or epoch == 0:
                loss_last_epoch = sum(losses[-len(self.dataloader):]) / len(self.dataloader)
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
                axs[0].plot(
                    np.convolve(losses[-1000:] or losses, np.ones(self.batch_size * 2) / (self.batch_size * 2),
                                mode="valid"))

                axs[1].plot(np.log(losses[-1000:] or losses))
                axs[1].set_title("Log Loss")
                axs[1].set_xlabel("Step")
                axs[1].set_ylabel("Log Loss")
                axs[1].plot(
                    np.convolve(np.log(losses[-1000:] or losses), np.ones(self.batch_size * 2) / (self.batch_size * 2),
                                mode="valid"))
                plt.show()

        return

    def step6(self):
        def pipeline_generation():
            print("""Generating images via pipeline""")

            from diffusers import DDPMPipeline, DDPMScheduler
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
            )
            image_pipe = DDPMPipeline(unet=self.model, scheduler=noise_scheduler)
            pipeline_output = image_pipe()
            image = pipeline_output.images[0]
            image.show()

        def sampling_loop():
            """Generating images via sampling loop"""
            import torch
            from diffusers import DDPMPipeline, DDPMScheduler
            sample = torch.randn(8, 3, self.image_size, self.image_size).to(self.device)
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
            )
            for i, t in enumerate(noise_scheduler.timesteps):
                # Get model pred
                with torch.no_grad():
                    residual = self.model(sample, t).sample
                # Update sample with step
                sample = noise_scheduler.step(residual, t, sample).prev_sample
            self.show_images(sample).show()

        pipeline_generation()
        sampling_loop()


if __name__ == '__main__':
    my_sd_class = SDClassUnit1()
