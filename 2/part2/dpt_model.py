"""Defines the VisualPrompting model (based on CLIP)"""
from pprint import pprint
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import warnings


def load_clip_to_cpu(cfg):
    """Loads CLIP model to CPU."""
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class DeepPromptCLIP(nn.Module):
    """Modified CLIP module to support prompting."""
    def __init__(self, args, dataset, template="This is a photo of {}"):
        super(DeepPromptCLIP, self).__init__()
        classnames = dataset.classes

        print(f"Loading CLIP (backbone: {args.arch})")
        clip_model = self.load_clip_to_cpu(args)
        clip_model.to(args.device)
        self.dtype = torch.float32 if args.device == "cpu" else torch.float16


        # hack to make model as float() (This is a CLIP hack)
        if args.device == "cpu":
            clip_model = clip_model.float()


        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        print("List of prompts:")
        pprint(prompts)

        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(args.device)

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # TODO: Write code to compute text features.
        # Hint: You can use the code from clipzs.py here!

        # Instructions:
        # - Given a list of prompts, compute the text features for each prompt.
        # - Return a tensor of shape (num_prompts, 512).

        clip_model.to(args.device)

        with torch.no_grad():
            # - Compute the text features (encodings) for each prompt.
            text_features = clip_model.encode_text(prompts)
            
            # - Normalize the text features.
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)


        #######################
        # END OF YOUR CODE    #
        #######################

        self.text_features = text_features
        self.clip_model = clip_model
        self.logit_scale = self.clip_model.logit_scale.exp().detach()

        self.injection_layer = args.injection_layer

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # TODO: Initialize the learnable deep prompt.
        # Hint: consider the shape required for the deep prompt to be compatible with the CLIP model 
        # Hint: CLIP uses different datatypes for CPU (float32) and GPU (float16)
        # Hint: use args.prompt_num to specify the number of deep prompts to use

        self.deep_prompt = nn.Parameter(torch.randn((args.prompt_num, 1, 768), dtype=self.dtype, device=args.device))
        print(f"self.deep_prompt.shape at initialization stage: {self.deep_prompt.shape}")
        
        #######################
        # END OF YOUR CODE    #
        #######################


    def forward(self, image):
        """Forward pass of the model."""
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # TODO: Implement the forward function. This is not exactly the same as
        # the model_inferece function in clipzs.py! Please see the steps below.

        # Steps:
        # - Compute the image features using the CLIP model (be sure use the custom_encode_image function).
        image_features = self.custom_encode_image(image)
        # - Normalize the image features.
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # - Compute similarity logits between the image features and the text features.
        # - You need to multiply the similarity logits with the logit scale (clip_model.logit_scale).
        similarity = self.logit_scale * image_features @ self.text_features.t()
        # - Return logits of shape (batch size, number of classes).
        return similarity

        #######################
        # END OF YOUR CODE    #
        #######################

    def custom_encode_image(self, x):
        """Encode image using CLIP model and add deep prompts."""
        # cf. https://github.com/openai/CLIP/blob/main/clip/model.py#L223

        print("Encoding the image using CLIP model...")
        #print(f"Initial x.shape (before encoding and class_embedding attachment): {x.shape}")
        x = x.type(self.clip_model.dtype)
        image_encoder = self.clip_model.visual
        
        x = image_encoder.conv1(x)  # shape = [*, width, grid, grid]
        #print(f"x.shape after conv1: {x.shape}")
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        #print(f"x.shape after reshape: {x.shape}")
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        #print("Adding the class embedding to the image...")
        #print(f"Initial x.shape: {x.shape}")
        x = torch.cat([image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        #print(f"Final x.shape: {x.shape}")
        #print(f"Adding the positional embedding to the image...")
        #print(f"Initial x.shape: {x.shape}")
        x = x + image_encoder.positional_embedding.to(x.dtype)
        #print(f"Final x.shape: {x.shape}")
        x = image_encoder.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        # TODO: Implement the part of the code where the deep prompt is injected into the CLIP model.
        # The custom_encode_image function largely follows the code from the CLIP repository.
        # You only need to modify the code responsible for running the transformer blocks.

        # Steps:
        # - Iterate over the transformer blocks (image_encoder.transformer.resblocks).
        for i, block in enumerate(image_encoder.transformer.resblocks):
            # - If the current layer == self.injection_layer, we add the deep prompt to the input of the transformer block.
            if i == self.injection_layer:
                batch_size = x.shape[1]
                #print(f"x.shape: {x.shape}") # shape = [num_patches, batch_size, 768]
                #batch_size = 128
                # - Repeat the deep prompt for each image in the batch.
                deep_prompt = self.deep_prompt.repeat(1, batch_size, 1) # shape = [num_prompts, batch_size, 768]
                # - Print the shapes of the deep prompt and the input of the transformer block.
                #print(f"deep_prompt.shape: {deep_prompt.shape}")
                #print(f"x.shape: {x.shape}")
                #print(f"deep_prompt.dtype: {deep_prompt.dtype}")
                #print(f"x.dtype: {x.dtype}")
                #print("Adding deep prompt to transformer block...")
                # - Concatenate the deep prompt with the input of the transformer block.
                # - The shape of the concatenated tensor should be (num_patches + num_deep_prompts, batch_size, 768).
                # - The deep prompt should be concatenated to the input of the transformer block.
                # - The deep prompt should be concatenated along the num_patches dimension, that is the 0-th.
                x = torch.cat([deep_prompt, x], dim=0) # shape = [num_prompts + num_patches, batch_size, 768]

                
            # - Run the transformer block.
            x = block(x)
        # Hint: Beware of the batch size (the deep prompt is the same for all images in the batch).

        x = x.permute(1, 0, 2)  # LND -> NLD

        x = image_encoder.ln_post(x[:, 0, :])

        if image_encoder.proj is not None:
            x = x @ image_encoder.proj

        return x

    def load_clip_to_cpu(self, args):
        """Loads CLIP model to CPU."""
        backbone_name = args.arch
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url, args.root)
        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        model = clip.build_model(state_dict or model.state_dict())
        return model

    @torch.no_grad()
    def visualize_prompt(self, method):
        """Visualizes the prompt."""
        warnings.warn("Deep prompts are not supported for visualization.")
