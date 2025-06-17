import torch
import types


def modified_forward(self, input_tensor, temb):
    hidden_states = input_tensor

    if self.time_embedding_norm == "ada_group":
        hidden_states = self.norm1(hidden_states, temb)
    else:
        hidden_states = self.norm1(hidden_states)

    hidden_states = self.nonlinearity(hidden_states)

    if self.upsample is not None:
        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            input_tensor = input_tensor.contiguous()
            hidden_states = hidden_states.contiguous()
        input_tensor = self.upsample(input_tensor)
        hidden_states = self.upsample(hidden_states)
    elif self.downsample is not None:
        input_tensor = self.downsample(input_tensor)
        hidden_states = self.downsample(hidden_states)

    hidden_states = self.conv1(hidden_states)

    if self.time_emb_proj is not None:
        temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]

    if temb is not None and self.time_embedding_norm == "default":
        hidden_states = hidden_states + temb

    if self.time_embedding_norm == "ada_group":
        hidden_states = self.norm2(hidden_states, temb)
    else:
        hidden_states = self.norm2(hidden_states)

    if temb is not None and self.time_embedding_norm == "scale_shift":
        scale, shift = torch.chunk(temb, 2, dim=1)
        hidden_states = hidden_states * (1 + scale) + shift

    hidden_states = self.nonlinearity(hidden_states)

    hidden_states = self.dropout(hidden_states)
    hidden_states = self.conv2(hidden_states)

    if self.conv_shortcut is not None:
        input_tensor = self.conv_shortcut(input_tensor)

    output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

    self.f_map_value = hidden_states # hidden_states corresponds to f_t^l (F)
    return output_tensor 


"""
A function that prepares a U-Net model for training by enabling gradient computation 
for a specified set of parameters and setting the forward pass to be performed by a 
custom cross attention processor.

Parameters:
unet: A U-Net model.

Returns:
unet: The prepared U-Net model.
"""
def prep_unet(unet):
    # set the gradients for feature maps to be true
    for name, params in unet.named_parameters():
        if ('down_blocks' in name or 'up_blocks' in name) and 'resnets' in name:
            params.requires_grad = True
        else:
            params.requires_grad = False
            
    # replace the fwd function
    for name, module in unet.named_modules():
        module_name = type(module).__name__

        if ('down_blocks' in name or 'up_blocks' in name) and module_name == "ResnetBlock2D":
            # Feature map
            module.forward = types.MethodType(modified_forward, module)

    return unet
