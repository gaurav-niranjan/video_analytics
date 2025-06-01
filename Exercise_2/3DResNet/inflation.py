import torch
import torch.nn as nn

def inflate_2d_to_3d(conv2d, conv3d):

    weights_2d = conv2d.weight.data
    out_c, in_c, H, W = weights_2d.shape
    T = conv3d.weight.shape[2]

    #Average the 2D weights across time:
    weights_3d = weights_2d.unsqueeze(2).repeat(1,1,T,1,1)/T
    conv3d.weight.data.copy_(weights_3d)

    if conv2d.bias is not None and conv3d.bias is not None:
        conv3d.bias.data.copy_(conv2d.bias.data)

    print(f"Infalted Conv2D {weights_2d.shape} -> Conv3D {weights_3d.shape}")

def inflate_resnet(model3d, model2d):

    state2d = model2d.state_dict()
    state3d = model3d.state_dict()

    new_state3d = {}

    for name, param3d in state3d.items():
        if name not in state2d:
            
            print(f"Skipping {name} — not in 2D model")
            new_state3d[name] = param3d
            continue

        param2d = state2d[name]

        # Conv layers
        if "conv" in name and len(param3d.shape) == 5:
            T = param3d.shape[2]
            param2d = param2d.unsqueeze(2).repeat(1, 1, T, 1, 1) / T  #Insert a Time dim -> Average the 2D weights across time:
            
            print(f"Inflating {name}: {param2d.shape} → {param3d.shape}")
            new_state3d[name] = param2d

        # BatchNorm or FC layers (same shape)
        elif param3d.shape == param2d.shape:
            new_state3d[name] = param2d
            
            print(f"Copied {name}")

        else:
            
            print(f"Shape mismatch in {name}, skipping")
            new_state3d[name] = param3d

    model3d.load_state_dict(new_state3d)
