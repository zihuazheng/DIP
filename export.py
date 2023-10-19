import os
import torch
import argparse
from nets_stereo.dip import DIP
from torch import nn
import torch.onnx
from collections import OrderedDict

def export_net(args):
    # Build model
    model = DIP(max_disp=args.max_disp, mixed_precision=False, test_mode=True)
    pre_train = torch.load(args.model)

    # ref: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/13
    new_state_dict = OrderedDict()
    for k, v in pre_train.items():
        name = k[7:] # remove module.
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    # set_inference_mode(model)
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.track_running_stats = False
            module.running_var = None
            module.running_mean = None

    # Input to the model
    x = torch.randn(1, 3, 768, 1024, requires_grad=False)

    # Export the model
    if not os.path.exists(args.export_path):
            os.makedirs(args.export_path)
    model_path = os.path.join(args.export_path,"DIP_stereo.onnx")
    torch.onnx.export(model,               # model being run
                    (x, x, args.iters),                         # model input (or a tuple for multiple inputs)
                    model_path,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=16,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['image1', 'image2', 'iters'],   # the model's input names
                    output_names = ['disp']) # the model's output names
                    # dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                    #                 'output' : {0 : 'batch_size'}})
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--export_path', help="export model path")
    parser.add_argument('--max_disp', type=float, default=256)
    parser.add_argument('--iters', type=int, default=4)
    args = parser.parse_args()

    export_net(args)
