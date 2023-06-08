import argparse
import os
import sys
import time
from datetime import datetime
import math
import re

import numpy as np
import torch
from torch.optim import Adam, NAdam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx

import utils
from transformer_net import TransformerNet
import clip


def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)

def uniform(b1, b2, shape=(1,)):
    return (b1 - b2) * torch.rand(shape) + b2

clip_preprocess = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


def flicker_loss_func(x, y, model, mse_loss):
    """Call it before y = utils.normalize_batch(y)"""
    angle = uniform(-2, 2).item()
    scale = uniform(0.97, 1.03).item()
    shear = uniform(-2, 2, (2,)).tolist()
    translate = uniform(-4, 4, (2,)).round().int().tolist()

    xd = transforms.functional.affine(x, angle=angle, translate=translate, scale=scale, shear=shear, 
                                      interpolation=transforms.InterpolationMode.BILINEAR)
    yd_x = transforms.functional.affine(y, angle=angle, translate=translate, scale=scale, shear=shear,
                                        interpolation=transforms.InterpolationMode.BILINEAR)
    y_xd = model(xd)

    loss = mse_loss(yd_x, y_xd)
    return loss

def train(args):

    now = datetime.now()

    if args.cuda:
        device = torch.device("cuda")
    elif args.mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    mse_loss = torch.nn.MSELoss()

    if args.image_size <= 320:
        clip_model, _ = clip.load('RN50x4', device, jit=False) #img size = 288
    elif args.image_size <= 416:
        clip_model, _ = clip.load('RN50x16', device, jit=False) #img size = 384
    else:
        clip_model, _ = clip.load('RN50x64', device, jit=False) #img size = 448
    
    clip_model = clip_model.visual

    for param in clip_model.parameters():
        param.requires_grad = False

    features = {}
    def get_features(name):
        def hook(model, input, output):
            features[name] = output
        return hook

    clip_model.layer1[2].register_forward_hook(get_features('feats1'))
    clip_model.layer2[2].register_forward_hook(get_features('feats2'))
    clip_model.layer3[2].register_forward_hook(get_features('feats3'))
    clip_model.layer4[2].register_forward_hook(get_features('feats4'))
    #clip_model.layer3[-1].register_forward_hook(get_features('feats4'))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    input_preprocess = transforms.Compose([
        transforms.RandomRotation((-20,20)),
        transforms.Resize(int(args.image_size*1.2)),
        transforms.CenterCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        clip_preprocess,
    ])


    train_dataset = datasets.ImageFolder(args.dataset, input_preprocess)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    transformer = TransformerNet().to(device)
    optimizer = NAdam(transformer.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.96)
    

    style_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        #transforms.RandomCrop(args.image_size),
        transforms.ToTensor(),
        clip_preprocess
    ])
    style = utils.load_image(args.style_image, size=args.style_size)
    style = style_transform(style)
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)

    clip_model(style)
    features_style = [f.float() for f in features.values()]
    gram_style = [utils.gram_matrix(y) for y in features_style]

    for e in range(args.epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):

            with open(r"neural_style\variables.txt","rb") as f:
                lines = f.readlines()
            a_s, a_c, a_f, clip_feat1,clip_feat2,clip_feat3,clip_feat4 = float(lines[0].strip()),float(lines[1].strip()),float(lines[2].strip()),float(lines[3].strip()),float(lines[4].strip()),float(lines[5].strip()),float(lines[6].strip())
            
            print("batch nr {}".format(batch_id), end = "\r")
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = x.to(device)
            y = transformer(x)

            flicker_loss = flicker_loss_func(x.clone(), y.clone(), transformer, mse_loss)

            y = clip_preprocess(y)

            clip_model(y)
            features_y = [f.float() for f in features.values()]

            clip_model(x)
            features_x = [f.float() for f in features.values()]

            content_loss = mse_loss(features_y[0], features_x[0])

            style_loss = 0.
            for ft_y, gm_s, w in zip(features_y, gram_style, [clip_feat1, clip_feat2, clip_feat3, clip_feat4]):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += w * mse_loss(gm_y, gm_s[:n_batch])

            total_loss = (a_s*style_loss) +( a_c*content_loss) + (a_f*flicker_loss)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 4)
            optimizer.step()

            agg_content_loss += a_c*content_loss.item()
            agg_style_loss += a_s*style_loss.item()


            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)

            if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                transformer.eval()
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                torch.save(transformer.state_dict(), ckpt_model_path)

                content_transform = transforms.Compose([
                    transforms.Resize(args.image_size),
                    transforms.CenterCrop(args.image_size),
                    transforms.ToTensor(),
                    clip_preprocess
                ])

                pic = utils.load_image(r"C:\Users\dan\Desktop\style_transfer\images\content-images\amber.jpg")
                pic = content_transform(pic).unsqueeze(0).to(device)

                with torch.no_grad():
                    result = transformer(pic).mul(255).clamp(0, 255).cpu()[0]

                snapshot_name = "snap_{}.{}_{}.{}".format(now.month,now.day,now.hour,now.minute)#a_s,a_c,a_f,args.lr)
                save_image_to = os.path.join(args.save_model_dir,snapshot_name)
                if not os.path.isdir(save_image_to): os.mkdir(save_image_to)
                image_name = os.path.join(save_image_to,"epoch_"+str(e+1)+"_"+str(batch_id+1)+".png")
                utils.save_image(image_name, result)    

                transformer.to(device).train()
            
            scheduler.step()
            # if batch_id==10000: break


    # save model
    transformer.eval().cpu()
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(":","x") + "_" + str(
        args.content_weight) + "_" + str(args.style_weight) + ".pth"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def stylize(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    content_image = utils.load_image(args.content_image, scale=args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    if args.model.endswith(".onnx"):
        output = stylize_onnx(content_image, args)
    else:
        with torch.no_grad():
            style_model = TransformerNet()
            state_dict = torch.load(args.model)
            # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
            for k in list(state_dict.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del state_dict[k]
            style_model.load_state_dict(state_dict)
            style_model.to(device)
            style_model.eval()
            if args.export_onnx:
                assert args.export_onnx.endswith(".onnx"), "Export model file should end with .onnx"
                output = torch.onnx._export(
                    style_model, content_image, args.export_onnx, opset_version=11,
                ).cpu()
            else:
                output = style_model(content_image).cpu()
    utils.save_image(args.output_image, output[0])


def webcam(args):
    import cv2
    from torchvision import transforms

    size = (384,512)

    input_preprocess = transforms.Compose([
        transforms.RandomHorizontalFlip(1),
        clip_preprocess,
    ])

    device = torch.device("cuda" if args.cuda else "cpu")

    with torch.no_grad(), torch.autocast('cuda', torch.float16):
        style_model = TransformerNet()
        state_dict = torch.load(args.model)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        style_model.eval()
        vid = cv2.VideoCapture(0)
        vid.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
        vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 384)

        while True:
            # Capture the video frame
            # by frame
            ret, frame = vid.read()

            frame = torch.permute(torch.from_numpy(frame), (2, 0, 1)).to(device).float()

            frame = input_preprocess(frame)[None]

            result = style_model(frame).mul(255).clamp(0, 255)[0]
            result = result.permute(1, 2, 0)
            result = torch.flip(result, [2]).cpu().numpy()

            # Display the resulting frame
            
            cv2.imshow('frame', result.astype("uint8"))

            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    vid.release()
    cv2.destroyAllWindows()


def stylize_onnx(content_image, args):
    """
    Read ONNX model and run it using onnxruntime
    """

    assert not args.export_onnx

    import onnxruntime

    ort_session = onnxruntime.InferenceSession(args.model)

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(content_image)}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]

    return torch.from_numpy(img_out_y)


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",
                                  help="path to style-image")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                  help="path to folder where checkpoints of trained models will be saved")
    train_arg_parser.add_argument("--image-size", type=int, default=512,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--style-size", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--cuda", type=int, required=True,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")
    train_arg_parser.add_argument("--style-weight", type=float, default=1e10,
                                  help="weight for style-loss, default is 1e10")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")
    train_arg_parser.add_argument("--log-interval", type=int, default=10,
                                  help="number of images after which the training loss is logged, default is 500")
    train_arg_parser.add_argument("--checkpoint-interval", type=int, default=100,
                                  help="number of batches after which a checkpoint of the trained model will be created")
    train_arg_parser.add_argument('--mps', action='store_true', default=False, help='enable macOS GPU training')

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
    eval_arg_parser.add_argument("--cuda", type=int, default=False,
                                 help="set it to 1 for running on cuda, 0 for CPU")
    eval_arg_parser.add_argument("--export_onnx", type=str,
                                 help="export ONNX model to a given file")
    eval_arg_parser.add_argument('--mps', action='store_true', default=False, help='enable macOS GPU training')

    webcam_arg_parser = subparsers.add_parser("webcam", help="parser for webcam arguments")
    webcam_arg_parser.add_argument("--content-scale", type=float, default=None,
                                   help="factor for scaling down the content image")
    webcam_arg_parser.add_argument("--cuda", type=int, default=False,
                                   help="set it to 1 for running on cuda, 0 for CPU")
    webcam_arg_parser.add_argument("--model", type=str, required=True,
                                   help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
    webcam_arg_parser.add_argument('--mps', action='store_true', default=False, help='enable macOS GPU training')
    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)
    if not args.mps and torch.backends.mps.is_available():
        print("WARNING: mps is available, run with --mps to enable macOS GPU")

    if args.subcommand == "train":
        check_paths(args)
        train(args)
    elif args.subcommand == "eval":
        stylize(args)
    elif args.subcommand == "webcam":
        webcam(args)


if __name__ == "__main__":
    main()
