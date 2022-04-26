import torch
import sys
import os
import argparse
import pickle
import PIL.Image

# Ref : https://github.com/NVlabs/stylegan2-ada-pytorch

parser = argparse.ArgumentParser(description='StyleGAN2 generator to generate face images')

parser.add_argument("--num", default=50, type=int, help='Number of images to be generated(default value is 2000)')
parser.add_argument("--outdir", default="/workspace/fairDL/data/stylegan2", type=str, help='path to save the generated images(default is ../data/stylegan2/)')


def main():
    global args
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    try:
        sys.path.index('/workspace/stylegan2-ada-pytorch')
    except:
        sys.path.append('/workspace/stylegan2-ada-pytorch')

    #print(sys.path)
    os.makedirs(args.outdir, exist_ok=True)

    with open('/workspace/fairDL/src/ffhq.pkl', 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()

    for img_num in range(args.num):
        z = torch.randn([1, G.z_dim]).cuda()
        c = None
        #img = G(z, c, truncation_psi=1, noise_mode='const')
        img = G(z, c, truncation_psi=1, noise_mode='const')
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        #print(img[0])
        #print("\n")
        #print(img[0].cpu().numpy())
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{args.outdir}/{img_num}.png')

if __name__ == "__main__":
    main()

    

