import os, sys, random
import cv2
import torch
from matplotlib import pyplot as plt
import conf as cfg
from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter
from torchvision import transforms
import numpy as np
import subprocess


def bgr2graycoord(img):

    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = grayimg.shape

    channelx = np.ones(shape=(h, w))
    for i in range(h):
        channelx[i, :] = i*channelx[i, :]
    channelx = 2*(channelx/h) - 1

    channely = np.ones(shape=(h, w))
    for i in range(w):
        channely[:, i] = i*channely[:, i]
    channely = 2*(channely/w) - 1
    
    img[:, :, 0] = grayimg
    img[:, :, 1] = channelx
    img[:, :, 2] = channely

    return img


trf = transforms.Compose(transforms=[transforms.ToTensor(), transforms.Normalize(mean=[125/255], std=[36/255])])
def imgpatchs(img):
    H, W = 224, 224
    h, w, c = img.shape
    graytrf = trf(img[:, :, 0])
    img[:, :, 0] = graytrf.numpy()
    numh = int(h/H) - 1
    numw = int(w/W) - 1
    patches = []
    for i in range(numh):
        hi = i*H
        for j in range(numw):
            wi = j*W
            patch = img[hi:hi+H, wi:wi+W, :]
            patches.append(patch)
    return patches


def extractiframes(srcpath, trgpath, numiframes=20):
    # vd = Video()
    # [D03_Huawei_P9, ....]
    srcfolders = os.listdir(srcpath)
    try:
        srcfolders.remove('.DS_Store')
    except Exception as e:
        print("already removed")

    for srcfolder in srcfolders:
        i=0
        trgfolderpath = os.path.join(trgpath, srcfolder)
        cfg.creatdir(trgfolderpath)
        # diskwriter = KeyFrameDiskWriter(location=trgfolderpath)

        # D03_Huawei_P9/vidos/
        srcfolderpath = os.path.join(srcpath, srcfolder, 'videos')
        # [outdoor, outdoorWA, outdoorYT]
        innersrcfolders = os.listdir(srcfolderpath)

        try:
            innersrcfolders.remove('.DS_Store')
        except Exception as e:
            print("already removed")

        for innersrcfolder in innersrcfolders:
            # D03_Huawei_P9/vidos/outdoor
            innersrcfolderpath = os.path.join(srcfolderpath, innersrcfolder)
            # [Do3_V_outdoor..., ...]
            vidoefiles = os.listdir(innersrcfolderpath)
            for videofile in vidoefiles:
                videofilepath = os.path.join(innersrcfolderpath, videofile)
                filename = os.path.join(trgfolderpath, f"img-{i}")
                # vd.extract_video_keyframes(no_of_frames=numiframes, file_path=videofilepath, writer=diskwriter)
                os.system(f"ffmpeg -skip_frame nokey -i {videofilepath} -vsync 0 -frame_pts true {filename}out%d.png")
                i+=1



def extractallpatches(src_path, trg_path):

    srcfolders = os.listdir(src_path)
    try:
        srcfolders.remove('.DS_Store')
    except Exception as e:
        print(f'{e}')

    for srcfolder in srcfolders:
        srcfolderpath = os.path.join(src_path, srcfolder)
        # trgtrainfolderpath = os.path.join(cfg.paths['train'], srcfolder)
        # trgtestfolderpath = os.path.join(cfg.paths['test'], srcfolder)
        trgfolderpath  = os.path.join(trg_path, srcfolder)

        srcfolderfiles = os.listdir(srcfolderpath)
        numsrcfiles = len(srcfolderfiles)
        train_size = int(0.8*numsrcfiles)

        # cfg.creatdir(trgtrainfolderpath)
        # cfg.creatdir(trgtestfolderpath)
        cfg.creatdir(trgfolderpath)
        i=0

        for cnt, srcfile in enumerate(srcfolderfiles):
            srcimgpath = os.path.join(srcfolderpath, srcfile)
            srcimg = cv2.imread(srcimgpath)
            coordimg = bgr2graycoord(srcimg)
            coordpatches = imgpatchs(coordimg)

            for patch in coordpatches:
                    patchname = f'patch_{i}.png'
                    patchpath = os.path.join(trgfolderpath, patchname)
                    cv2.imwrite(filename=patchpath, img=patch)
                    i+=1

            # if cnt<train_size:
            #     for patch in coordpatches:
            #         patchname = f'patch_{i}.png'
            #         patchpath = os.path.join(trgtrainfolderpath, patchname)
            #         cv2.imwrite(filename=patchpath, img=patch)
            #         i+=1
            # else:
            #     for patch in coordpatches:
            #         patchname = f'patch_{i}.png'
            #         patchpath = os.path.join(trgtestfolderpath, patchname)
            #         cv2.imwrite(filename=patchpath, img=patch)
            #         i+=1





def main():
    print(2)
    # foderspath = os.path.join(cfg.paths['data'], 'D03_Huawei_P9', 'videos', 'outdoor', 'D03_V_outdoor_move_0001.mp4')
    # vd = Video()
    # diskwriter = KeyFrameDiskWriter(location=cfg.paths['data'])
    # vd.extract_video_keyframes(no_of_frames=2, file_path=foderspath, writer=diskwriter)

    # src_path = cfg.paths['videos']
    # trg_path = cfg.paths['iframes']
    # extractiframes(srcpath=src_path, trgpath=trg_path, numiframes=20)

    srcpath = cfg.paths['iframes']
    trgpath = cfg.paths['dataset']
    extractallpatches(src_path=srcpath, trg_path=trgpath)


if __name__ == '__main__':
    main()
