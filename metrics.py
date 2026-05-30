import math
import cv2
import numpy as np
import torch
from niqe_utils import calculate_niqe
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

def preprocessing(d_img_org):
    d_img_org = padding_img(d_img_org)
    x_his = build_historgram(d_img_org)
    return {"x": d_img_org, "x_his": x_his}

def padding_img(img):
    b, c, h, w = img.shape
    h_out = math.ceil(h / 32) * 32
    w_out = math.ceil(w / 32) * 32

    left_pad = (w_out - w) // 2
    right_pad = w_out - w - left_pad
    top_pad = (h_out - h) // 2
    bottom_pad = h_out - h - top_pad

    img = torch.nn.ZeroPad2d((left_pad, right_pad, top_pad, bottom_pad))(img)

    return img

def build_historgram(img):
    with torch.no_grad():
        b, _, _, _ = img.shape

        r_his = torch.histc(img[0][0], 64, min=0.0, max=1.0)
        g_his = torch.histc(img[0][1], 64, min=0.0, max=1.0)
        b_his = torch.histc(img[0][2], 64, min=0.0, max=1.0)

        historgram = torch.cat((r_his, g_his, b_his)).unsqueeze(0).unsqueeze(0)

        for i in range(1, b):
            r_his = torch.histc(img[i][0], 64, min=0.0, max=1.0)
            g_his = torch.histc(img[i][1], 64, min=0.0, max=1.0)
            b_his = torch.histc(img[i][2], 64, min=0.0, max=1.0)

            historgram_temp = torch.cat((r_his, g_his, b_his)).unsqueeze(0).unsqueeze(0)
            historgram = torch.cat((historgram, historgram_temp), dim=0)

    return historgram

def get_uciqe(image):
    hsv = cv2.cvtColor(np.array(image * 255, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)
    delta = np.std(H) / 180
    mu = np.mean(S) / 255
    n, m = np.shape(V)
    number = math.floor(n * m / 100)
    Maxsum, Minsum = 0, 0
    V1, V2 = V / 255, V / 255

    for i in range(1, number + 1):
        Maxvalue = np.amax(np.amax(V1))
        x, y = np.where(V1 == Maxvalue)
        Maxsum = Maxsum + V1[x[0], y[0]]
        V1[x[0], y[0]] = 0

    top = Maxsum / number

    for i in range(1, number + 1):
        Minvalue = np.amin(np.amin(V2))
        X, Y = np.where(V2 == Minvalue)
        Minsum = Minsum + V2[X[0], Y[0]]
        V2[X[0], Y[0]] = 1

    bottom = Minsum / number

    conl = top - bottom
    uciqe = 0.4680 * delta + 0.2745 * conl + 0.2576 * mu
    return uciqe

def getUCIQE(image):
    # image:  B, H, W, C

    UCIQE = 0
    for i in range(image.shape[0]):
        UCIQE += get_uciqe(image[i, :, :, :])
    return UCIQE


### NIQE ### 
def getNIQE(image):
    # image:  B, H, W, C
    NIQE = 0
    for i in range(image.shape[0]):
        NIQE += calculate_niqe(image[i, :, :, :][:, :, ::-1] * 255)
    return NIQE

def getURanker(image: np.array, uranker_model):
    inputs = torch.from_numpy(image).float().to(next(uranker_model.parameters()).device)
    inputs = inputs.permute(0, 3, 1, 2)  # B, H, W, C => B, C, H, W
    inputs = preprocessing(inputs)
    uiqa = 0.0
    with torch.no_grad():
        uiqa += torch.sum(
            uranker_model(**inputs)["final_result"].squeeze(-1).squeeze(-1)
        ).item()
    return uiqa

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])