# Copyright (c) 2025 Yu-Chen, Chiu
# Tamkang University, RVLab

import os
import cv2
import numpy as np

import scipy
from scipy.ndimage import distance_transform_edt, convolve
from scipy.signal.windows import gaussian

def fspecial(n, size, sigma):
    if n == "gaussian":
        gaussian_1d = gaussian(size, sigma)
        K = np.outer(gaussian_1d, gaussian_1d)
        K /= K.sum()
        return K
    else:
        print("error")
        return 0
    
def WFb(FG,GT):
    #  WFb Compute the Weighted F-beta measure (as proposed in "How to Evaluate
    #  Foreground Maps?" [Margolin et. al - CVPR'14])
    #  Usage:
    #  Q = FbW(FG,GT)
    #  Input:
    #    FG - Binary/Non binary foreground map with values in the range [0 1]. Type: double.
    #    GT - Binary ground truth. Type: logical.
    #  Output:
    #    Q - The Weighted F-beta score

    # Check input
    if not np.issubdtype(FG.dtype, np.float64):
        raise ValueError("FG must be a double (float64) array")
    if (np.max(FG)>1 or np.min(FG)<0):
        raise ValueError('FG should be in the range of [0 1]')
    if not np.issubdtype(GT.dtype, np.bool_):
        raise ValueError("GT must be a boolean value")
    
    dGT = GT.astype(np.double) #Use double for computations.
    
    E = np.abs(FG - dGT)

    # Dst, IDXT = bwdist(GT)
    Dst, IDXT = distance_transform_edt(~GT, return_indices=True)

    # Pixel dependency
    K = fspecial('gaussian',7,5)
    Et = E.copy()
    Et[~GT] = E[IDXT[0, ~GT], IDXT[1, ~GT]] #To deal correctly with the edges of the foreground region
    # EA = imfilter(Et,K);
    EA = convolve(Et, K, mode='reflect') # convolve == imfilter, reflect is matlab defalut mode

    MIN_E_EA = E.copy()
    MIN_E_EA[GT & (EA < E)] = EA[GT & (EA < E)]
    # Pixel importance
    B = np.ones(GT.shape)
    B[~GT] = 2 - np.exp(np.log(1 - 0.5) / 5 * Dst[~GT])

    Ew = MIN_E_EA * B

    eps = np.spacing(1)
    TPw = np.sum(dGT) - np.sum(Ew[GT])
    FPw = np.sum(Ew[~GT])

    R = 1 - np.mean(Ew[GT])  # Weighed Recall
    P = TPw / (eps + TPw + FPw)  # Weighted Precision


    Q = 2 * (R * P) / (eps + R + P)  # Beta=1
    # Q = (1 + beta**2) * (R * P) / (eps + R + (beta * P))

    return Q

def evaluate_Fwb_non_rank(path_predited, path_gt):

    # affordances index
    aff_start=2  # ignore {background} label
    aff_end=10   # change based on the dataset 
    
    list_predicted = os.listdir(path_predited) # get all files in current folder
    list_gt = os.listdir(path_gt)
    list_predicted = sorted(list_predicted)
    list_gt = sorted(list_gt) # make the same style
    c1 = len(list_predicted)
    c2 = len(list_gt)
    assert c1==c2 # test length
    num_of_files = len(list_gt)

    # F_wb_aff = []
    F_wb_aff = np.full((num_of_files, 1), np.nan)
    F_wb_non_rank = []

    for aff_id in range(aff_start, aff_end+1): # from 2 --> final_aff_id
        for i in range(num_of_files):
            print('------------------------------------------------')
            print('affordance id={}, image i={}'.format(aff_id, i))
            print('current pred: {}'.format(os.path.join(path_predited, list_predicted[i])))
            print('current grth: {}'.format(os.path.join(path_gt, list_gt[i])))

            pred_im = cv2.imread(os.path.join(path_predited, list_predicted[i]), -1)
            gt_im = cv2.imread(os.path.join(path_gt, list_gt[i]), -1)

            print('size pred_im: {}'.format(pred_im.shape))
            print('size gt_im: {}'.format(gt_im.shape))

            targetID = aff_id - 1 # labels are zero-indexed so we minus 1

            # only get current affordance
            pred_aff = pred_im == targetID
            gt_aff = gt_im == targetID

            if gt_aff.sum() > 0: # only compute if the affordance has ground truth
                F_wb_aff[i] = WFb(pred_aff.astype(np.float64), gt_aff) # all WFb function
            else:
                print('no ground truth at i={}'.format(i))
                # pass
        print('Averaged F_wb for affordance id={} is: {}'.format(aff_id-1, np.nanmean(F_wb_aff)))
        F_wb_non_rank.append(np.nanmean(F_wb_aff))
    
    """ Result """
    print('------------------------------------------------')
    print("ans =")
    for r in F_wb_non_rank:
        print("    {}".format(round(r, 4)))

    print("ans_m =")
    print("    {}".format(round(np.nanmean(F_wb_non_rank), 4)))
    # print(F_wb_non_rank)
    
if __name__ == "__main__":
    pred_seg = "pred_seg"
    gt_seg = "gt_seg"
    evaluate_Fwb_non_rank(pred_seg, gt_seg)