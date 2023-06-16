import cv2
import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity as ssim
import warnings
import os
#↑載入OS模組

warnings.filterwarnings("ignore")

# 載入預訓練的LPIPS模型
loss_fn = lpips.LPIPS(net='alex')


dir_path = os.path.dirname(os.path.realpath("./1_138_111598092/U+82DB.png"))
#↑獲取當前資料夾名稱然後存成dir_path變數
all_file_name = os.listdir(dir_path)
#↑讀取資料夾內所有檔案名稱然後放進all_file_name這個list裡
id_list = ['111598001/','111598002/','111598006/','111598023/','1115980040/','111598069/',
           '111598012/','108820038/','111C52002/','111598068/','111598081/']

score = []

for k in id_list:  
    #lpips_distance list
    lpips_list = []
    #lpips_score
    lpips_score = 0
    for i in range(0,len(all_file_name)):
        # 自己的手寫字圖片路徑
        gt = cv2.imread("./1_138_111598092/"+all_file_name[i], cv2.IMREAD_GRAYSCALE)
        gt = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).float() / 255.0
        
        # 別人的手寫字圖片路徑
        gradient = cv2.imread("./1_138_"+k+all_file_name[i], cv2.IMREAD_GRAYSCALE)
        gradient = torch.from_numpy(gradient).unsqueeze(0).unsqueeze(0).float() / 255.0
        
        # 計算MSE
        #mse = np.mean((gt.squeeze().numpy() - gradient.squeeze().numpy()) ** 2)
        
        # 計算SSIM相似度
        #ssim_score = ssim(gt.squeeze().numpy(), gradient.squeeze().numpy(), win_size=7)
        
        # 計算LPIPS距離
        lpips_distance = loss_fn(gt, gradient)
        
        lpips_list.append(lpips_distance.item())
    #加總算平均
    lpips_score = sum(lpips_list)/len(lpips_list)
    score.append([k,lpips_score])
    print(k+'LPIPS distance:')
    print(lpips_score)

# Print the score
#print("MSE score:", "{:.5f}".format(mse) )
#print("SSIM score:", "{:.5f}".format(ssim_score) )
#print("LPIPS distance:", "{:.5f}".format(lpips_distance.item()))




