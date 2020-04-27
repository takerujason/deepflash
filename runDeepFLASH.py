import torch
import sys
from configs.getConfig import getConfig
from fileIO.io import saveConfig2Json
from fileIO.io import createExpFolder
from torch.utils.data import DataLoader
import SimpleITK as sitk
import os, glob
import numpy as np
from fileIO.io import safeLoadMedicalImg, convertTensorformat, loadData2
from torchvision import transforms
from fileIO.io import safeDivide

<<<<<<< HEAD
def loadDataVol(inputfilepath):
=======

def runExp(config):
    #%% 1. Set configration and device    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    #%% 2. Set Data
    import numpy as np
    from fileIO.io import safeLoadMedicalImg, convertTensorformat, loadData2

    # Load training and valid data
>>>>>>> f18da4fd426e9e7a56cddc4cb4c4de70b5842554
    SEG, COR, AXI = [0,1,2]
    targetDim = 2
    for idx, filename in enumerate (sorted(glob.glob(inputfilepath), key=os.path.getmtime)):
        img = sitk.GetArrayFromImage(sitk.ReadImage(filename))
        img = np.rollaxis(img, 0, 3)
        temp = convertTensorformat(img,
                                sourceFormat = 'single3DGrayscale', 
                                targetFormat = 'tensorflow', 
                                targetDim = targetDim, 
                                sourceSliceDim = AXI)      
        if idx == 0:        
            outvol = temp
        else:
            outvol  = np.concatenate((outvol , temp), axis=0)
    return outvol
    # outvol = input_src_data

def runExp(config):
    #%% 1. Set configration and device    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    #%% 2. Set Data

    # Load training data
    SEG, COR, AXI = [0,1,2]
    targetDim = 2
    import os, glob
    input_src_data = loadDataVol("./data/src_fourier_real/*.mhd")
    input_tar_data = loadDataVol("./data/tar_fourier_real/*.mhd")
    input_vel_data_x = loadDataVol("./data/velo_fourier_real/*.mhd")
    input_vel_data_y = loadDataVol("./data/velo_fourier_real_y/*.mhd")
    input_vel_data = np.concatenate((input_vel_data_x, input_vel_data_y), axis=3)

    #load validation data
    pred_src_data = loadDataVol("./data/src_fourier_real/*.mhd")
    pred_tar_data =loadDataVol("./data/tar_fourier_real/*.mhd")
    # print(input_vel_data.shape)

    xNorm = lambda img : safeDivide(img - np.min(img), (np.max(img) - np.min(img))) - 0.5
    trans3DTF2Torch = lambda img: np.moveaxis(img, -1, 0)
    img_transform = transforms.Compose([
        xNorm,
        trans3DTF2Torch
    ])
    from src.DataSet import DataSet2D, DataSetDeep, DataSetDeepPred
    training = DataSetDeep (source_data = input_src_data, target_data = input_tar_data,  groundtruth = input_vel_data, transform=img_transform, device = device  )
    from src.DFModel import DFModel
    def weights_init_uniform(m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # apply a uniform distribution to the weights and a bias=0
            m.weight.data.uniform_(0.0, 1.0)
            m.bias.data.fill_(0)   
    deepflashnet = DFModel(net_config = config['net'], 
                        loss_config = config['loss'],
                        device=device)
    deepflashtestnet = DFModel(net_config = config['net'], 
                        loss_config = config['loss'],
                        device=device)
    testing = DataSetDeepPred (source_data = pred_src_data, target_data = pred_tar_data, transform=img_transform, device = device)

    #%% 6. Training and Validation
    loss = deepflashnet.trainDeepFlash(training_dataset=training, training_config = config['training'], testing_dataset = testing, valid_img= None, expPath = None)
    #deepflashnet.save('./savenet3d.pth')
    # torch.save(deepflashnet.state_dict(), '2Dnet')
      
    # deepflashtestnet = DFModel(net_config = config['net'], 
    #                     loss_config = config['loss'],
    #                     device=device)
    # deepflashtestnet.load('./savenet.pth')
   


configName = 'deepflash'
config = getConfig(configName)
runExp(config)


<<<<<<< HEAD
=======
#     valid_data = convertTensorformat(img=safeLoadMedicalImg(config['data']['valid'][0]),
#                                 sourceFormat = 'single3DGrayscale', 
#                                 targetFormat = 'pytorch', 
#                                 targetDim = targetDim, 
#                                 sourceSliceDim = AXI)
    
#     #%% 3. Set transformations
#     from torchvision import transforms
#     from utils.io import safeDivide
#     #xNorm = lambda img : (img - np.min(img)) / (np.max(img) - np.min(img)) - 0.5
#     xNorm = lambda img : safeDivide(img - np.min(img), (np.max(img) - np.min(img))) - 0.5
#     trans3DTF2Torch = lambda img: np.moveaxis(img, -1, 0)
#     img_transform = transforms.Compose([
#         xNorm,
#         trans3DTF2Torch
#     #    transforms.ToTensor()
#     ])

#     #%% 4. Set dataset


#     training_dataset = DataSet2D(imgs = training_data, transform=img_transform, device = device)

#     #%% 5. Set network
#     from modules.AEModel import AEModel
#     autoencoder = AEModel(net_config = config['net'], 
#                         loss_config = config['loss'], 
#                         device=device)
#     if continueTraining:
#         # autoencoder.load(oldExpPath + '/model/model.pth')
#         autoencoder.load(oldExpPath + '/checkpoint/checkpoint.pth')

#     #%% 6. Training
#     print('Start configure: ' + expName)
#     loss_history, past_time = autoencoder.train(training_dataset=training_dataset, training_config = config['training'], valid_img=valid_data, expPath = expPath)
#     autoencoder.save(expPath + '/checkpoint/checkpoint.pth')

#     # Add loss and training time to config json file
#     config['training']['loss'] = loss_history[-1]
#     config['training']['time_hour'] = past_time / 3600
#     saveConfig2Json(config, expPath + '/config.json')

#     return expPath, expName, loss_history, past_time

# def runExpGroup(configGroup, configGruopName, resultPath = '../result', continueTraining = False, oldExpPath = None):    
#     expGroupPath, _ = createExpFolder(resultPath, configGruopName)
#     # resultPath += '/' + expGroupName
#     for expIdx, expConfig in enumerate(configGroup):
#         runExp(expConfig, f'idx-{expIdx}', expGroupPath, addDate = False)
>>>>>>> f18da4fd426e9e7a56cddc4cb4c4de70b5842554
