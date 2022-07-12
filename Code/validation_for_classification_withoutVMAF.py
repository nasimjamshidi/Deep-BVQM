
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr

import data_preprocessing_for_classification as dpClass
import CNN_Models
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle as pkl
import torch
import math
import pickle
import glob

np.set_printoptions(precision=2)
use_gpu = torch.cuda.is_available()
np.random.seed(12)

import warnings
warnings.filterwarnings('ignore')



sz = 89 
batch_size = 32
Framrate=30
threshold=0



def CNN_Validation(Bitstream_Data, path_to_Normalization_Models, path_to_RandomForest_Model,path_to_CNN_Model,path_to_Result_CNN):
    
        """# **Normalisation**"""
        
        qpValues=Bitstream_Data.iloc[:,4:]
        
        """**qpvalues_Normalisation**"""
        
        
        with open(path_to_Normalization_Models + "Normalized_Model_qpValues.pkl", "rb") as infile:
          scaler_qpValues = pkl.load(infile)
          Norm_qpValues = scaler_qpValues.transform(qpValues)
        
        """### **Define and normalize the additive features**"""
        
        Bitstream_Data.insert(3,'Preset','veryfast')
        Bitstream_Data.columns=range(Bitstream_Data.shape[1])
        Bitstream_Data = Bitstream_Data.rename(columns={0: 'Frame_type', 1: 'Game', 2: 'Frame_size', 3: 'Preset', 4: 'Avg_qp'})
        
        
        Add_Frame_size= Bitstream_Data['Frame_size']
        Add_Frame_type=Bitstream_Data['Frame_type']
        Add_Preset= Bitstream_Data['Preset']
        
        """**Convert categorical additive features to numerical**
        
        **Normalise the additive features**"""
        
        label_encoder1 = LabelEncoder()
        Add_Frame_type = label_encoder1.fit_transform(Add_Frame_type)
        
        label_encoder2 = LabelEncoder()
        Add_Preset = label_encoder2.fit_transform(Add_Preset)
        
        
        Add_Frame_size = Add_Frame_size.values.reshape((len(Add_Frame_size), 1))
        Add_Frame_type = Add_Frame_type.reshape((len(Add_Frame_type), 1))
        Add_Preset = Add_Preset.reshape((len(Add_Preset), 1))
        
        
        with open(path_to_Normalization_Models + "Normalized_Model_Additive_Frame_size.pkl", "rb") as infile:
          scaler_Add_Frame_size = pkl.load(infile)
          Norm_Add_Frame_size = scaler_Add_Frame_size.transform(Add_Frame_size)
        
        with open(path_to_Normalization_Models + "Normalized_Model_Additive_Frame_type.pkl", "rb") as infile:
          scaler_Add_Frame_type = pkl.load(infile)
          Norm_Add_Frame_type = scaler_Add_Frame_type.transform(Add_Frame_type)
        
        with open(path_to_Normalization_Models + "Normalized_Model_Additive_Preset.pkl", "rb") as infile:
          scaler_Add_Preset = pkl.load(infile)
          Norm_Add_Preset = scaler_Add_Preset.transform(Add_Preset)
          
          
        
        """## **Filter out the qp values from column 173th onwards in order to be fit to 89*89**"""
        
        bitstream_short = Norm_qpValues[:,171:]
        
        
        
        
        """# **Reshape 2D to 3D ( i.e., (x,y)-->(x,89,89) )**"""
        
        bitstream_short = np.reshape(bitstream_short, (bitstream_short.shape[0],sz,sz))
        
        
        
        """#Add a new dimension (i.e., (x,89,89)-->(x,89,89,1) )"""
       
        BS_4D=bitstream_short[:,:, :, np.newaxis]
        
        
        
        
        """# Transpose the dimensions (i.e., (x,89,89,1)-->(x,1,89,89) )"""
        
        BS_4D = np.transpose(BS_4D, (0, 3, 1,2))
        
        
        

        """# **Convert Numpy to Tensor (to be prepared to enter the dataloader)**"""
        
        BS_4D_tensor=torch.from_numpy(BS_4D)
        Add_Frame_size_tensor = torch.from_numpy(Norm_Add_Frame_size)
        Add_Frame_type_tensor = torch.from_numpy(Norm_Add_Frame_type)
        Add_Preset_tensor = torch.from_numpy(Norm_Add_Preset)        
        
        # print('Bitstream 4D shape == {}. ' .format(BS_4D_tensor.shape))
        # print('Additive Frame size shape == {}. ' .format(Add_Frame_size_tensor.shape))
        # print('Additive Frame type shape == {}. ' .format(Add_Frame_type_tensor.shape))
        # print('Additive Preset shape == {}. ' .format(Add_Preset_tensor.shape))
        
        
        """# **Validation Loader**"""
        
        Valid_Dataset = torch.utils.data.TensorDataset(BS_4D_tensor)
        
        Valid_dl = torch.utils.data.DataLoader(Valid_Dataset, batch_size=batch_size, shuffle=False, num_workers=8)
        
        """# **Validation_Additive_Frame_size Loader**"""
        
        Valid_Additive_Data_1 = torch.utils.data.TensorDataset(Add_Frame_size_tensor)
        
        Valid_dl_additive_1 = torch.utils.data.DataLoader(Valid_Additive_Data_1, batch_size=batch_size, shuffle=False, num_workers=8)
        
        """# **Validation_Additive_Frame_type Loader**"""
        
        Valid_Additive_Data_2 = torch.utils.data.TensorDataset(Add_Frame_type_tensor)
        
        Valid_dl_additive_2 = torch.utils.data.DataLoader(Valid_Additive_Data_2, batch_size=batch_size, shuffle=False, num_workers=8)
        
        """# **Validation_Additive_Preset Loader**"""
        
        Valid_Additive_Data_3 = torch.utils.data.TensorDataset(Add_Preset_tensor)
        
        Valid_dl_additive_3 = torch.utils.data.DataLoader(Valid_Additive_Data_3, batch_size=batch_size, shuffle=False, num_workers=8)
                
        
        
        """# **Loading the model**"""
        
        model = CNN_Models.HeavyCNN() #SimpleCNN
        model.load_state_dict(torch.load(path_to_CNN_Model ), strict=False)
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
            
            
        
        """# **Extracting Activation Features**"""
        
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        
        ActivationFeatures_Fc2=[]
        for i, (bitstream,additional1,additional2,additional3) in enumerate(zip(Valid_dl,Valid_dl_additive_1,Valid_dl_additive_2,Valid_dl_additive_3)):
              inputs, additional1,additional2,additional3 = (bitstream[0]), (additional1[0]),(additional2[0]),(additional3[0])
              model.fc2.register_forward_hook(get_activation('fc2'))
              outputs = model(inputs.float(),additional1.float(),additional2.float(),additional3.float())
              ActivationFeatures_Fc2.append(activation['fc2'])
        
        
        
        """# **Concatenate all activation features in axis 0**"""
        
        Activation_Features=pd.DataFrame()
        for i in range(len(ActivationFeatures_Fc2)):
            Activation_Features = pd.concat([Activation_Features, pd.DataFrame(ActivationFeatures_Fc2[i].cpu().numpy())], axis = 0)
        
        
        
        """# **Concatenate the Act features and vmaf file in axis 1 to have the name and Act features of each frame as one single file**"""
        
        ActFeatures=pd.DataFrame()
        ActFeatures=pd.concat([Bitstream_Data.iloc[:,0].reset_index(drop=True),Activation_Features.reset_index(drop=True)], axis = 1)
        
        ActFeatures.head(1)
        
        
        
        """# **Add 'Frame_type', 'Preset' and 'Frame-size' to the file consisted of Act-features and vmaf file** """
        
        ActFeatures_FrameType_FrameSize_Preset=pd.DataFrame()
        ActFeatures_FrameType_FrameSize_Preset=pd.concat([ActFeatures.reset_index(drop=True),pd.DataFrame(Add_Frame_size).reset_index(drop=True),pd.DataFrame(Add_Frame_type).reset_index(drop=True),pd.DataFrame(Add_Preset).reset_index(drop=True)], axis = 1)
        
        ActFeatures_FrameType_FrameSize_Preset.columns = range(ActFeatures_FrameType_FrameSize_Preset.shape[1])
        
        ActFeatures_FrameType_FrameSize_Preset.head(1)
        
        ActFeatures_FrameType_FrameSize_Preset = ActFeatures_FrameType_FrameSize_Preset.rename(columns={0:'Game', 1:'Act_feature1', 2:'Act_feature2', 3:'Act_feature3', 4:'Act_feature4', 5:'Act_feature5', 6:'Frame_size', 7:'Frame_type', 8:'Preset'})
        
        ActFeatures_FrameType_FrameSize_Preset.head(1)
        
        
        
        """# **Define the X , y for random forest**"""
        
        X=ActFeatures_FrameType_FrameSize_Preset.iloc[:,1:]
        
        """# **Make predictions using random forest for classification**"""
        
        
        filename = path_to_RandomForest_Model
        regressor = pickle.load(open(filename, 'rb'))
        
        Y_pred = regressor.predict(X)
        
        
        predVMAF=ActFeatures_FrameType_FrameSize_Preset.iloc[:,0].to_frame()
        predVMAF.insert(1,'Vmaf_pred',Y_pred)
        
        predVMAF.to_excel(path_to_Result_CNN + 'PredictedVMAF.xlsx', engine='openpyxl' )
        
       
        #return predVMAF
