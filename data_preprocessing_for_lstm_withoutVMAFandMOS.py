
from sklearn.preprocessing import LabelEncoder
import data_preprocessing_for_classification
import CNN_Models
import pickle as pkl
import numpy as np
import pandas as pd
import torch
import os.path
import glob
import os

import warnings
warnings.filterwarnings('ignore')


np.set_printoptions(precision=2)
use_gpu = torch.cuda.is_available()
np.random.seed(12)


sz = 89 
batch_size = 32
Framrate=30
threshold=0

 



def valid_Loader(bitstream_tensor):
  valid_dataset = torch.utils.data.TensorDataset(bitstream_tensor)
  valid_dl = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8) 

  return valid_dl

def valid_Additive_Frame_size_Loader(Add_Frame_size_tensor):
  valid_additive_data_1 = torch.utils.data.TensorDataset(Add_Frame_size_tensor)
  valid_dl_additive_1 = torch.utils.data.DataLoader(valid_additive_data_1, batch_size=batch_size, shuffle=False, num_workers=8)

  return valid_dl_additive_1

def valid_Additive_Frame_type_Loader(Add_Frame_type_tensor):
  valid_additive_data_2 = torch.utils.data.TensorDataset(Add_Frame_type_tensor)
  valid_dl_additive_2 = torch.utils.data.DataLoader(valid_additive_data_2, batch_size=batch_size, shuffle=False, num_workers=8)

  return valid_dl_additive_2

def valid_Additive_Preset_Loader(Add_Preset_tensor):
  valid_additive_data_3 = torch.utils.data.TensorDataset(Add_Preset_tensor)
  valid_dl_additive_3 = torch.utils.data.DataLoader(valid_additive_data_3, batch_size=batch_size, shuffle=False, num_workers=8)
  return valid_dl_additive_3





def SubSample(file,Framrate):

    SubSampled=pd.DataFrame()
 #   temp1=pd.DataFrame()
    temp2=pd.DataFrame()

    if Framrate==30:
      for i in range(0,len(file),Framrate):
        temp=file.iloc[i:i+Framrate]
        temp2=temp[(temp.index % 2 == 0)]
        SubSampled=pd.concat((SubSampled,temp2),axis=0)
    elif Framrate==60:
      for i in range(0,len(file),Framrate):
        temp=file.iloc[i:i+Framrate]
        temp2=temp[(temp.index % 4 == 0)]
        SubSampled=pd.concat((SubSampled,temp2),axis=0)
    return SubSampled






def Extracting_Activation_Features(df_1,path_to_CNN_Model, valid_dl, valid_Additive_Frame_size_dl, valid_Additive_Frame_type_dl, valid_Additive_Preset_dl,Add_Frame_size,Add_Frame_type,Add_Preset):

      activation = {}
      Activation_Features=pd.DataFrame()
      ActivationFeatures_Fc2=[]
      ActFeatures=pd.DataFrame()
      ActFeatures_FrameType_FrameSize_Preset=pd.DataFrame()

      model = CNN_Models.HeavyCNN()
      model.load_state_dict(torch.load(path_to_CNN_Model), strict=False)
      model.eval()


      def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

      for i, (bitstream,additional1,additional2,additional3) in enumerate(zip(valid_dl,valid_Additive_Frame_size_dl,valid_Additive_Frame_type_dl,valid_Additive_Preset_dl)):
            inputs, additional1,additional2,additional3 = (bitstream[0]), (additional1[0]),(additional2[0]),(additional3[0])#,targets= to_var(bitstream[1])
            model.fc2.register_forward_hook(get_activation('fc2'))
            outputs = model(inputs.float(),additional1.float(),additional2.float(),additional3.float())
            ActivationFeatures_Fc2.append(activation['fc2'])

      for i in range(len(ActivationFeatures_Fc2)):
          Activation_Features = pd.concat([Activation_Features, pd.DataFrame(ActivationFeatures_Fc2[i].numpy())], axis = 0)

      ActFeatures=pd.concat([df_1['Game_Name'].reset_index(drop=True), Activation_Features.reset_index(drop=True)], axis = 1)
      ActFeatures.columns = range(ActFeatures.shape[1])

      ActFeatures_FrameType_FrameSize_Preset=pd.concat([ActFeatures.reset_index(drop=True),pd.DataFrame(Add_Frame_size).reset_index(drop=True),pd.DataFrame(Add_Frame_type).reset_index(drop=True),pd.DataFrame(Add_Preset).reset_index(drop=True)], axis = 1)
      ActFeatures_FrameType_FrameSize_Preset.columns = range(ActFeatures_FrameType_FrameSize_Preset.shape[1])
      ActFeatures_FrameType_FrameSize_Preset = ActFeatures_FrameType_FrameSize_Preset.rename(columns={0: 'Game_Name', 1:'Act_feature1' ,2: 'Act_feature2', 3: 'Act_feature3', 4: 'Act_feature4', 5: 'Act_feature5', 6: 'Add_Frame_size', 7: 'Add_Frame_type', 8: 'Add_Preset'})

      return ActFeatures_FrameType_FrameSize_Preset




def Normalize_additive_features(df_1, path_to_Normalization_Models):

    Add_Frame_size= df_1['Frame_Size']
    Add_Frame_type=df_1['Frame_Type']
    Add_Preset= df_1['Preset']

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
      

    return Norm_Add_Frame_size,Norm_Add_Frame_type,Norm_Add_Preset,Add_Frame_size,Add_Frame_type,Add_Preset




def Get_ActFeatures_MetaData(df_1, path_to_Normalization_Models, path_to_CNN_Model):
    
    df_1=df_1.reset_index(drop=True)  

    df_1.insert(4,'Preset','veryfast')
    df_1.columns=range(df_1.shape[1])
    df_1 = df_1.rename(columns={0:'Game_Name', 1:'Frame_Type',2:'Frame_Size',3:'Avg_qp' ,4:'Preset'})

    qpValues=df_1.iloc[:,5:].to_numpy()


    with open(path_to_Normalization_Models + "Normalized_Model_qpValues.pkl", "rb") as infile:
      scaler_qpValues = pkl.load(infile)
      Norm_qpValues = scaler_qpValues.transform(qpValues)

    
    Norm_Add_Frame_size,Norm_Add_Frame_type,Norm_Add_Preset,Add_Frame_size,Add_Frame_type,Add_Preset = Normalize_additive_features(df_1, path_to_Normalization_Models)


    bitstream_short = Norm_qpValues[:,Norm_qpValues.shape[1]-(sz*sz):]

    bitstream_short = np.reshape(bitstream_short, (bitstream_short.shape[0],sz,sz))

    BS=bitstream_short[:,:, :, np.newaxis]
    BS = np.transpose(BS, (0, 3, 1,2))

    bitstream_tensor=torch.from_numpy(BS)
    Add_Frame_size_tensor = torch.from_numpy(Norm_Add_Frame_size)
    Add_Frame_type_tensor = torch.from_numpy(Norm_Add_Frame_type)
    Add_Preset_tensor = torch.from_numpy(Norm_Add_Preset)

    valid_dl = valid_Loader(bitstream_tensor)
    valid_Additive_Frame_size_dl = valid_Additive_Frame_size_Loader(Add_Frame_size_tensor)
    valid_Additive_Frame_type_dl = valid_Additive_Frame_type_Loader(Add_Frame_type_tensor)
    valid_Additive_Preset_dl = valid_Additive_Preset_Loader(Add_Preset_tensor)

    vmaf_ActFeatures_FrameType_FrameSize_Preset = Extracting_Activation_Features(df_1, path_to_CNN_Model, valid_dl, valid_Additive_Frame_size_dl, valid_Additive_Frame_type_dl, valid_Additive_Preset_dl,Add_Frame_size,Add_Frame_type,Add_Preset)

    return vmaf_ActFeatures_FrameType_FrameSize_Preset




def ActFeatures_MetaData_Bitstream(Bitstream_Data, path_to_Result_LSTM, path_to_Normalization_Models, path_to_CNN_Model):
    
    Total_Act_Meta_Data_MOS=pd.DataFrame()
    df_1 = Bitstream_Data
    df_1 = SubSample(df_1,Framrate)
    Act_Meta_Data_MOS = Get_ActFeatures_MetaData(df_1, path_to_Normalization_Models, path_to_CNN_Model)
    Total_Act_Meta_Data_MOS = pd.concat((Total_Act_Meta_Data_MOS,Act_Meta_Data_MOS),axis=0)
    #Total_Act_Meta_Data_MOS.to_excel(path_to_Result_LSTM + 'Total_Act_Meta_Data_MOS.xlsx', engine='openpyxl')
    
    return Total_Act_Meta_Data_MOS
        
        
        
        
        