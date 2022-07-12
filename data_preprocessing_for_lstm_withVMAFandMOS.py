
from sklearn.preprocessing import LabelEncoder
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
  valid_dataset = torch.utils.data.TensorDataset(bitstream_tensor)#, vmaf_train_tensor
  valid_dl = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8) 

  return valid_dl

def valid_Additive_Frame_size_Loader(Add_Frame_size_tensor):
  valid_additive_data_1 = torch.utils.data.TensorDataset(Add_Frame_size_tensor)#vmaf_train_tensor
  valid_dl_additive_1 = torch.utils.data.DataLoader(valid_additive_data_1, batch_size=batch_size, shuffle=False, num_workers=8)

  return valid_dl_additive_1

def valid_Additive_Frame_type_Loader(Add_Frame_type_tensor):
  valid_additive_data_2 = torch.utils.data.TensorDataset(Add_Frame_type_tensor)#, vmaf_train_tensor
  valid_dl_additive_2 = torch.utils.data.DataLoader(valid_additive_data_2, batch_size=batch_size, shuffle=False, num_workers=8)

  return valid_dl_additive_2

def valid_Additive_Preset_Loader(Add_Preset_tensor):
  valid_additive_data_3 = torch.utils.data.TensorDataset(Add_Preset_tensor)#, vmaf_train_tensor

  valid_dl_additive_3 = torch.utils.data.DataLoader(valid_additive_data_3, batch_size=batch_size, shuffle=False, num_workers=8)
  return valid_dl_additive_3



def GetMeanVMAF_PerSecond_PerVideo(vmaf_file,Framrate):
    temp=[]
    #New_df=pd.DataFrame()
    Mean_VMAF_perVideo=[]
    Mean_VMAF_perSec=[]

    for i in range(0,len(vmaf_file),Framrate):
      temp.extend([np.repeat((np.mean(vmaf_file['Vmaf'][i:i+Framrate])), Framrate)])

    Mean_VMAF_perSec = [j for i in temp for j in i]
    vmaf_file.insert(11,'Mean_VMAF_perSec',Mean_VMAF_perSec)

    Mean_VMAF_perVideo = np.mean(vmaf_file['Vmaf'][:])
    vmaf_file.insert(12,'Mean_VMAF_perVideo',Mean_VMAF_perVideo)

    vmaf_file=vmaf_file[(vmaf_file.index % Framrate == 0)]
    vmaf_file.drop(['Vmaf'], axis=1)
    
    return vmaf_file

def Make_VMAF_Intervals(vmaf_file,threshold):

  min=threshold

  for i in range(len(vmaf_file)):

    if (vmaf_file['Vmaf'][i]>=min).any() and (vmaf_file['Vmaf'][i]<min+17).any() :
        vmaf_file['Target_class'][i]=0
    
    elif (vmaf_file['Vmaf'][i]>min+23).any()  and (vmaf_file['Vmaf'][i]<min+37).any() :
        vmaf_file['Target_class'][i]=1
    
    elif (vmaf_file['Vmaf'][i]>min+43).any()  and (vmaf_file['Vmaf'][i]<min+57).any() :
        vmaf_file['Target_class'][i]=2

    elif (vmaf_file['Vmaf'][i]>min+63).any()  and (vmaf_file['Vmaf'][i]<min+77).any() :
        vmaf_file['Target_class'][i]=3
        
    elif (vmaf_file['Vmaf'][i]>min+83).any() :
        vmaf_file['Target_class'][i]=4

    else:
        vmaf_file['Target_class'][i]=5

  return vmaf_file

def SubSample(file, Framrate):

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

def Concat_MOS_vmaf(classified_VMAF_values, MOS_file, Framrate):

    classified_VMAF_values = classified_VMAF_values.reset_index(drop=True)
    MOS_file = MOS_file.reset_index(drop=True)


    classified_VMAF_values.insert(13,'MOS', np.nan)

    MOS_file.columns = ['Game', 'Framrate', 'Duration', 'Version','Resolution', 'Bitrate', 'Codec' ,'MOS','Vmaf']

    for i in range(len(MOS_file)):
      for j in range(len(classified_VMAF_values)):
        if ((MOS_file['Game'][i]==classified_VMAF_values['Game'][j]) and (MOS_file['Bitrate'][i]==classified_VMAF_values['Bitrate'][j])):
          classified_VMAF_values['MOS'][j] = MOS_file['MOS'][i]*(classified_VMAF_values['Mean_VMAF_perSec'][j] / classified_VMAF_values['Mean_VMAF_perVideo'][j])

    return classified_VMAF_values

def Make_Seconds_Order(vmaf_file,Framrate):

  x = np.arange(0,Framrate)
  y = [Framrate]
  Seconds_Order_list= np.repeat(x, y)
  vmaf_file.insert(1,'Seconds', Seconds_Order_list ) 
  
  return vmaf_file

def Extracting_Activation_Features(df_1,path_to_CNN_Model, valid_dl, valid_Additive_Frame_size_dl, valid_Additive_Frame_type_dl, valid_Additive_Preset_dl,Add_Frame_size,Add_Frame_type,Add_Preset):

      activation = {}
      Activation_Features=pd.DataFrame()
      ActivationFeatures_Fc2=[]
      vmaf_ActFeatures=pd.DataFrame()
      vmaf_ActFeatures_FrameType_FrameSize_Preset=pd.DataFrame()

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

      vmaf_ActFeatures=pd.concat([df_1['Game_Name'].reset_index(drop=True),df_1['MOS'].reset_index(drop=True), Activation_Features.reset_index(drop=True)], axis = 1)
      vmaf_ActFeatures.columns = range(vmaf_ActFeatures.shape[1])

      vmaf_ActFeatures_FrameType_FrameSize_Preset=pd.concat([vmaf_ActFeatures.reset_index(drop=True),pd.DataFrame(Add_Frame_size).reset_index(drop=True),pd.DataFrame(Add_Frame_type).reset_index(drop=True),pd.DataFrame(Add_Preset).reset_index(drop=True)], axis = 1)
      vmaf_ActFeatures_FrameType_FrameSize_Preset.columns = range(vmaf_ActFeatures_FrameType_FrameSize_Preset.shape[1])
      vmaf_ActFeatures_FrameType_FrameSize_Preset = vmaf_ActFeatures_FrameType_FrameSize_Preset.rename(columns={0: 'Game_Name', 1: 'MOS', 2:'Act_feature1' ,3: 'Act_feature2', 4: 'Act_feature3', 5: 'Act_feature4', 6: 'Act_feature5', 7: 'Add_Frame_size', 8: 'Add_Frame_type', 9: 'Add_Preset'})

      return vmaf_ActFeatures_FrameType_FrameSize_Preset

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
    qpValues=df_1.iloc[:,6:].to_numpy()


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

def Make_Total_VMAF_MOS(vmaf_excel, MOS_file):
    
    Total_VMAF_MOS = pd.DataFrame()

    
    for i in range(len(vmaf_excel)):
            vmaf_features = pd.DataFrame()
            vmaf_value = pd.DataFrame()
            vmaf_features =  vmaf_features.append(pd.Series(vmaf_excel.iloc[i,:7]), ignore_index=True)
            vmaf_value= vmaf_value.append(vmaf_excel.iloc[i,7:], ignore_index=True)
    
    
    
            vmaf_features=pd.DataFrame(np.repeat(vmaf_features.values,vmaf_value.shape[1],axis=0))
    
           
    
    
            vmaf_value=vmaf_value.to_numpy()
            vmaf_value=vmaf_value.reshape((vmaf_value.shape[0]*vmaf_value.shape[1], 1))
            vmaf_value=pd.DataFrame(vmaf_value)
            vmaf_file = pd.concat([vmaf_features, vmaf_value], axis = 1, ignore_index=True)
            vmaf_file.insert(7, "7", 'veryfast')
            vmaf_file.columns = ['Game','Framrate', 'Duration', 'Version','Resolution', 'Bitrate', 'Codec',  'Preset','Vmaf']
    
    
            vmaf_file.insert(8, "Target_class", np.nan)
            classified_VMAF_values = Make_VMAF_Intervals(vmaf_file,threshold)
    
            classified_VMAF_values= Make_Seconds_Order(classified_VMAF_values,Framrate)
    
            classified_VMAF_values = GetMeanVMAF_PerSecond_PerVideo(classified_VMAF_values, Framrate)
    
            classified_VMAF_values= Concat_MOS_vmaf(classified_VMAF_values, MOS_file, Framrate)
    
            #vmaf_file=SubSample(vmaf_file,Framrate)
            Total_VMAF_MOS = pd.concat((Total_VMAF_MOS,classified_VMAF_values),axis=0,ignore_index=True )#df.reset_index(drop=True, inplace=True)
    
            #Total_Act_Meta_Data = ActFeatures_MetaData_Bitstream(vmaf_file,Framrate)

    #Total_VMAF_MOS.to_excel(path_to_Result_LSTM + '/VMAF_MOS.xlsx' , engine='openpyxl')
    return Total_VMAF_MOS


def ActFeatures_MetaData_Bitstream(path_to_bitsream_dataset, vmaf_excel, path_to_Result_LSTM, path_to_Normalization_Models, path_to_CNN_Model, path_to_MOS, Bs_files):
    path_list=[]
    count1=0
    Total_Act_Meta_Data_MOS=pd.DataFrame()
    
    MOS_file = pd.read_excel(path_to_MOS, engine='openpyxl')
    
    Total_VMAF_MOS = Make_Total_VMAF_MOS(vmaf_excel, MOS_file)
    
    for i in range(0,len(Total_VMAF_MOS),Framrate):
    
        path_game_name = str(Total_VMAF_MOS['Game'][i])
        path_Resolution = str(Total_VMAF_MOS['Resolution'][i])
        path_bitrate = str(int(Total_VMAF_MOS['Bitrate'][i]))
        path_codec = str(Total_VMAF_MOS['Codec'][i])
        path_Version = str(Total_VMAF_MOS['Version'][i])
    
        path1= path_to_bitsream_dataset + 'Bitstream_Dataset/' +path_game_name + '_' + '30fps_30sec'  + '_' + path_Version + '_' + path_Resolution + '_' + path_bitrate +'_'+ path_codec + '.csv'
    
    
        if os.path.exists(path1):
    
            count1+=1
            path_list.append(path1)
            df_1 = pd.read_csv(path1, skiprows=[0],header=None )
    
            df_1.insert(0,'Game_Name',os.path.basename(path1))
            temp_preset = (np.repeat(Total_VMAF_MOS.iloc[i,:]['Preset'], df_1.shape[0])).tolist()
            temp_MOS=np.repeat(Total_VMAF_MOS.iloc[i:i+Framrate,:]['MOS'].tolist(), Framrate)
    
    
            df_1.insert(4,'Preset',temp_preset)
            df_1.insert(5,'MOS',temp_MOS)
    
    
    
            df_1.columns=range(df_1.shape[1])
            df_1 = df_1.rename(columns={0: 'Game_Name', 1: 'Frame_Type',2: 'Frame_Size' ,3: 'Avg_qp', 4: 'Preset', 5: 'MOS'})
    
    
            df_1 = SubSample(df_1,Framrate)
            Act_Meta_Data_MOS = Get_ActFeatures_MetaData(df_1, path_to_Normalization_Models, path_to_CNN_Model)
            Total_Act_Meta_Data_MOS = pd.concat((Total_Act_Meta_Data_MOS,Act_Meta_Data_MOS),axis=0)
    
    
    #Total_Act_Meta_Data_MOS.to_excel(path_to_Result_LSTM + 'Total_Act_Meta_Data_MOS.xlsx', engine='openpyxl')
    
    return Total_Act_Meta_Data_MOS
        
        
        
#ActFeatures_MetaData_Bitstream(path_to_bitsream_dataset,vmaf_excel,Framrate)       
        
        
        