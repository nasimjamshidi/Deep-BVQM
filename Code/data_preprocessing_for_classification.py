
from random import sample
import numpy as np
import pandas as pd

import torch
import os.path
import os
import glob

import warnings
warnings.filterwarnings('ignore')

np.set_printoptions(precision=2)
use_gpu = torch.cuda.is_available()
np.random.seed(12)
count1=0
threshold=0



"""**Collect_Bitstream_Files** function gets the bitstram information of each frame for all videos and store them as one single file."""

def Collect_Bitstream_Files(Bs_files):
    Total_Bitstream= pd.DataFrame()
    
    for filename in Bs_files:
      temp = pd.read_csv(filename, skiprows=[0],header=None,  index_col=False )
      temp.insert(0,'Game_name',os.path.basename(filename))
      temp.columns=range(temp.shape[1])
      Total_Bitstream=Total_Bitstream.append(temp)
  
    return Total_Bitstream

"""**Make_VMAF_Intervals** function aims to divide vmaf values into five intervals to be used for classification task"""

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

"""**Make_Same_Frequency_VMAF_Intervals** function aims to divide vmaf values into five intervals to be used for classification task"""

def Make_Same_Frequency_VMAF_Intervals(Total_VMAF,Total_Bitstream):
      
    new_vmaf=Total_VMAF
    new_bs=Total_Bitstream
    Target0=new_vmaf.query('Target_class == 0')
    Target1=new_vmaf.query('Target_class == 1')
    Target2=new_vmaf.query('Target_class == 2')
    Target3=new_vmaf.query('Target_class == 3')
    Target4=new_vmaf.query('Target_class == 4')


    Index_Target0 = Target0.index.tolist()
    Index_Target1 = Target1.index.tolist()
    Index_Target2 = sample(Target2.index.tolist(),min(len(Target2), len(Index_Target1)))
    Index_Target3 = sample(Target3.index.tolist(),min(len(Target3), len(Index_Target1)))
    Index_Target4 = sample(Target4.index.tolist(),min(len(Target4), len(Index_Target1)))


    remained_index=Index_Target0 + Index_Target1 + Index_Target2 + Index_Target3 + Index_Target4
    new_vmaf = new_vmaf.iloc[remained_index]
    new_bs = new_bs.iloc[remained_index]

    return new_vmaf,new_bs



def Make_Total_VMAF_Bitstream(vmaf_excel, path_to_bitsream_dataset, path_to_Result_CNN):
    
    
    Total_VMAF = pd.DataFrame()
    Total_Bitstream= pd.DataFrame()
    path_list=[]
    error_list=[]
    error_list2=[]
    count1=0
    threshold=0
    
    #len(vmaf_excel)
    for i in range(len(vmaf_excel)):
      try:
            vmaf_features = pd.DataFrame()
            vmaf_value = pd.DataFrame()
    
            vmaf_features =  vmaf_features.append(pd.Series(vmaf_excel.iloc[i,:7]), ignore_index=True)
            vmaf_value= vmaf_value.append(vmaf_excel.iloc[i,7:], ignore_index=True)
            vmaf_features=pd.DataFrame(np.repeat(vmaf_features.values,vmaf_value.shape[1],axis=0))
    
    
            vmaf_value=vmaf_value.to_numpy()
            vmaf_value=vmaf_value.reshape((vmaf_value.shape[0]*vmaf_value.shape[1], 1))
            vmaf_value=pd.DataFrame(vmaf_value)
    
            vmaf_file = pd.concat([vmaf_features, vmaf_value], axis = 1, ignore_index=True)
            vmaf_file.columns = ['Game', 'Framrate', 'Duration','Version','Resolution', 'Bitrate', 'Codec','Vmaf']
            vmaf_file.insert(1, "Target_class", np.nan)
            classified_VMAF_values = Make_VMAF_Intervals(vmaf_file,threshold)
            #print(path1)
            path_game_name = str(classified_VMAF_values['Game'][0])
            path_Resolution = str(classified_VMAF_values['Resolution'][0])
            path_bitrate = str(int(classified_VMAF_values['Bitrate'][0]))
            path_codec = str(classified_VMAF_values['Codec'][0])
            path_Version = str(classified_VMAF_values['Version'][0])
    
            path1= path_to_bitsream_dataset + 'Bitstream_Dataset/'+ path_game_name + '_' + '30fps_30sec'  + '_' + path_Version + '_' + path_Resolution + '_' + path_bitrate +'_'+ path_codec + '.csv'
            if os.path.exists(path1):
              count1+=1
              path_list.append(path1)
              df_1 = pd.read_csv(path1, skiprows=[0],header=None )
    
              selected_idx = classified_VMAF_values[classified_VMAF_values.iloc[:,1]!=5].index.values
              classified_VMAF_values = classified_VMAF_values.iloc[selected_idx]
              Bss = df_1.iloc[selected_idx]

              # Bs_short=Bss[np.mod(np.arange(Bss.index.size),10)==0]
              Bss.insert(1, "path", np.nan)
              Bss['path']=os.path.basename(path1)
              # vmaf_short=classified_VMAF_values[np.mod(np.arange(classified_VMAF_values.index.size),10)==0]
              Total_VMAF=Total_VMAF.append(classified_VMAF_values, ignore_index=True )

              Total_Bitstream=Total_Bitstream.append(Bss, ignore_index=True )
    
      except IndexError: 
            error_list.append(path_list[-1])
            pass
      except: 
            error_list2.append(path_list[-1])
            pass
    
    
    Adjusted_VMAF, Adjusted_Bs = Make_Same_Frequency_VMAF_Intervals(Total_VMAF,Total_Bitstream)
    #Adjusted_Bs.to_csv( path_to_Result_CNN + 'Total_Bitstream.csv')
    #Adjusted_VMAF.to_excel( path_to_Result_CNN + 'Total_VMAF.xlsx', engine='openpyxl')

    return Adjusted_Bs, Adjusted_VMAF

