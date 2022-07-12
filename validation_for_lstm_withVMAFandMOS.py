from keras.models import load_model
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

import data_preprocessing_for_lstm_withVMAFandMOS as dpLSTM_withVMAFandMOS
import data_preprocessing_for_classification as dpClass
import validation_for_classification_withVMAF as valClass_withVMAF
import pickle as pkl
import pandas as pd
import numpy as np
import torch
import math
import glob
import warnings
import argparse
import os


warnings.filterwarnings('ignore')
np.set_printoptions(precision=2)
use_gpu = torch.cuda.is_available()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(12)



def LSTM_Validation_withVMAFandMOS(  dpLSTM_withVMAFandMOS, path_to_bitsream_dataset,Framrate,path_to_Result_LSTM, path_to_LSTM_Model, path_to_Normalization_for_LSTM, path_to_VMAF, path_to_Normalization_Models, path_to_CNN_Model, path_to_MOS, Bs_files):
        
        vmaf_excel=pd.read_excel(path_to_VMAF, engine='openpyxl')
        vmaf_excel.columns=range(vmaf_excel.shape[1])
        
        test_df = dpLSTM_withVMAFandMOS.ActFeatures_MetaData_Bitstream(path_to_bitsream_dataset, vmaf_excel, path_to_Result_LSTM, path_to_Normalization_Models, path_to_CNN_Model, path_to_MOS, Bs_files)
        
    
        
        ts_cols = list(test_df)[2:]
        
        df_for_testing = test_df[ts_cols].astype(float)
        
        
        with open(path_to_Normalization_for_LSTM, "rb") as infile:
            scaler = pkl.load(infile)
            df_for_testing_scaled = scaler.transform(df_for_testing)
                
            
            
        X_ts = df_for_testing_scaled
        Y_ts = test_df['MOS'].tolist()
        
        testX = []
        testY = []
        n_past = 15
        
        for i in range(0,len(X_ts),n_past):
          testX.append(X_ts[i:i+n_past,:])
          testY.append(Y_ts[i])
        
        
        testX = np.array(testX)
        
        testX, testY = np.array(testX), np.array(testY)
        testY=testY.reshape(len(testY), 1)
        
        #print('testX shape == {}.'.format(testX.shape))
        #print('testY shape == {}.'.format(testY.shape))
        
        
        model=load_model(path_to_LSTM_Model)
        
        Y_pred=model.predict(testX)
        
        prediction_copies = np.repeat(Y_pred, df_for_testing.shape[1], axis=-1)
        y_pred_future = scaler.inverse_transform(prediction_copies)[:,0]
        
        y_pred_future_new = (0.1189 * y_pred_future) + 1.2572
        corr, _ = pearsonr(testY.flatten(), y_pred_future_new)
        rmse = mean_squared_error(testY.flatten(), y_pred_future_new, squared=False)        
        plt.scatter(testY.flatten(), y_pred_future_new)
        plt.title('LSTM Performance, considering Actual VMAF and MOS exists \n  (PCC = {}, RMSE = {}) '.format(np.round(corr,2), np.round(rmse,2)), fontsize=12, fontweight='bold')
        plt.xlabel('Actual MOS', fontsize=12)
        plt.ylabel('Predicted MOS', fontsize=12)
        plt.show()
        
        temp=(test_df.iloc[:,0]).to_numpy()
        subsample = temp[::15]
        
        PredMOS_perSecond=pd.DataFrame(subsample)
        PredMOS_perSecond.insert(1,'Actual_MOS_perSecond',testY)
        PredMOS_perSecond.insert(2,'Predicted_MOS_perSecond',y_pred_future_new )
        PredMOS_perSecond = PredMOS_perSecond.rename(columns={0:'Game_Name',1:'Actual_MOS_perSecond',2:'Predicted_MOS_perSecond'})
        

        PredMOS_perSecond.to_excel(path_to_Result_LSTM + 'MOS_PredMOS_perSecond.xlsx')
        
        PredMOS_perVideo=pd.DataFrame()
        MOS_perVideo=[]
        PredMOS_perVid=[]
        Video_name=[]
        
        
        for i in range(0,len(PredMOS_perSecond),30):
            
          MOS_perVideo.append(np.mean(PredMOS_perSecond.iloc[i:i+30,1]))
          PredMOS_perVid.append(np.mean(PredMOS_perSecond.iloc[i:i+30,2]))
          Video_name.append(PredMOS_perSecond.iloc[i,0])
        
        PredMOS_perVideo.insert(0,'Game_Name',Video_name)
        PredMOS_perVideo.insert(1,'Actual_MOS_perVideo',MOS_perVideo)
        PredMOS_perVideo.insert(2,'PredMOS_perVideo',PredMOS_perVid)
        
        PredMOS_perVideo.to_excel(path_to_Result_LSTM + 'MOS_PredMOS_perVideo.xlsx')

        #return PredMOS_perSecond,PredMOS_perVideo

       
def model_execute(MOS, VMAF, ResPath, InputPath, ModelPath):   
    
    path_to_CNN_Model = ModelPath + 'Trained_Model_Classification'
    path_to_LSTM_Model = ModelPath + 'Trained_Model_LSTM/'
    path_to_Normalization_for_LSTM = ModelPath + 'Normalization_Models/Normalized_Model_LSTM.pkl'
    path_to_RandomForest_Model = ModelPath + 'Trained_Model_Randomforest.sav'
    path_to_Normalization_Models = ModelPath + 'Normalization_Models/'

    
    path_to_Result_CNN = ResPath + 'Res_WithVMAFandMOS/'
    path_to_Result_LSTM = ResPath + 'Res_WithVMAFandMOS/'

    path_to_bitsream_dataset =  InputPath
    
    path_to_MOS = InputPath + 'MOS/' + MOS
    path_to_VMAF = InputPath + 'VMAF/' + VMAF
    
    Bs_files=glob.glob(path_to_bitsream_dataset + "Bitstream_Dataset/**.csv")
    
    MOS_file=pd.read_excel(path_to_MOS, engine='openpyxl')
    
    vmaf_excel=pd.read_excel(path_to_VMAF, engine='openpyxl')
    vmaf_excel.columns=range(vmaf_excel.shape[1])
       
    Bitstream_Data,VMAF_Data = dpClass.Make_Total_VMAF_Bitstream(vmaf_excel, path_to_bitsream_dataset, path_to_Result_CNN)
    
    Framrate=30
    threshold=0
    
    LSTM_Validation_withVMAFandMOS( dpLSTM_withVMAFandMOS, path_to_bitsream_dataset,Framrate,path_to_Result_LSTM, path_to_LSTM_Model, path_to_Normalization_for_LSTM, path_to_VMAF, path_to_Normalization_Models, path_to_CNN_Model, path_to_MOS, Bs_files)
    valClass_withVMAF.CNN_Validation(VMAF_Data,Bitstream_Data, path_to_Normalization_Models, path_to_RandomForest_Model,path_to_CNN_Model,path_to_Result_CNN)







if __name__== "__main__":

    parser = argparse.ArgumentParser()
    

    parser.add_argument('-mo', '--MOS', action='store', dest='MOS', default=r'KUGVD_MOS.xlsx' ,
                    help='Specify the excel file name for MOS Scores, e.g. KUGVD_MOS.xlsx')
    
    parser.add_argument('-vf', '--VMAF', action='store', dest='VMAF', default=r'KUGVD_VMAF_PerFrameValues.xlsx' ,
                    help='Specify the excel file name for VMAF Scores, e.g. KUGVD_VMAF_PerFrameValues.xlsx')
    
    parser.add_argument('-rp', '--ResPath', action='store', dest='ResPath', default=r'../Dataset/Results/' , help='Specify the path to the outputs of the models, e.g. ../Dataset/Results/')
    
    parser.add_argument('-ip', '--InputPath', action='store', dest='InputPath', default=r'../Dataset/Inputs/' ,help='Specify the path to the Inputs files required for execution of the models, e.g. ../Dataset/Inputs/')
    
    parser.add_argument('-mp', '--ModelPath', action='store', dest='ModelPath', default=r'../Dataset/Models/' ,help='Specify the path to the location of models, e.g. ../Dataset/Models/')
    
    values = parser.parse_args()

    model_execute(values.MOS, values.VMAF, values.ResPath, values.InputPath, values.ModelPath);
        
















