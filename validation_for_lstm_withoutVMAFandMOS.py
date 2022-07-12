from keras.models import load_model
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import data_preprocessing_for_lstm_withoutVMAFandMOS as dpLSTM_withoutVMAFandMOS
import data_preprocessing_for_classification as dpClass
import validation_for_classification_withoutVMAF as valClass_withoutVMAF
import pickle as pkl
import pandas as pd
import numpy as np
import argparse
import torch
import math
import glob
import warnings
import os

warnings.filterwarnings('ignore')

np.set_printoptions(precision=2)
use_gpu = torch.cuda.is_available()
np.random.seed(12)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def LSTM_Validation( Bitstream_Data, path_to_bitsream_dataset,path_to_Result_LSTM, path_to_LSTM_Model, path_to_Normalization_for_LSTM, path_to_Normalization_Models, path_to_CNN_Model, Bs_files):
        
        test_df=dpLSTM_withoutVMAFandMOS.ActFeatures_MetaData_Bitstream(Bitstream_Data,path_to_Result_LSTM, path_to_Normalization_Models, path_to_CNN_Model)
        
        ts_cols = list(test_df)[1:]
        
        df_for_testing = test_df[ts_cols].astype(float)
        
        
        with open(path_to_Normalization_for_LSTM, "rb") as infile:
            scaler = pkl.load(infile)
            df_for_testing_scaled = scaler.transform(df_for_testing)
                
            
        X_ts = df_for_testing_scaled
        
        testX = []
        n_past = 15
        
        for i in range(0,len(X_ts),n_past):
          testX.append(X_ts[i:i+n_past,:])
        
        testX = np.array(testX)
                    
        #print('testX shape == {}.'.format(testX.shape))
        
        
        model=load_model(path_to_LSTM_Model)
        
        Y_pred=model.predict(testX)
        
        prediction_copies = np.repeat(Y_pred, df_for_testing.shape[1], axis=-1)
        y_pred_future = scaler.inverse_transform(prediction_copies)[:,0]
        
        y_pred_future_new = (0.1189 * y_pred_future) + 1.2572        
        
     
        
        temp=(test_df.iloc[:,0]).to_numpy()
        subsample = temp[::15]
        
        
        PredMOS_perSecond=pd.DataFrame(subsample)
        
        PredMOS_perSecond.insert(1,'Predicted_MOS_perSecond',y_pred_future_new )
        PredMOS_perSecond=PredMOS_perSecond.rename(columns={0:'Game_Name',1:'Predicted_MOS_perSecond'})
        
        PredMOS_perSecond.to_excel(path_to_Result_LSTM + 'PredMOS_perSecond.xlsx')
        
        PredMOS_perVideo=pd.DataFrame()
        PredMOS_perVid=[]
        Video_name=[]
        
        
        for i in range(0,len(PredMOS_perSecond),30):
          PredMOS_perVid.append(np.mean(PredMOS_perSecond.iloc[i:i+30,1]))
          Video_name.append(PredMOS_perSecond.iloc[i,0])
        
        PredMOS_perVideo.insert(0,'Game_Name',Video_name)
        PredMOS_perVideo.insert(1,'PredMOS_perVideo',PredMOS_perVid)
        
        PredMOS_perVideo.to_excel(path_to_Result_LSTM + 'PredMOS_perVideo.xlsx')
            

        #return PredMOS_perSecond,PredMOS_perVideo

        
def model_execute(ResPath, InputPath, ModelPath):   
    
    path_to_CNN_Model = ModelPath + 'Trained_Model_Classification'
    path_to_LSTM_Model = ModelPath + 'Trained_Model_LSTM/'
    path_to_Normalization_for_LSTM = ModelPath + 'Normalization_Models/Normalized_Model_LSTM.pkl'
    path_to_RandomForest_Model = ModelPath + 'Trained_Model_Randomforest.sav'
    path_to_Normalization_Models = ModelPath + 'Normalization_Models/'

    path_to_Result_CNN = ResPath + 'Res_WithoutVMAFandMOS/'
    path_to_Result_LSTM = ResPath + 'Res_WithoutVMAFandMOS/'

    path_to_bitsream_dataset =  InputPath

    
    Bs_files=glob.glob(path_to_bitsream_dataset + "Bitstream_Dataset/**.csv")
    #print(Bs_files[0])
    Bitstream_Data = dpClass.Collect_Bitstream_Files(Bs_files)
    #print(Bitstream_Data.iloc[0,:10])

    Framrate=30
    threshold=0


    LSTM_Validation(Bitstream_Data, path_to_bitsream_dataset,path_to_Result_LSTM, path_to_LSTM_Model, path_to_Normalization_for_LSTM, path_to_Normalization_Models, path_to_CNN_Model, Bs_files)#,Framrate

    valClass_withoutVMAF.CNN_Validation(Bitstream_Data, path_to_Normalization_Models, path_to_RandomForest_Model,path_to_CNN_Model,path_to_Result_CNN)



if __name__== "__main__":

    parser = argparse.ArgumentParser()
    
    
    parser.add_argument('-rp', '--ResPath', action='store', dest='ResPath', default=r'../Dataset/Results/' ,
                    help='Specify the path to the outputs of the models, e.g. ../Dataset/Results/')
    
    parser.add_argument('-ip', '--InputPath', action='store', dest='InputPath', default=r'../Dataset/Inputs/' ,
                    help='Specify the path to the Inputs files required for execution of the models, e.g. ../Dataset/Inputs/')
    
    parser.add_argument('-mp', '--ModelPath', action='store', dest='ModelPath', default=r'../Dataset/Models/' ,
                    help='Specify the path to the location of models, e.g. ../Dataset/Models/')
    
    values = parser.parse_args()

    model_execute(values.ResPath, values.InputPath, values.ModelPath);


















