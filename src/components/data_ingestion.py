import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation,DataTransfomerConfig

from src.components.model_trainer import ModelTrainer,ModelTrainerConfig

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

#creating the artifacts folder path for data
@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the Data ingestion method/component")
        try:
            df=pd.read_csv(r'notebook\data\StudentsPerformance.csv')
            logging.info("Read the dataset as dataframe")
            
            ## pointing the file path to make a directory in the file system
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            
            ## storing the raw data in the path
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train Test Split Started")

            train,test=train_test_split(df,test_size=0.25,random_state=43)

            train.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return self.ingestion_config.train_data_path,self.ingestion_config.test_data_path
        
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()    
    train_dath_path,test_data_path=obj.initiate_data_ingestion()

    dt_obj=DataTransformation()
    train_arr,test_arr,_=dt_obj.initiate_data_transformation(train_dath_path,test_data_path)

    model=ModelTrainer()
    print(model.initiate_model_training(train_arr=train_arr,test_arr=test_arr))


            
    
         