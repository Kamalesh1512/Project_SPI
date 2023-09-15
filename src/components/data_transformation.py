import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from dataclasses import dataclass

@dataclass
class DataTransfomerConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config=DataTransfomerConfig()

    def data_transformation(self):
        '''
        This function helps in data tansfromation
        '''

        try:
            numerical_columns=["writing score","reading score"]

            categorical_columns=["gender","race/ethnicity",
                                 "parental level of education","lunch","test preparation course"]
            
            num_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy='median')),
                ("scaler",StandardScaler(with_mean=False))
                 ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder(drop="first")),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"categorical columns:{categorical_columns}")
            logging.info(f"numerical columns:{numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
    
        except Exception as e:
            raise CustomException(e,sys)
    

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            logging.info("obtaining preprocessing object")

            preprocess_obj=self.data_transformation()
            
            target_column_name='math score'
            numerical_columns=['writing score','reading score']

            input_features_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_train_df=train_df[target_column_name]

            input_features_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_test_df=test_df[target_column_name]

            logging.info("Applying preprocessing obj on train_df,test_df")

            
            input_features_train_transformed=preprocess_obj.fit_transform(input_features_train_df)
            input_features_test_transformed=preprocess_obj.transform(input_features_test_df)

            train_arr=np.c_[input_features_train_transformed,np.array(target_train_df)]

            test_arr=np.c_[input_features_test_transformed,np.array(target_test_df)] 

            logging.info(f"saved preprocessing object")

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocess_obj)

            return train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path
        except Exception as e:
            raise CustomException(e,sys)
            

        
