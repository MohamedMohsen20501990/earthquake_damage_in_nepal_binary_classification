import os
import sys
from pathlib import Path
# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd 
from sqlalchemy import create_engine
from src.config import settings
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.preprocess import Preprocessor


from pydantic import BaseModel, field_validator
import re


@dataclass
class DataConfig():
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_data_path: str=os.path.join('artifacts', 'data.csv')

class TableIdentifier(BaseModel):
    """To avoid sql injection"""
    table_name:str
    
    
    @field_validator("table_name")
    @classmethod
    def validate_table_name(cls, value:str) -> str:
        pattern = r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$"
        if not re.fullmatch(pattern,value):
            raise ValueError("Invalid table name")
        return value


class DataHandler:
    """
    Handle data reading, processing, and saving operations.

    Provides utilities for reading data from SQL Server and CSV files,
    wrangling data, and saving processed DataFrames to SQL or CSV.
    """

    def __init__(self, server: str, driver: str, database: str ):
        self.data_config = DataConfig()
        self.server = server
        self.driver = driver
        self.database = database 
        self.engine = create_engine(
                f"mssql+pyodbc://{settings.DATABASE_USERNAME}:{settings.DATABASE_PASSWORD}@{self.server}:1433/{self.database}"
                f"?driver={self.driver.replace(' ', '+')}"
                f"&TrustServerCertificate=yes"
            )
        
    def read_sql_table(self,table_name:str):
        """Read table from sql database

        Args:
            table_name : str
        return:
                Pandas DataFrame    
        """
        # To avoid sql injection
        validated = TableIdentifier(table_name=table_name)
        
        query = f"select * from {validated.table_name}"
        
        try: 
            df = pd.read_sql(sql=query, con=self.engine)
            return df
        except Exception as e:
            print(f"Database error: {e}")
            raise

            
            
            
    @staticmethod
    def wrangle_data(data_frame:pd.DataFrame | None = None ,path: str | None =None , index_col=None):
        
        """Read data from csv file into Clean Data Frame

        Args:
            path:str
                Path to the raw CSV file.
        returns:        
        -------
        pd.DataFrame
            Cleaned dataframe ready for modeling.
        """
        if data_frame is None and path is None:
            raise ValueError("Etither DataFrame or path must be provided")
        
        if data_frame is not None and path is not None:
            raise ValueError("Privide either a DataFrame or path")
        
        if data_frame is not None:
            df = data_frame.copy()
        else:
            df = pd.read_csv(path, index_col=index_col)
                
            
            
        # Mask for buildings located in "GORKHA"
        df = df[df["district_id"]==36]
        
        # encoding damage_grade columns, our target column
        df["damage_grade"]=df["damage_grade"].str[-1].astype(int)
        
        # Extracing severe damage column
        df["severe_damage"] = df["damage_grade"].apply(lambda x: x>3).astype(int)
        
        # Drop_list for leakage-multicollinearity
        
        # leakage
        drop_list = [col for col in df.columns if "post_eq" in col]
        
        # drop damage grade columns
        drop_list.append("damage_grade")
        
        # drop multicollinareaity columns "count_floors_pre_eq"
        drop_list.append("count_floors_pre_eq")
        
        
        # Drop low and high cardilality features
        drop_list.append("district_id")
        
        df.drop(columns=drop_list, inplace=True)
        
        
        return df
    
    
    def save_to_sql(self, data_frame: pd.DataFrame, table_name:str):
        """Save a pandas DataFrame to a SQL database table.

            The table name is validated using `TableIdentifier` and must contain
            a schema and table name separated by a dot. If the table already
            exists, it is replaced.

            Parameters
            ----------
            data_frame : pd.DataFrame
                The DataFrame containing the data to be saved.
            table_name : str
                Fully qualified table name in the format ``schema.table``.

            Raises
            ------
            Exception
                Re-raises any exception encountered while validating the table
                name or writing the DataFrame to the database.
            """
        validated = TableIdentifier(table_name=table_name)
        try:
            schema,table = validated.table_name.split(".")
            data_frame.to_sql(name=table, schema=schema, con=self.engine, if_exists="replace", index=False)
        except Exception as e:
            print(f"DataBase error{e}")
            raise    
    
                
                
    def split_and_save(self, data_frame: pd.DataFrame):
        """
        Save a pandas DataFrame to a CSV file.

        Parameters
        ----------
        data_frame : pd.DataFrame
            The DataFrame containing the data to be saved.
        path : str
            The file path where the CSV file will be saved.
        """
        os.makedirs(os.path.dirname(self.data_config.train_data_path), exist_ok=True)
        train_data, test_data = train_test_split(data_frame, test_size=0.2, random_state=42, stratify=data_frame["severe_damage"])
        try:
            train_data.to_csv(self.data_config.train_data_path, index=False, header=True)
            test_data.to_csv(self.data_config.test_data_path, index=False, header=True)
            data_frame.to_csv(self.data_config.raw_data_path, index=False, header=True)
        except Exception as e:
            print(f"Error saving file: {e}")    
        

                
if __name__ =="__main__":
    data_handler = DataHandler(server="localhost", driver="ODBC Driver 18 for Sql Server", database="MyDatabase")
    raw_df = data_handler.read_sql_table("dbo.earthquake_data")
    df = data_handler.wrangle_data(data_frame=raw_df)
    data_handler.split_and_save(df)   
    
    data_preprocessor = Preprocessor()
    data_preprocessor.start_data_preprocessing(train_path = data_handler.data_config.train_data_path, test_path=data_handler.data_config.test_data_path)
    
    