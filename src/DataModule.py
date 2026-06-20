import pandas as pd 
from sqlalchemy import create_engine



class DataHandler:
    def __init__(self, server: str, driver: str, database: str ):
        self.server = server
        self.driver = driver
        self.database = database
        
    def read_sql_table(self,table_name:str):
        """Read table from sql database

        Args:
            table_name : str
        return:
                Pandas DataFrame    
        """
        if not isinstance(table_name, str):
            raise ValueError("table name must be a string")
        
        query = f"select * from {table_name}"
        
        try: 
            engine = create_engine(
                fr"mssql+pyodbc:///?odbc_connect=DRIVER={{{self.driver}}};SERVER={self.server};DATABASE={self.database};Trusted_Connection=yes;TrustServerCertificate=yes"
            )
            
            df = pd.read_sql(sql=query, con=engine)
            
            return df
        except Exception as e:
            print(e)
            
            
    @staticmethod
    def wrangle_data(path: str, index=None):
        
        """Read data from csv file into Clean Data Frame

        Args:
            path:str
                Path to the raw CSV file.
        returns:        
        -------
        pd.DataFrame
            Cleaned dataframe ready for modeling.
        """
        
        # Load data
        df = pd.read_csv(path, index_col=index)
        
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
                
             
    