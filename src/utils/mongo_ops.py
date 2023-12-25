import os
from dotenv import load_dotenv
from typing import Optional

import numpy as np
import pandas as pd
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

load_dotenv()

MONGO_DB_URI = os.environ["MONGO_DB_URI"]
MONGO_DATABSE_NAME = os.environ["MONGO_DATABASE_NAME"]

# ca = certifi.where()


class MongoDBOps:
    # client = None

    def __init__(self, database_name=MONGO_DATABSE_NAME) -> None:
        try:
            try:
                self.client = MongoDBOps.client
                self.database_name = self.client[database_name]
            except AttributeError:
                mongo_db_uri = MONGO_DB_URI
                self.client = MongoClient(mongo_db_uri, server_api=ServerApi("1"))
                self.database = self.client[database_name]
                self.database_name = database_name
        except Exception as e:
            raise e

    def export_collection_as_dataframe(
        self,
        collection_name: str,
        rows_to_load: int,
        database_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        export entire collectin as dataframe:
        return pd.DataFrame of collection
        """
        if database_name is None:
            collection = self.client[self.database_name][collection_name]

        else:
            collection = self.client[database_name][collection_name]

        df = pd.DataFrame(list(collection.find().limit(rows_to_load)))

        if "_id" in df.columns.to_list():
            df = df.drop(columns=["_id"], axis=1)

        df.replace({"na": np.nan}, inplace=True)

        return df

    def insert_many(self, collection_name, records: list):
        self.client[self.database_name][collection_name].insert_many(records)

    def insert(self, collection_name, record):
        self.client[self.database_name][collection_name].insert_one(record)
