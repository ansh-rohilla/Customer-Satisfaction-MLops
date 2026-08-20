import os
import pickle
from typing import Any, Type, Union

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer


DEFAULT_FILENAME = "CustomerSatisfactionEnvironment"


class cs_materializer(BaseMaterializer):
    """
    Custom materializer for the Customer Satisfaction Project.
    """

    ASSOCIATED_TYPES = (
        str,
        np.ndarray,
        pd.Series,
        pd.DataFrame,
        CatBoostRegressor,
        RandomForestRegressor,
        LGBMRegressor,
        XGBRegressor,
    )

    def handle_input(
        self,
        data_type: Type[Any],
    ) -> Any:
        """
        Load an object from the ZenML artifact store.
        """

        super().handle_input(data_type)

        filepath = os.path.join(
            self.artifact.uri,
            DEFAULT_FILENAME,
        )

        with fileio.open(filepath, "rb") as fid:
            return pickle.load(fid)

    def handle_return(
        self,
        obj: Any,
    ) -> None:
        """
        Save an object to the ZenML artifact store.
        """

        super().handle_return(obj)

        filepath = os.path.join(
            self.artifact.uri,
            DEFAULT_FILENAME,
        )

        with fileio.open(filepath, "wb") as fid:
            pickle.dump(obj, fid)