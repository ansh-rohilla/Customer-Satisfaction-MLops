import logging
from typing import Tuple

import pandas as pd
from typing_extensions import Annotated

from model.data_cleaning import (
    DataCleaning,
    DataDivideStrategy,
    DataPreprocessStrategy,
)
from zenml import step


@step
def clean_data(
    data: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """Preprocess the data and divide it into training and testing sets."""

    try:
        preprocess_strategy = DataPreprocessStrategy()

        data_cleaning = DataCleaning(
            data,
            preprocess_strategy,
        )

        preprocessed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()

        data_cleaning = DataCleaning(
            preprocessed_data,
            divide_strategy,
        )

        x_train, x_test, y_train, y_test = (
            data_cleaning.handle_data()
        )

        return x_train, x_test, y_train, y_test

    except Exception as e:
        logging.error(e)
        raise