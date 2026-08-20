from pydantic import BaseModel


class ModelNameConfig(BaseModel):
    """Model configurations."""

    model_name: str = "lightgbm"
    fine_tuning: bool = False