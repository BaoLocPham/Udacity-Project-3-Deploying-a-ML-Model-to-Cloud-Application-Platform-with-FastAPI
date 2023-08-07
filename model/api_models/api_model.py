'''
 # @ Author: Bao Loc Pham
 # @ Create Time: 2023-08-06 11:22:16
 # @ Modified by: Bao Loc Pham
 # @ Modified time: 2023-08-06 11:23:27
 # @ Description:
 '''

from pydantic import BaseModel


class CensusRequest(BaseModel):
    """Define UserRequest

    Attributes:
        - age, hours_per_week: continous columns
        - workclass, education,
        marital_status, occupation,
        relationship, race, sex, native_country:
        categorical columns.
    """
    age: int
    hours_per_week: int
    workclass: str
    education: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str


class PredictResponse(BaseModel):
    """Defining the response body of an inference API"""
    prediction: int
    class_name: str
    success: bool
