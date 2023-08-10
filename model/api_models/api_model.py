'''
 # @ Author: Bao Loc Pham
 # @ Create Time: 2023-08-06 11:22:16
 # @ Modified by: Bao Loc Pham
 # @ Modified time: 2023-08-06 11:23:27
 # @ Description:
 '''

from pydantic import BaseModel
from typing import Literal


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
    workclass: Literal[
        'State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov',
        'Local-gov', 'Self-emp-inc', 'Without-pay']
    education: Literal[
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th',
        'Some-college',
        'Assoc-acdm', '7th-8th', 'Doctorate', 'Assoc-voc', 'Prof-school',
        '5th-6th', '10th', 'Preschool', '12th', '1st-4th']
    marital_status: Literal[
        'Never-married', 'Married-civ-spouse', 'Divorced',
        'Married-spouse-absent', 'Separated', 'Married-AF-spouse',
        'Widowed']
    occupation: Literal[
        'Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',
        'Prof-specialty', 'Other-service', 'Sales', 'Transport-moving',
        'Farming-fishing', 'Machine-op-inspct', 'Tech-support',
        'Craft-repair', 'Protective-serv', 'Armed-Forces',
        'Priv-house-serv']
    relationship: Literal[
        'Not-in-family', 'Husband', 'Wife', 'Own-child',
        'Unmarried', 'Other-relative']
    race: Literal[
        'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo',
        'Other']
    sex: Literal['Male', 'Female']
    native_country: Literal[
        'United-States', 'Cuba', 'Jamaica', 'India', 'Mexico',
        'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran',
        'Philippines', 'Poland', 'Columbia', 'Cambodia', 'Thailand',
        'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
        'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala',
        'Italy', 'China', 'South', 'Japan', 'Yugoslavia', 'Peru',
        'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
        'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary',
        'Holand-Netherlands']


class PredictResponse(BaseModel):
    """Defining the response body of an inference API"""
    prediction: int
    class_name: str
    success: bool
