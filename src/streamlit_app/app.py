from fastapi import FastAPI ,Path ,HTTPException ,Query
from typing import Annotated,Literal,Optional
from pydantic import Field,computed_field , BaseModel,field_validator
import json
from fastapi.responses import JSONResponse
import pickle
import pandas as pd
from fastapi.responses import JSONResponse

