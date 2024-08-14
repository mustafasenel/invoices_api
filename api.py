from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import cv2
import os
import pandas as pd
import numpy as np
from typing import List, Any
from pydantic import BaseModel
from pymongo import MongoClient
import dns.resolver
from pymongo.server_api import ServerApi
from datetime import datetime
import base64
from detect import detect_with_yolo
from tableocr import get_list
from deskew import deskew
from read_data import get_text


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

app = FastAPI()

dns.resolver.default_resolver=dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers=['8.8.8.8']

uri = "mongodb+srv://senel1806:dEWqI0FVFMDRpiFl@cluster0.f2kg7rn.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['invoices']
collection = db['invoices']


class ProcessedData(BaseModel):
    company_name: str
    invoice_number: str
    invoice_date: str
    table_data: List[Any]

class ImageData(BaseModel):
    image: str

class BatchProcessedData(BaseModel):
    results: List[ProcessedData]

class Invoice(BaseModel):
    code: str
    timestamp: str
    date: str
    isTransferred: bool
    images: List[Any]
    data: List[Any]

def extractData(input_img):
    result_img, table_img, company_img, number_img, date_img = detect_with_yolo(input_img)
    try:
        company_name = ""  # Şirket adı
        if company_img is not None and company_img.size > 0:
            company_name = get_text(company_img)
    
        invoice_number = ""  # Fatura numarası
        if number_img is not None and number_img.size > 0:
            invoice_number = get_text(number_img)
    
        invoice_date = ""  # Fatura tarihi
        if date_img is not None and date_img.size > 0:
            invoice_date = get_text(date_img)
    
        table_img = deskew(table_img)
        data = get_list(table_img)
        max_columns = max(len(row) for row in data)
        column_names = [f"Column{i+1}" for i in range(max_columns)]  # Column1, Column2, ... şeklinde sütun adları oluştur
        df = pd.DataFrame(data, columns=column_names)
    except:
        df = pd.DataFrame()
        company_name = ""
        invoice_number = ""
        invoice_date = ""
    return df, company_name, invoice_number, invoice_date


@app.post("/process-images/", response_model=BatchProcessedData)
async def process_images(image_data_list: List[ImageData]):
    if not image_data_list:
        raise HTTPException(status_code=400, detail="Image data list cannot be empty.")
    
    results = []
    for image_data in image_data_list:
        image_bytes = base64.b64decode(image_data.image)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        df, company_name, invoice_number, invoice_date = extractData(img)
        
        if df.empty:
            table_data = []  # Ensure table_data is an empty list if "not detected"
        else:
            table_data = df.to_dict(orient='records')
        
        response_data = ProcessedData(
            company_name=company_name,
            invoice_number=invoice_number,
            invoice_date=invoice_date,
            table_data=table_data
        )
        
        try:
            results.append(response_data.model_dump())  # Use model_dump instead of dict
        except ValidationError as e:
            # Log the validation error
            print(f"Validation Error: {e}")
            # Continue with the loop or add default/error information if needed
    
    return JSONResponse(content=results)

@app.post("/save-invoice")
async def save_invoice(invoice: Invoice, request: Request):
    try:
        invoice_data = await request.json()
        print("Received payload:", invoice_data)  # Log the payload
        invoice = Invoice(**invoice_data)  # Validate the payload using Pydantic
        collection.insert_one(invoice.model_dump())
        return {"message": "Invoice saved successfully"}
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))