from fastapi import APIRouter, UploadFile, HTTPException
import pandas as pd
import sqlite3
from io import BytesIO
import os
import json
import tempfile

# Optional: load .env automatically
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

import google.generativeai as genai


router = APIRouter(
    prefix="/file",
    tags=["file"],
)

DB_PATH = "cleaned_data.db"


# ----------------------- XLSX HELPERS -----------------------

def validate_xlsx(file: UploadFile):
    if not file.filename.lower().endswith(".xlsx"):
        raise HTTPException(
            status_code=400,
            detail=f"File '{file.filename}' must be an .xlsx file"
        )


def read_excel_upload(upload_file: UploadFile) -> pd.DataFrame:
    contents = upload_file.file.read()
    return pd.read_excel(BytesIO(contents))


# ----------------------- XLSX ENDPOINT -----------------------

@router.post("/uploadfile/")
async def create_upload_file(
    file_mes: UploadFile,
    file_erp: UploadFile,
    file_plm: UploadFile
):
    for f in (file_mes, file_erp, file_plm):
        validate_xlsx(f)

    erp_df = read_excel_upload(file_erp)
    mes_df = read_excel_upload(file_mes)
    plm_df = read_excel_upload(file_plm)

    erp_df = erp_df.drop(columns=[
        "Âge", "Description du poste",
        "Niveau d'expérience", "Commentaire de Carrière"
    ], errors="ignore")

    mes_df = mes_df.drop(columns=[
        "Aléas Industriels", "Nombre pièces", "Cause Potentielle"
    ], errors="ignore")

    plm_df = plm_df.drop(columns=[
        "Criticité", "Masse (kg)", "Fournisseur"
    ], errors="ignore")

    conn = sqlite3.connect(DB_PATH)
    erp_df.to_sql("erp_cleaned", conn, if_exists="replace", index=False)
    mes_df.to_sql("mes_cleaned", conn, if_exists="replace", index=False)
    plm_df.to_sql("plm_cleaned", conn, if_exists="replace", index=False)
    conn.close()

    return {
        "status": "success",
        "message": "Excel files cleaned & saved in SQLite",
        "database": DB_PATH,
        "tables": ["erp_cleaned", "mes_cleaned", "plm_cleaned"]
    }


# ----------------------- GEMINI PDF TREE EXTRACTION -----------------------

MODEL_NAME = "gemini-3-pro-preview"


async def extract_construction_tree_from_pdf(pdf_file: UploadFile) -> dict:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key missing")

    genai.configure(api_key=GEMINI_API_KEY)

    # Save PDF to a temporary file
    pdf_bytes = await pdf_file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        # Upload using path (correct API)
        uploaded = genai.upload_file(
            path=tmp_path,
            display_name=pdf_file.filename
        )

        file_id = uploaded.name

        prompt = f"""
        You are an expert in construction documentation analysis.

        I uploaded a PDF referenced as: {file_id}.

        TASK:
        Extract a full construction dependency tree between all elements described in the PDF.

        Return STRICT JSON ONLY using this schema:

        {{
          "document": "{file_id}",
          "tree": [
            {{
              "id": "unique-id",
              "title": "Element Name",
              "description": "Short description",
              "dependencies": ["id-of-other-elements"],
              "children": []
            }}
          ]
        }}

        Rules:
        - NO text outside JSON
        - "dependencies" = prerequisites required before building this element
        - "children" = logical sub-elements
        """

        response = genai.GenerativeModel(MODEL_NAME).generate_content(
            [prompt, uploaded],
            generation_config={
                "response_mime_type": "application/json"
            }
        )

        return json.loads(response.text)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gemini error: {e}"
        )

    finally:
        # Delete temp file
        os.remove(tmp_path)


# ----------------------- PDF ENDPOINT -----------------------

@router.post("/pdf-tree/")
async def pdf_construction_tree(pdf: UploadFile):
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a PDF.")

    return await extract_construction_tree_from_pdf(pdf)

@router.post("/return-json/")
async def pdf_return_json()-> dict:
    with open("return.json", "r", encoding="utf-8") as f:
        return json.load(f)
