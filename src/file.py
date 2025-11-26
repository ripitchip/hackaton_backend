from fastapi import APIRouter, UploadFile

router = APIRouter(
    prefix="/users",
    tags=["items"],
)

@router.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}
