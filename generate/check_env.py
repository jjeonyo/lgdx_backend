from dotenv import load_dotenv
import os
from pathlib import Path

project_root = Path("/Users/harry/LG DX SCHOOL/lgdx_backend")
load_dotenv(project_root / ".env")
bucket = os.getenv('FIREBASE_STORAGE_BUCKET')
print(f"BUCKET_FROM_ENV: '{bucket}'")
