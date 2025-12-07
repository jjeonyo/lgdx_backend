import firebase_admin
from firebase_admin import credentials, storage
from pathlib import Path

key_path = Path("/Users/harry/LG DX SCHOOL/lgdx_backend/vision/FirebaseAdmin.json")

try:
    cred = credentials.Certificate(str(key_path))
    firebase_admin.initialize_app(cred)
    print("Auth Success!")
    
    candidates = [
        "lgdx-6054d.appspot.com",
        "lgdx-6054d.firebasestorage.app",
        "staging.lgdx-6054d.appspot.com"
    ]

    for name in candidates:
        try:
            print(f"Testing bucket: {name}")
            bucket = storage.bucket(name)
            # check access
            blobs = list(bucket.list_blobs(max_results=1))
            print(f"✅ SUCCESS with {name}!")
            break
        except Exception as e:
            print(f"❌ Failed {name}: {e}")
            
except Exception as e:
    print(f"Global Failed: {e}")
