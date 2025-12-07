import firebase_admin
from firebase_admin import credentials, storage
from pathlib import Path

key_path = Path("vision/FirebaseAdmin.json")
if not key_path.exists():
    print("Key file not found!")
    # Try absolute path
    key_path = Path("/Users/harry/LG DX SCHOOL/lgdx_backend/vision/FirebaseAdmin.json")

cred = credentials.Certificate(str(key_path))
firebase_admin.initialize_app(cred)

try:
    # Using google-cloud-storage client directly via firebase_admin credentials
    from google.cloud import storage as gcs
    
    # Re-use credentials from firebase-admin
    # Note: accessing the credentials object directly might vary, 
    # so let's just load them again for the gcs client
    gcs_client = gcs.Client.from_service_account_json(str(key_path))
    
    print(f"Project: {gcs_client.project}")
    print("Listing buckets:")
    buckets = list(gcs_client.list_buckets())
    if not buckets:
        print("No buckets found.")
    for bucket in buckets:
        print(f" - {bucket.name}")
        
except Exception as e:
    print(f"Error listing buckets: {e}")
