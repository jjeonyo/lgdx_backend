import sys
import os
from pathlib import Path
import firebase_admin
from firebase_admin import storage, credentials, firestore

# Add current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# We'll use the correct key file now
PROJECT_ROOT = Path("/Users/harry/LG DX SCHOOL/lgdx_backend")
KEY_PATH = PROJECT_ROOT / "vision/FirebaseAdmin.json"

def init_firebase_custom():
    if not firebase_admin._apps:
        print(f"🔑 Initializing Firebase with: {KEY_PATH}")
        cred = credentials.Certificate(str(KEY_PATH))
        firebase_admin.initialize_app(cred)

def get_latest_session_id():
    init_firebase_custom()
    try:
        db = firestore.client()
        print("🔎 Searching for latest session in 'sessions' (from vision.py style) or 'chat_rooms'...")
        
        # Try 'sessions' first (vision.py uses this)
        sessions_ref = db.collection('sessions').order_by('start_time', direction=firestore.Query.DESCENDING).limit(1)
        sessions = list(sessions_ref.stream())
        
        if sessions:
            sid = sessions[0].id
            print(f"✅ Found latest session in 'sessions': {sid}")
            return sid, 'sessions'
            
        # Try 'chat_rooms' (generate.py uses this)
        # Note: generate.py looks for messages globally, but let's look for chat_rooms
        chat_rooms_ref = db.collection('chat_rooms').limit(1) # Just get any if no timestamp
        # Actually generate.py does complex query. Let's keep it simple.
        # If we can't find one, default to 'room_user_001'
        print("⚠️ No session in 'sessions'. Using default 'room_user_001'.")
        return 'room_user_001', 'chat_rooms'
        
    except Exception as e:
        print(f"❌ Error getting session: {e}")
        return None, None

def upload_video_custom(file_path, bucket_name):
    print(f"   Trying bucket: {bucket_name}")
    try:
        # Ensure app is initialized
        init_firebase_custom()
            
        bucket = storage.bucket(name=bucket_name)
        
        # Check if bucket exists/accessible by listing blobs
        blobs_iter = bucket.list_blobs(prefix="chat_rooms/", max_results=1)
        _ = list(blobs_iter) 
        
        # If we are here, bucket exists. Proceed with logic.
        blobs = list(bucket.list_blobs(prefix="chat_rooms/"))
        max_num = 0
        for b in blobs:
            name = b.name
            if name.startswith("chat_rooms/video_") and name.endswith(".mp4"):
                try:
                    num_part = name[17:-4]
                    num = int(num_part)
                    if num > max_num:
                        max_num = num
                except ValueError:
                    continue
        
        next_num = max_num + 1
        new_filename = f"video_{next_num:05d}.mp4"
        storage_path = f"chat_rooms/{new_filename}"
        
        print(f"   🔢 Filename: {storage_path}")
        blob = bucket.blob(storage_path)
        blob.metadata = {"contentType": "video/mp4"}
        blob.upload_from_filename(file_path)
        blob.make_public()
        
        return blob.public_url

    except Exception as e:
        print(f"   ❌ Failed with {bucket_name}: {e}")
        return None

def save_video_message(session_id, collection_type, video_url):
    try:
        db = firestore.client()
        if collection_type == 'sessions':
             # vision.py style: sessions/{id}/messages
             ref = db.collection('sessions').document(session_id).collection('messages')
             ref.add({
                'sender': 'gemini', # vision.py uses 'gemini' or 'ai'? vision.py uses 'gemini'
                'content': '솔루션 영상을 생성했습니다.',
                'video_url': video_url,
                'type': 'video', # Maybe custom type?
                'created_at': firestore.SERVER_TIMESTAMP
             })
        else:
            # generate.py style: chat_rooms/{id}/messages
            ref = db.collection('chat_rooms').document(session_id).collection('messages')
            ref.add({
                "sender": "ai",
                "text": "솔루션 영상을 생성했습니다.",
                "video_url": video_url,
                "message_type": "VIDEO",
                "created_at": firestore.SERVER_TIMESTAMP
            })
        print(f"💾 Firestore saved to {collection_type}/{session_id}")
    except Exception as e:
        print(f"❌ Firestore save failed: {e}")

def manual_upload_flow(file_path):
    print(f"🚀 Starting manual upload flow (using lgdx-6054d key) for: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    # 1. Get Session ID
    session_id, collection_type = get_latest_session_id()
    if not session_id:
        print("⚠️ Could not determine session.")

    # 2. Upload
    print("\nStep 2: Uploading to Firebase Storage...")
    
    # Buckets for lgdx-6054d
    candidate_buckets = [
        "lgdx-6054d.appspot.com",
        "lgdx-6054d.firebasestorage.app",
        "staging.lgdx-6054d.appspot.com"
    ]
    
    video_url = None
    for b in candidate_buckets:
        video_url = upload_video_custom(file_path, b)
        if video_url:
            print(f"✅ Upload successful! URL: {video_url}")
            break
            
    if not video_url:
        print("❌ All bucket attempts failed.")
        return

    # 3. Save to Firestore
    print("\nStep 3: Saving to Firestore...")
    if session_id:
        save_video_message(session_id, collection_type, video_url)
    else:
        print("⏭️ Skipped Firestore save.")

    print("\n🎉 Manual upload flow completed!")

if __name__ == "__main__":
    target_file = "/Users/harry/LG DX SCHOOL/lgdx_backend/generate/assets_generate/result_solution_20251204_192053.mp4"
    manual_upload_flow(target_file)
