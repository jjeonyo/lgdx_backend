import asyncio
import os
from dotenv import load_dotenv
import traceback
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("google-genai not installed")
    exit(1)

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_ID = "gemini-2.5-flash-native-audio-preview-09-2025"

config = {
    "response_modalities": ["AUDIO"],
    "speech_config": {
        "voice_config": {
            "prebuilt_voice_config": {
                "voice_name": "Kore"
            }
        }
    }
}

async def main():
    client = genai.Client(api_key=API_KEY)
    print(f"Connecting to {MODEL_ID}...")
    try:
        async with client.aio.live.connect(model=MODEL_ID, config=config) as session:
            print("Connected. Speak now or wait for events...")
            # Just listen for a bit to capture any incoming message structure
            async for response in session.receive():
                print(f"Response type: {type(response)}")
                print(f"Response dir: {dir(response)}")
                if hasattr(response, 'server_content'):
                     print(f"Server content: {response.server_content}")
                break 
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

