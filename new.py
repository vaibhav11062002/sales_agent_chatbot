# test_gemini_api.py

import os

# Set your API key
os.environ['GOOGLE_API_KEY'] = "AIzaSyBvGk-pDi2hqdq0CLSoKV2Sa8TH5IWShtE"

print("Testing Gemini API...")
print("="*60)

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    print("✅ Import successful")
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )
    
    print("✅ LLM initialized")
    
    # Test simple query
    response = llm.invoke("Say 'Hello, API is working!'")
    
    print("✅ Response received:")
    print(f"   {response.content}")
    print("="*60)
    print("✅ SUCCESS: Your Gemini API key is working!")
    
except ImportError as e:
    print("❌ Import Error:", e)
    print("   Install: pip install langchain-google-genai")

except Exception as e:
    print("❌ API Error:", e)
    print("   Check your API key or quota")
    
    # Check for specific errors
    error_msg = str(e).lower()
    if '429' in error_msg or 'quota' in error_msg:
        print("   → Rate limit exceeded. Wait 60 seconds.")
    elif 'api key' in error_msg or 'authentication' in error_msg:
        print("   → Invalid API key. Check your key.")
    elif 'timeout' in error_msg:
        print("   → Request timeout. Check internet connection.")
