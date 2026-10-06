import os
from dotenv import load_dotenv
from groq import Groq

def verify_groq_api():
    # Load environment variables from the .env file
    load_dotenv()
    
    # Retrieve the API key
    api_key = os.getenv("GROQ_API_KEY_2")
    
    if not api_key:
        print("❌ Error: GROQ_API_KEY not found in the .env file.")
        return

    print("🔄 Connecting to Groq...")
    
    try:
        # Initialize the Groq client
        client = Groq(api_key=api_key)
        
        # Send a minimal, lightweight prompt to test connection
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "user",
                    "content": "Respond with the word 'Success' if you can read this."
                }
            ],
            max_completion_tokens=5
        )
        
        # Print the response to confirm it works
        response_text = completion.choices[0].message.content.strip()
        print("✅ API Key is valid!")
        print(f"🤖 Groq Response: {response_text}")
        
    except Exception as e:
        print("❌ API Key verification failed.")
        print(f"📁 Error Details: {e}")

if __name__ == "__main__":
    verify_groq_api()
