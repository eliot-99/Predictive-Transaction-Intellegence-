from google import genai

# ✅ Add your Gemini API key here
api_key = "AIzaSyD1AnT79jL7JrP_ffzumjMocq1bogFdb_M"

# Initialize the client with the API key
client = genai.Client(api_key=api_key)

# Generate content using the Gemini model
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how data encryption works in simple terms."
)

print(response.text)
