from google import genai

# ✅ Add your Gemini API key here
api_key = "AIzaSyDC1bLkLqB4jcFHwqIkbyLkRDbxVxh1rvE"

# Initialize the client with the API key
client = genai.Client(api_key=api_key)

# Generate content using the Gemini model
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how data encryption works in simple terms."
)

print(response.text)
