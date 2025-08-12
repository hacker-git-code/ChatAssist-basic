# # Import necessary libraries
# import nltk
# import spacy
# import transformers
# import requests  # For making HTTP requests to the Canva API
# from typing import List, Dict
# from textblob import TextBlob
# import os  # For accessing environment variables (API keys)
# from flask import Flask, render_template, request, jsonify, send_from_directory  # Flask imports
# from werkzeug.utils import secure_filename # Import secure_filename

# app = Flask(__name__)  # Create the Flask app

# # Configuration for image uploads
# UPLOAD_FOLDER = 'static/uploads'
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# # Create the uploads folder if it doesn't exist
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# class AIAgent:
#     def __init__(self, name="ContentMasterAI"):
#         self.name = name

#         # Attempt to load the SpaCy model, handle the case where it's not installed
#         try:
#             self.nlp = spacy.load("en_core_web_sm")  # For NLP tasks (e.g., POS tagging, NER)
#         except OSError as e:
#             print(f"Error loading SpaCy model: {e}")
#             print("Please download the model by running: python -m spacy download en_core_web_sm")
#             self.nlp = None  # Set to None so the code doesn't crash later

#         try:
#             self.chatbot_model = transformers.pipeline("conversational", model="microsoft/DialoGPT-medium")  # Example model (can be changed)
#         except Exception as e:
#             print(f"Error loading chatbot model: {e}")
#             print("Make sure you have PyTorch or TensorFlow installed.")
#             self.chatbot_model = None

#         self.canva_api_key = os.environ.get("CANVA_API_KEY")  # Retrieve API key from environment variable

#         if not self.canva_api_key:
#             print("Error: Canva API key not found in environment variables.")
#             # Handle the error gracefully (e.g., disable Canva functionality)

#         # Download required NLTK data (if not already downloaded)
#         try:
#             nltk.data.find('tokenizers/punkt')
#         except LookupError:
#             nltk.download('punkt')

#         #Initialize sentiment analysis
#         self.sentiment = TextBlob

#     def chatbot_respond(self, user_input: str, conversation_history: List[str] = None) -> str:
#         """Handles user input and generates a response."""
#         if self.chatbot_model is None:
#              print("Chatbot model not initialized. Returning a default response.")
#              return "I'm sorry, the chatbot is currently unavailable."

#         conversation = {}

#         if conversation_history:
#             conversation["past_user_inputs"] = conversation_history[::2]
#             conversation["generated_responses"] = conversation_history[1::2]

#         response = self.chatbot_model(user_input, conversation)
#         return response["generated_responses"][-1]


#     def canva_edit(self, image_path: str = None, text_overlay: str = None, template_id: str = None, adjustments: Dict = None) -> str:
#         """Edits a Canva design using the Canva API."""
#         if not self.canva_api_key:
#             print("Canva API key is missing.  Cannot edit Canva designs.")
#             return None

#         #Construct the API endpoint URL
#         api_url = "https://api.canva.com/v4/designs" # Example endpoint, check the documentation
#         headers = {
#             "Authorization": f"Bearer {self.canva_api_key}", #Authentication is required
#             "Content-Type": "application/json" #Required if you send json
#         }

#         data = {} #JSON payload, you'd need to consult Canva API documentation
#         if template_id:
#             data["templateId"] = template_id
#         if text_overlay:
#             data["text_overlay"] = text_overlay
#         if image_path:
#             data["image_path"] = image_path
#         if adjustments:
#             data["adjustments"] = adjustments

#         try:
#             response = requests.post(api_url, headers=headers, json=data)

#             response.raise_for_status() #Raises HTTPError for bad responses (4xx or 5xx)

#             #Process response
#             result = response.json()
#             print(f"Canva API response: {result}")
#             edited_image_url = result.get("preview_url") # example, parse from json response
#             print ("Editing done")

#             return edited_image_url # return the result from the API

#         except requests.exceptions.RequestException as e:
#             print(f"Canva API request failed: {e}")
#             return None


#     def post_ready(self, text: str = None, image_path: str = None, platform: str = "Twitter", goal: str = "engagement") -> Dict:
#         """Analyzes content and provides suggestions for posting."""

#         analysis = {}

#         if text:
#             analysis["text_length"] = len(text)
#             # Sentiment Analysis
#             analysis["text_sentiment"] = self.sentiment(text).sentiment.polarity
#             analysis["text_subjectivity"] = self.sentiment(text).sentiment.subjectivity

#             # Placeholder for more sophisticated text analysis (keywords, topic modeling)

#         if image_path:
#             # Placeholder for image analysis (using computer vision libraries)
#             analysis["image_description"] = "Placeholder image description"  # Use a CV library

#         # Platform-Specific Recommendations
#         if platform == "Twitter":
#             analysis["recommendations"] = {
#                 "hashtag_suggestions": ["#example", "#AI", "#content"],  # Placeholder
#                 "tone_adjustment": "Consider a more concise tone." if analysis["text_length"] > 280 else None,
#                 "caption" : f"Check out the {analysis['image_description']} #example #ai" if image_path else "Check out content #example #ai"
#             }
#         elif platform == "Instagram":
#             analysis["recommendations"] = {
#                 "hashtag_suggestions": ["#example", "#AI", "#content", "#instagood"],
#                 "caption" : f"Check out the amazing  {analysis['image_description']} #example #ai #instagood" if image_path else "Check out content #example #ai #instagood"
#             }
#         # Add other platform logic here

#         return analysis

# # Create the AI agent instance
# ai_agent = AIAgent()

# # Flask Routes

# @app.route("/")
# def index():
#     """Renders the main page with the form."""
#     return render_template("index.html")  # Create an index.html file in a 'templates' folder

# @app.route("/chat", methods=["POST"])
# def chat():
#     """Handles the chatbot interaction."""
#     user_input = request.form["user_input"]
#     response = ai_agent.chatbot_respond(user_input)
#     return jsonify({"response": response}) #Returning a json response

# @app.route("/edit", methods=["POST"])
# def edit():
#     """Handles the Canva edit request with image upload."""
#     # Check if an image was uploaded
#     if 'image' not in request.files:
#         return jsonify({"error": "No image part"})
#     file = request.files['image']

#     # If the user does not select a file, the browser submits an
#     # empty file without a filename.
#     if file.filename == '':
#         return jsonify({"error": "No selected image"})

#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(image_path) #Save in the upload directory

#         #Get other parameters
#         text_overlay = request.form.get("text_overlay")
#         template_id = request.form.get("template_id")
#         adjustments_str = request.form.get("adjustments")

#         # Attempt to convert adjustments string to dictionary. Add better error handling.
#         try:
#             adjustments = eval(adjustments_str) if adjustments_str else None
#             if not isinstance(adjustments, dict) and adjustments is not None:
#                 raise ValueError("Adjustments must be a dictionary-like string")

#         except (SyntaxError, NameError, ValueError) as e:
#             return jsonify({"error": f"Invalid adjustments format: {e}"})

#         #Now call the ai_agent.canva_edit
#         result = ai_agent.canva_edit(image_path=image_path, text_overlay=text_overlay, template_id=template_id, adjustments=adjustments)

#         if result:
#             return jsonify({"result": result, "image_url": '/' + image_path}) #Return the URL to the new image
#         else:
#             return jsonify({"error": "Canva edit failed."})

#     else:
#         return jsonify({"error": "Invalid file type. Allowed types: png, jpg, jpeg, gif"})


# @app.route("/uploads/<filename>")
# def download_file(filename):
#     return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# @app.route("/post", methods=["POST"])
# def post():
#     """Handles the post-ready analysis request."""
#     text = request.form.get("text")
#     image_path = request.form.get("image_path")
#     platform = request.form.get("platform", "Twitter")  # Default to Twitter
#     goal = request.form.get("goal", "engagement")

#     analysis = ai_agent.post_ready(text=text, image_path=image_path, platform=platform, goal=goal)
#     return jsonify(analysis)

# # Run the Flask app
# if __name__ == "__main__":
#     app.run(debug=True)  # Enable debug mode for development







# # Import necessary libraries
# import nltk
# import spacy
# import transformers
# from typing import List
# from flask import Flask, render_template, request, jsonify

# app = Flask(__name__)

# class AIAgent:
#     def __init__(self, name="ContentMasterAI"):
#         self.name = name
#         self.chatbot_model = None # Initialize to None for error handling

#         try:
#             self.chatbot_model = transformers.pipeline("conversational", model="microsoft/DialoGPT-medium")
#             print("Chatbot model loaded successfully!")  # Debug message
#         except Exception as e:
#             print(f"Error loading chatbot model: {e}")
#             print("Make sure you have PyTorch or TensorFlow installed correctly.")
#             # It's crucial to set the model to None if there's an error
#             self.chatbot_model = None


#     def chatbot_respond(self, user_input: str, conversation_history: List[str] = None) -> str:
#         """Handles user input and generates a response."""

#         if self.chatbot_model is None:
#             print("Chatbot model is not initialized! Returning a default response.")
#             return "I'm sorry, the chatbot is currently unavailable."

#         conversation = {}

#         if conversation_history:
#             conversation["past_user_inputs"] = conversation_history[::2]
#             conversation["generated_responses"] = conversation_history[1::2]

#         try:
#             response = self.chatbot_model(user_input, conversation)
#             return response["generated_responses"][-1]
#         except Exception as e:
#             print(f"Error generating chatbot response: {e}")
#             return "An error occurred while generating a response."


# ai_agent = AIAgent()  # Create the agent instance

# @app.route("/")
# def index():
#     """Renders the main page with the form."""
#     return render_template("index.html")

# @app.route("/chat", methods=["POST"])
# def chat():
#     """Handles the chatbot interaction."""
#     user_input = request.form["user_input"]
#     response = ai_agent.chatbot_respond(user_input)
#     return jsonify({"response": response})

# if __name__ == "__main__":
#     app.run(debug=True)















# # Import necessary libraries
# import nltk
# import spacy
# import transformers
# from typing import List
# from flask import Flask, render_template, request, jsonify

# app = Flask(__name__)

# class AIAgent:
#     def __init__(self, name="ContentMasterAI"):
#         self.name = name
#         self.chatbot_model = None # Initialize to None for error handling

#         try:
#             # Use a text generation pipeline instead of "conversational"
#             self.chatbot_model = transformers.pipeline("text-generation", model="gpt2")
#             print("Chatbot model (text-generation) loaded successfully!")  # Debug message
#         except Exception as e:
#             print(f"Error loading chatbot model: {e}")
#             print("Make sure you have PyTorch or TensorFlow installed correctly and the model is available.")
#             # It's crucial to set the model to None if there's an error
#             self.chatbot_model = None


#     def chatbot_respond(self, user_input: str, conversation_history: List[str] = None) -> str:
#         """Handles user input and generates a response."""

#         if self.chatbot_model is None:
#             print("Chatbot model is not initialized! Returning a default response.")
#             return "I'm sorry, the chatbot is currently unavailable."

#         # Adjust the prompt for text generation
#         prompt = f"User: {user_input}\nAI:"  # A simple prompt format
#         try:
#             response = self.chatbot_model(prompt, max_length=150, num_return_sequences=1) # Setting a limit on the length of generated text

#             # Extract generated text from the response (different format than conversational)
#             generated_text = response[0]['generated_text']

#             # Clean up the generated text
#             cleaned_response = generated_text.replace(prompt, "").strip()
#             return cleaned_response

#         except Exception as e:
#             print(f"Error generating chatbot response: {e}")
#             return "An error occurred while generating a response."


# ai_agent = AIAgent()  # Create the agent instance

# @app.route("/")
# def index():
#     """Renders the main page with the form."""
#     return render_template("index.html")

# @app.route("/chat", methods=["POST"])
# def chat():
#     """Handles the chatbot interaction."""
#     user_input = request.form["user_input"]
#     response = ai_agent.chatbot_respond(user_input)
#     return jsonify({"response": response})

# if __name__ == "__main__":
#     app.run(debug=True)





















# # Import necessary libraries
# import nltk
# import spacy
# import transformers
# from typing import List
# from flask import Flask, render_template, request, jsonify
# import google.generativeai as genai  # Import the Gemini API library
# import os  # Import the os module

# app = Flask(__name__)

# # Configure Gemini API
# GOOGLE_API_KEY = os.environ.get("AIzaSyBRf5Mb8deEd-RNlgDmjgrUibIJoOrJmvo") # Retrieve API key from environment variable
# if not GOOGLE_API_KEY:
#     raise ValueError("No Google API key found.  Please set the GOOGLE_API_KEY environment variable.")
# genai.configure(api_key=GOOGLE_API_KEY)

# # Load the Gemini Pro model
# try:
#     model = genai.GenerativeModel('gemini-pro')
#     print("Gemini Pro model loaded successfully!")
# except Exception as e:
#     print(f"Error loading Gemini Pro model: {e}")
#     model = None # Handle the case where the model fails to load.



# class AIAgent:
#     def __init__(self, name="ContentMasterAI"):
#         self.name = name
#         self.gemini_model = model # Use the loaded Gemini model (if available)
#         self.chatbot_model = None # Initialize to None for error handling


#         # Optionally keep GPT-2 as a fallback:
#         try:
#              #Use a text generation pipeline instead of "conversational"
#              self.chatbot_model = transformers.pipeline("text-generation", model="gpt2")
#              print("Chatbot model (GPT-2 - text-generation) loaded successfully!")  # Debug message
#         except Exception as e:
#              print(f"Error loading GPT-2 chatbot model: {e}")
#              print("Make sure you have PyTorch or TensorFlow installed correctly and the model is available.")
#             # It's crucial to set the model to None if there's an error
#              self.chatbot_model = None



#     def chatbot_respond(self, user_input: str, conversation_history: List[str] = None) -> str:
#         """Handles user input and generates a response, prioritizing Gemini if available."""

#         if self.gemini_model:  # Use Gemini if it's loaded successfully
#             try:
#                 response = self.gemini_model.generate_content(user_input)
#                 return response.text # Access the generated text from the Gemini response
#             except Exception as e:
#                 print(f"Error generating response with Gemini: {e}")
#                 print("Falling back to GPT-2 (if available).")
#                 # Fallback to GPT-2 only if Gemini fails

#         if self.chatbot_model is None:
#             print("Both Gemini and GPT-2 chatbot models are unavailable! Returning a default response.")
#             return "I'm sorry, the chatbot is currently unavailable."


#         # If Gemini failed or wasn't available, use GPT-2
#         # Adjust the prompt for text generation
#         prompt = f"User: {user_input}\nAI:"  # A simple prompt format
#         try:
#             response = self.chatbot_model(prompt, max_length=150, num_return_sequences=1) # Setting a limit on the length of generated text

#             # Extract generated text from the response (different format than conversational)
#             generated_text = response[0]['generated_text']

#             # Clean up the generated text
#             cleaned_response = generated_text.replace(prompt, "").strip()
#             return cleaned_response

#         except Exception as e:
#             print(f"Error generating GPT-2 chatbot response: {e}")
#             return "An error occurred while generating a response."



# ai_agent = AIAgent()  # Create the agent instance

# @app.route("/")
# def index():
#     """Renders the main page with the form."""
#     return render_template("index.html")

# @app.route("/chat", methods=["POST"])
# def chat():
#     """Handles the chatbot interaction."""
#     user_input = request.form["user_input"]
#     response = ai_agent.chatbot_respond(user_input)
#     return jsonify({"response": response})

# if __name__ == "__main__":
#     app.run(debug=True)




























# # Import necessary libraries
# import nltk
# import spacy
# import transformers
# from typing import List
# from flask import Flask, render_template, request, jsonify
# import google.generativeai as genai  # Import the Gemini API library
# import os  # Import the os module

# app = Flask(__name__)

# # Configure Gemini API
# GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") # Retrieve API key from environment variable
# if not GOOGLE_API_KEY:
#     GOOGLE_API_KEY = "AIzaSyBRf5Mb8deEd-RNlgDmjgrUibIJoOrJmvo"  # Hardcoding the API key as a fallback
#     print("Warning: No GOOGLE_API_KEY environment variable set. Using hardcoded key. This is NOT recommended for production.")
#     #raise ValueError("No Google API key found.  Please set the GOOGLE_API_KEY environment variable.") #Remove the raise if you would like it to use hardcoded key

# genai.configure(api_key=GOOGLE_API_KEY)

# # Load the Gemini Pro model
# try:
#     model = genai.GenerativeModel('gemini-pro')
#     print("Gemini Pro model loaded successfully!")
# except Exception as e:
#     print(f"Error loading Gemini Pro model: {e}")
#     model = None # Handle the case where the model fails to load.



# class AIAgent:
#     def __init__(self, name="ContentMasterAI"):
#         self.name = name
#         self.gemini_model = model # Use the loaded Gemini model (if available)
#         self.chatbot_model = None # Initialize to None for error handling


#         # Optionally keep GPT-2 as a fallback:
#         try:
#              #Use a text generation pipeline instead of "conversational"
#              self.chatbot_model = transformers.pipeline("text-generation", model="gpt2")
#              print("Chatbot model (GPT-2 - text-generation) loaded successfully!")  # Debug message
#         except Exception as e:
#              print(f"Error loading GPT-2 chatbot model: {e}")
#              print("Make sure you have PyTorch or TensorFlow installed correctly and the model is available.")
#             # It's crucial to set the model to None if there's an error
#              self.chatbot_model = None



#     def chatbot_respond(self, user_input: str, conversation_history: List[str] = None) -> str:
#         """Handles user input and generates a response, prioritizing Gemini if available."""

#         if self.gemini_model:  # Use Gemini if it's loaded successfully
#             try:
#                 response = self.gemini_model.generate_content(user_input)
#                 return response.text # Access the generated text from the Gemini response
#             except Exception as e:
#                 print(f"Error generating response with Gemini: {e}")
#                 print("Falling back to GPT-2 (if available).")
#                 # Fallback to GPT-2 only if Gemini fails

#         if self.chatbot_model is None:
#             print("Both Gemini and GPT-2 chatbot models are unavailable! Returning a default response.")
#             return "I'm sorry, the chatbot is currently unavailable."


#         # If Gemini failed or wasn't available, use GPT-2
#         # Adjust the prompt for text generation
#         prompt = f"User: {user_input}\nAI:"  # A simple prompt format
#         try:
#             response = self.chatbot_model(prompt, max_length=150, num_return_sequences=1) # Setting a limit on the length of generated text

#             # Extract generated text from the response (different format than conversational)
#             generated_text = response[0]['generated_text']

#             # Clean up the generated text
#             cleaned_response = generated_text.replace(prompt, "").strip()
#             return cleaned_response

#         except Exception as e:
#             print(f"Error generating GPT-2 chatbot response: {e}")
#             return "An error occurred while generating a response."



# ai_agent = AIAgent()  # Create the agent instance

# @app.route("/")
# def index():
#     """Renders the main page with the form."""
#     return render_template("index.html")

# @app.route("/chat", methods=["POST"])
# def chat():
#     """Handles the chatbot interaction."""
#     user_input = request.form["user_input"]
#     response = ai_agent.chatbot_respond(user_input)
#     return jsonify({"response": response})

# if __name__ == "__main__":
#     app.run(debug=True)












# # Import necessary libraries
# import nltk
# import spacy
# import transformers
# from typing import List
# from flask import Flask, render_template, request, jsonify
# import google.generativeai as genai  # Import the Gemini API library
# import os  # Import the os module
# from dotenv import load_dotenv

# app = Flask(__name__)

# # Load environment variables from .env file
# load_dotenv()

# # Configure Gemini API
# GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") # Retrieve API key from environment variable
# if not GOOGLE_API_KEY:
#     print("Warning: No GOOGLE_API_KEY environment variable set. Please set it for secure operation.") #added period at the end
#     GOOGLE_API_KEY = "AIzaSyBRf5Mb8deEd-RNlgDmjgrUibIJoOrJmvo"  # Hardcoding the API key as a fallback
#     print("Warning: No GOOGLE_API_KEY environment variable set. Using hardcoded key. This is NOT recommended for production.")
#     #raise ValueError("No Google API key found.  Please set the GOOGLE_API_KEY environment variable.") #Remove the raise if you would like it to use hardcoded key

# genai.configure(api_key=GOOGLE_API_KEY)

# # Load the Gemini Pro model
# try:
#     #List available models to help debug model name issues
#     try: #added inner try-except for model listing
#         for m in genai.list_models():
#             print(f"Available model: {m.name}")
#             if 'generateContent' in m.supported_generation_methods:
#                 print("  Supports generateContent")
#     except Exception as list_error:
#         print(f"Error listing models: {list_error}")
#         raise #Re-raise to prevent the rest of the block from executing

#     model = genai.GenerativeModel('gemini-pro')  # Or try 'gemini-1.0-pro' depending on your region and access
#     print("Gemini Pro model loaded successfully!")

# except Exception as e:
#     print(f"Error loading Gemini Pro model: {e}")
#     model = None # Handle the case where the model fails to load.



# class AIAgent:
#     def __init__(self, name="ContentMasterAI"):
#         self.name = name
#         self.gemini_model = model # Use the loaded Gemini model (if available)
#         self.chatbot_model = None # Initialize to None for error handling


#         # Optionally keep GPT-2 as a fallback:
#         try:
#              #Use a text generation pipeline instead of "conversational"
#              self.chatbot_model = transformers.pipeline("text-generation", model="gpt2")
#              print("Chatbot model (GPT-2 - text-generation) loaded successfully!")  # Debug message
#         except Exception as e:
#              print(f"Error loading GPT-2 chatbot model: {e}")
#              print("Make sure you have PyTorch or TensorFlow installed correctly and the model is available.")
#             # It's crucial to set the model to None if there's an error
#              self.chatbot_model = None



#     def chatbot_respond(self, user_input: str, conversation_history: List[str] = None) -> str:
#         """Handles user input and generates a response, prioritizing Gemini if available."""

#         if self.gemini_model:  # Use Gemini if it's loaded successfully
#             try:
#                 response = self.gemini_model.generate_content(user_input)
#                 return response.text # Access the generated text from the Gemini response
#             except Exception as e:
#                 print(f"Error generating response with Gemini: {e}")
#                 print("Falling back to GPT-2 (if available).")
#                 # Fallback to GPT-2 only if Gemini fails

#         if self.chatbot_model is None:
#             print("Both Gemini and GPT-2 chatbot models are unavailable! Returning a default response.")
#             return "I'm sorry, the chatbot is currently unavailable."


#         # If Gemini failed or wasn't available, use GPT-2
#         # Adjust the prompt for text generation
#         prompt = f"User: {user_input}\nAI:"  # A simple prompt format
#         try:
#             response = self.chatbot_model(prompt, max_length=150, num_return_sequences=1) # Setting a limit on the length of generated text
#             response = self.chatbot_model(prompt, max_length=150, num_return_sequences=1, truncation=True)  # Added truncation=True

#             # Extract generated text from the response (different format than conversational)
#             generated_text = response[0]['generated_text']

#             # Clean up the generated text
#             cleaned_response = generated_text.replace(prompt, "").strip()
#             return cleaned_response

#         except Exception as e:
#             print(f"Error generating GPT-2 chatbot response: {e}")
#             return "An error occurred while generating a response."



# ai_agent = AIAgent()  # Create the agent instance

# @app.route("/")
# def index():
#     """Renders the main page with the form."""
#     return render_template("index.html")

# @app.route("/chat", methods=["POST"])
# def chat():
#     """Handles the chatbot interaction."""
#     user_input = request.form["user_input"]
#     response = ai_agent.chatbot_respond(user_input)
#     return jsonify({"response": response})

# if __name__ == "__main__":
#     app.run(debug=True)







# Import necessary libraries
import nltk
import spacy
import transformers
from typing import List
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai  # Import the Gemini API library
import os  # Import the os module
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") # Retrieve API key from environment variable
if not GOOGLE_API_KEY:
    print("Warning: No GOOGLE_API_KEY environment variable set. Please set it for secure operation.") #added period at the end
    GOOGLE_API_KEY = "AIzaSyBRf5Mb8deEd-RNlgDmjgrUibIJoOrJmvo"  # Hardcoding the API key as a fallback
    print("Warning: No GOOGLE_API_KEY environment variable set. Using hardcoded key. This is NOT recommended for production.")
    #raise ValueError("No Google API key found.  Please set the GOOGLE_API_KEY environment variable.") #Remove the raise if you would like it to use hardcoded key

genai.configure(api_key=GOOGLE_API_KEY)

# Load the Gemini Pro model
try:
    model = genai.GenerativeModel('gemini-pro')
    print("Gemini Pro model loaded successfully!")
except Exception as e:
    print(f"Error loading Gemini Pro model: {e}")
    model = None # Handle the case where the model fails to load.



class AIAgent:
    def __init__(self, name="ContentMasterAI"):
        self.name = name
        self.gemini_model = model # Use the loaded Gemini model (if available)
        self.chatbot_model = None # Initialize to None for error handling


        # Optionally keep GPT-2 as a fallback:
        try:
             #Use a text generation pipeline instead of "conversational"
             self.chatbot_model = transformers.pipeline("text-generation", model="gpt2")
             print("Chatbot model (GPT-2 - text-generation) loaded successfully!")  # Debug message
        except Exception as e:
             print(f"Error loading GPT-2 chatbot model: {e}")
             print("Make sure you have PyTorch or TensorFlow installed correctly and the model is available.")
            # It's crucial to set the model to None if there's an error
             self.chatbot_model = None



    def chatbot_respond(self, user_input: str, conversation_history: List[str] = None) -> str:
        """Handles user input and generates a response, prioritizing Gemini if available."""

        if self.gemini_model:  # Use Gemini if it's loaded successfully
            try:
                response = self.gemini_model.generate_content(user_input)
                return response.text # Access the generated text from the Gemini response
            except Exception as e:
                print(f"Error generating response with Gemini: {e}")
                print("Falling back to GPT-2 (if available).")
                return "An error occurred while generating a response." # Prevents crashing on the webpage

        if self.chatbot_model is None:
            print("Both Gemini and GPT-2 chatbot models are unavailable! Returning a default response.")
            return "I'm sorry, the chatbot is currently unavailable."


        # If Gemini failed or wasn't available, use GPT-2
        # Adjust the prompt for text generation
        prompt = f"User: {user_input}\nAI:"  # A simple prompt format
        try:
            response = self.chatbot_model(prompt, max_length=150, num_return_sequences=1) # Setting a limit on the length of generated text
            response = self.chatbot_model(prompt, max_length=150, num_return_sequences=1, truncation=True)  # Added truncation=True

            # Extract generated text from the response (different format than conversational)
            generated_text = response[0]['generated_text']

            # Clean up the generated text
            cleaned_response = generated_text.replace(prompt, "").strip()
            return cleaned_response

        except Exception as e:
            print(f"Error generating GPT-2 chatbot response: {e}")
            return "An error occurred while generating a response."



ai_agent = AIAgent()  # Create the agent instance

@app.route("/")
def index():
    """Renders the main page with the form."""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Handles the chatbot interaction."""
    user_input = request.form["user_input"]
    response = ai_agent.chatbot_respond(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)