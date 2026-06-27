HOW TO RUN THE AI RECRUITER PROJECT
===================================

Follow these step-by-step instructions to run the AI candidate ranking system on your Mac.
-> IN data Folder -> paste candidates.jsonl from Hackathon data. 

Copy and paste the following commands into your Terminal, one line at a time, and press Enter after each:

1. Go into the project folder:
   cd ~/Desktop/ai-recruiter

2. Create a Python Virtual Environment (to keep everything clean):
   python3 -m venv venv

3. Activate the virtual environment:
   source venv/bin/activate

4. Install all the required AI libraries (This might take a minute):
   pip install -r requirements.txt

5. Start the application:
   streamlit run app.py

-------------------------
After running the last command, a browser window will automatically open with your AI Recruiter Dashboard.

IMPORTANT: GEMINI API KEY REQUIRED
----------------------------------
Since this uses Gemini to act as the AI Recruiter (Stage 2 ranking), you will need a Gemini API Key. 
1. You can get one for free from: https://aistudio.google.com/app/apikey
2. Once you have it, just paste it into the left sidebar of the Streamlit Dashboard where it says "Gemini API Key".
