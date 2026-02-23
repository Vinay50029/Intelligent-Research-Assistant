# How to Deploy Your Intelligent Research Assistant

The absolute easiest and best place to deploy a Streamlit app (like yours) for free is **Streamlit Community Cloud**. It is designed specifically for this!

Here is the step-by-step guide to get your project live on the internet so you can share the link with anyone (and put it on your resume!).

---

## Step 1: Push Your Code to GitHub
Streamlit Cloud works by reading your code directly from a public GitHub repository.

1. Go to [GitHub.com](https://github.com/) and create a **New Repository**.
   * Give it a name (e.g., `intelligent-research-assistant`).
   * Make sure it is set to **Public**.
   * Do NOT check "Add a README file" (keep it completely empty).
2. Open your VS Code terminal (make sure you are inside the `research-assistant` folder).
3. Run these exact commands, one by one:
```bash
git init
git add .
git commit -m "Initial commit for deployment"
git branch -M main

# REPLACE THIS URL WITH YOUR ACTUAL GITHUB REPO URL
git remote add origin https://github.com/YOUR_USERNAME/intelligent-research-assistant.git

git push -u origin main
```
*(Note: I already created a `.gitignore` file for you, which ensures your secret `.env` file and heavy `chroma_db` database are NOT uploaded. This is very important for security!)*

---

## Step 2: Deploy to Streamlit Cloud

1. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with your GitHub account.
2. Click the **"New app"** button.
3. Select **"Use existing repo"**.
4. Fill in the details:
   * **Repository:** Select your `intelligent-research-assistant` repository from the dropdown.
   * **Branch:** `main`
   * **Main file path:** Type in `app.py`
5. **CRITICAL STEP - Adding Secrets:**
   * Before you click deploy, click on **"Advanced settings..."** at the bottom.
   * A "Secrets" box will pop up. This is where we put the keys that we kept hidden in your local `.env` file.
   * Paste your keys in this exact format:
     ```toml
     GOOGLE_API_KEY = "your-actual-gemini-key-goes-here"
     OPENAI_API_KEY = "your-actual-openai-key-goes-here"
     ```
   * Click **Save**.
6. Finally, click **"Deploy!"**

---

## Step 3: Wait and Test

* Streamlit will now read your `requirements.txt` file and install the 18 packages we just organized. This can take about 2-5 minutes.
* You will see the progress in the bottom right corner (the "Manage app" tab).
* Once it finishes, your app will be live and you will have a public URL! 

*(Optional) Any time you make changes to your code on your laptop, simply `git push` them to GitHub, and Streamlit will automatically update your live website!*
