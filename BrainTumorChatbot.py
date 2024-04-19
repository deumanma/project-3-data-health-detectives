import streamlit as st
pip install openai==0.28
from pyngrok import ngrok
import openai
# Set your OpenAI API key
openai.api_key = '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'
def ask_chatgpt(question, model="text-davinci-002"):
    """Function to get responses from ChatGPT."""
    response = openai.Completion.create(
        model=model,
        prompt=question,
        max_tokens=250
    )
    return response.choices[0].text.strip()
def main():
    st.title('MRI Brain Tumor Analysis Helper')
    user_query = st.text_input("Hello, I'm your AI helper. Ask me anything about MRI scans for brain tumors:")
    if user_query:
        response = ask_chatgpt(user_query)
        st.write(response)
    st.write("Note: This tool provides information based on textual data and cannot analyze actual MRI images.")
if __name__ == '__main__':
    main()