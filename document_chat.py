import tempfile

import streamlit as st
from modules.llm import OllamaModel
from modules.vectordb import QueryVectorDB

st.set_page_config(layout='wide')
st.title("Let's chat")


# Initialize message history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Upload Document Tab
uploaded_file = st.file_uploader("Choose a file", type="pdf")
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_filepath = temp_file.name
    
    init = QueryVectorDB(temp_filepath)
    init.create_vectordb()

    st.write("File uploaded successfully!")

model = OllamaModel(stream=True,
                    prompt_file_path="prompts/document_qa_prompt.txt")
if prompt := st.chat_input("How may I help you today?"):

    # Add user message to history
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()  # Placeholder for streaming response
        full_response = ""

        response = init.query(q=prompt)
        response_text = "\n\n".join([i['page_content'] for i in response.values()])
        model.system_prompt.format(question=prompt, context=response_text)

        # Stream response
        for chunk in model.answer(query=prompt+'\n\n'+response_text):
            full_response += chunk
            response_placeholder.markdown(full_response)

        # Add assistant response to history
        st.session_state["messages"].append({"role": "assistant", "content": full_response})