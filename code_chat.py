import streamlit as st
from modules.llm import OllamaModel

st.set_page_config(layout='wide')
st.title("Let's Code")

# Initialize message history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

model = OllamaModel(model_name='codellama:7b', stream=True)
if prompt := st.chat_input("How may I help you today?"):

    # Add user message to history
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()  # Placeholder for streaming response
        full_response = ""

        # Stream response
        for chunk in model(query=prompt):
            full_response += chunk
            response_placeholder.markdown(full_response)

        # Add assistant response to history
        st.session_state["messages"].append({"role": "assistant", "content": full_response})
