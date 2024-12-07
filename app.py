import streamlit as st
from modules.llm import OllamaModel

st.set_page_config(layout='wide')


st.title("Let's chat")

# initialize history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


text_tab, upload_document_tab, code_tab = st.tab(["ChatBot", "Chat With Document", "Coding Assistant"])

if text_tab:
    model = OllamaModel()
    if prompt := st.chat_input("How may I help you today?"):

        # add latest message to history on format {role, content}
        st.session["messages"].append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            q = st.session_state["messages"][-1]["content"]


        st.session_state["messages"].append({"role": "assistant", "content": model(query=q)})

elif code_tab:
    model = OllamaModel(model_name='codellama:7b')

    if prompt := st.chat_input("How may I help you today?"):

        # add latest message to history on format {role, content}
        st.session["messages"].append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            q = st.session_state["messages"][-1]["content"]


        st.session_state["messages"].append({"role": "assistant", "content": model(query=q)})

else:
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        pass
