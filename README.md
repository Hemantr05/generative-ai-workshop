# generative-ai-workshop

This repository uses open source packages to create chatbots for general question answering, document question answering and code assistance.

# Setting up

1. Virtual Environment:

    1.1. Create a virtual environment:

    ```
    $ python3 -m venv venv
    ```

    1.2. Activate environment
    ```
    $ source venv/bin/activate
    ```


2. Python package installation

    2.1. Install poetry package manager

    ```
    $ pip install poetry
    ```

    2.2. Install python packages

    ```
    $ poetry install
    ```

    2.3. Install watchdog (for streamlit)

    ```
    $ pip install watchdog
    ```

3. Install Ollama and download models

    3.1. Install Ollama:<br>

    <ul>3.1.1. Open https://ollama.com/download in any browser.</ul>
    <ul>3.1.2. Select OS of the current system being used.</ul>
    <ul>3.1.3. Navigate to the directory of download</ul>
    <ul>3.1.4. Unzip Ollama-*.zip (depends on OS)</ul>
    <ul>3.1.5. Open Ollama by double-clicking on the application </ul>
    <ul>3.1.6. Follow the instruction to further set it up</ul>

    3.2. Download Ollama models:

    <ul> 3.2.1. Download model for text understanding and generation:</ul>

        $ ollama pull phi

    <ul> 3.2.2. Download model for code generation: </ul>

        $ ollama pull codellama:7b

# Usage

Run streamlit application:

```
$ streamlit run general_chat.py
```