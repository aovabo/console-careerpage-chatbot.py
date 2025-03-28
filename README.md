# Console Careers Chatbot

This is a simple RAG demo for a career page chatbot, built to demonstrate the skills and qualifications of Alex for the IT Solutions Engineer role at Console. The project is inspired by the [Multinear](https://multinear.com) platform.

<img align="right" width="300" src="static/console-demo.png">

## Introduction

This project shows how simple it is to build a proof-of-concept using RAG for a career page chatbot answering user questions on job roles and application processes, with platforms like [LangChain](https://github.com/langchain-ai/langchain). 

The <span style="color: red">real challenge</span> is to ensure that this bot is **reliable** - always giving the right answer, not hallucinating, and knowing how to deal with ambiguous or off-topic questions. GenAI is a powerful technology, but it's also **unpredictable by design**, and the only way to make it reliable is to build comprehensive test coverage and guardrails. 

That's exactly what the Multinear platform is for. Multinear allows developers to define evaluations in a simple yet powerful way and iteratively develop their GenAI applications, ensuring reliability and security.

## Why Did Alexander Build This Application?

Alexander built this chatbot to demonstrate his skills and qualifications for the IT Solutions Engineer role. He has experience in building trainable agents, implementing RAG, fine-tuning models, and prompt engineering. Alex has a deep understanding of customer pain points from his extensive experience in various IT roles.

## Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/aovabo/console-careerpage-chatbot.py
    cd console-careerpage-chatbot.py
    ```

2. **Configure Environment Variables**

   Create a `.env` file in the root directory and add your OpenAI API key:

    ```bash
    echo "OPENAI_API_KEY=your-api-key-here" > .env
    ```

### Option 1: Using `uv` (Recommended)

   [`uv`](https://github.com/astral-sh/uv) is the fastest way to run the application with minimal setup.

```bash
# Setup Environment
uv sync

# Start the Application
uv run main.py
```

### Option 2: Using `pyenv`

   [`pyenv`](https://github.com/pyenv/pyenv) allows you to manage multiple Python versions and virtual environments.

```bash
# Setup Environment
pyenv install 3.9
pyenv virtualenv 3.9 onsole-careerpage-chatbot
pyenv local onsole-careerpage-chatbot
pip install -r requirements.txt

# Start the Application
python main.py
```

### Option 3: Using Python's built-in `venv`

```bash
# Setup Environment
python3 -m venv .venv
source .venv/bin/activate
# On Windows:
# .\.venv\Scripts\activate
pip install -r requirements.txt

# Start the Application
python3 main.py
```

Open http://127.0.0.1:8080 to see the application.

Try asking different questions to see how the bot handles them:

- Hi there!
- How do I reset my password?
- What's the current exchange rate?
- Where is the closest coffee shop?

## Tracing

Enable LLM tracing with [Arize Phoenix](https://phoenix.arize.com) in the `.env` file (see [.env.example](.env.example) and [tracing.py](tracing.py)).

---


## Architecture

   Key system components:

1. [RAG Engine](engine.py) for document ingestion, indexing, and query processing using the `LangChain` library and `OpenAI` model.
2. [API Server](api.py) with `FastAPI` endpoints for chat, reindexing, and session management.
3. [HTML](static/index.html) & [React JS](static/app.js) frontend.
4. [Dataset](data/console_careers.txt) for the RAG engine.
5. [Experiment Runner](.console/task_runner.py) entry point for `console` platform.
6. [Configuration](.console/config.yaml) for evaluation tasks.

## Experimentation Platform

   The platform is designed to facilitate the development and evaluation of GenAI applications through systematic experimentation.

### Running Experiments

1. **Define Tasks**

   Configure your evaluation tasks in `.console/config.yaml`. Each task represents a specific input scenario for the career page chat bot, and defines how to evaluate the output.

2. **Execute Experiments**

   Run `console` platform.

    ```bash
    # Using uv
    uv run console web_dev

    # Using pyenv / virtualenv
    consoletinear web_dev
    ```

   Open http://127.0.0.1:8000 and start experimenting.

## License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
    <i>Built by <a href="https://github.com/aovabo">Alexander</a>.</i>
</p>
