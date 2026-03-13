# LangChain AI Tools Repository

This repository contains a collection of Jupyter notebooks and Python scripts demonstrating LangChain fundamentals, AI agents, tools integration, and related ML experiments. It serves as a practical guide for building AI applications with LangChain, including introductions to chains, agents, and multi-tool setups.

Key components include:
- LangChain tutorials for beginners
- Agent-based AI implementations
- Tool integrations for enhanced functionality
- Additional ML scripts (e.g., bigram language model, Micrograd neural net from scratch)

This project explores AI agent development, chain orchestration, and foundational ML concepts for educational and prototyping purposes.

## Motivation
As a BSc Computer Science student passionate about AI, I created this repo to:
- Document hands-on learning with LangChain
- Explore agentic AI and tool integrations
- Bridge foundational ML (e.g., Micrograd) with modern frameworks
- Provide reusable notebooks for quantum/AI hybrids and general education

This work complements my broader interests in Retrieval-Augmented Generation (RAG), deep learning, and quantum computing.

## Repository Contents
- **1-langchainintro.ipynb**: Introduction to LangChain basics and chain creation.
- **2-tools.ipynb**: Implementing and integrating tools in LangChain workflows.
- **3-agentintro.ipynb**: Building simple AI agents with LangChain.
- **4-agentmultipletool.ipynb**: Advanced agents with multiple tool integrations.
- **bigram.py**: Simple bigram language model implementation.
- **micrograd.ipynb**: From-scratch neural network implementation (inspired by Andrej Karpathy).
- **runnable.py**: Script for runnable LangChain components.
- **sten.py**: Utility script for string encoding/processing.
- **stream.py**: Streaming output handler for AI responses.

## Tech Stack
- Python 3.10+
- Jupyter Notebooks
- LangChain
- Groq (for LLM inference in updated notebooks)
- Additional: NumPy, PyTorch (implied in ML scripts)

## Setup and Usage
1. Clone Repository
git clone https://github.com/asfandyar-prog/langchain-ai-tools.git
cd langchain-ai-tools
text2. Create Virtual Environment
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
text3. Install Dependencies
Create and install from a `requirements.txt` (example provided below; adjust as needed):
langchain
groq
numpy
torch
jupyter

pip install -r requirements.txt
text4. Add Environment Variables  
For Groq integration, create .env file:
GROQ_API_KEY=your_groq_api_key
text5. Run Notebooks
jupyter notebook
textOpen and run the .ipynb files in order (1–4 for LangChain progression).

## Example Use Cases
- Build a basic LangChain chain for text processing (1-langchainintro.ipynb).
- Create an AI agent that uses multiple tools for tasks like search/math (4-agentmultipletool.ipynb).
- Experiment with from-scratch NNs for understanding backprop (micrograd.ipynb).
- Train a simple language model on text data (bigram.py).

## Engineering Notes
- Updated Groq client initialization in notebooks for efficient LLM calls.
- Scripts are modular and self-contained for easy extension.
- Focus on educational clarity with comments and step-by-step implementations.

## Future Improvements
- Add more advanced agents (e.g., memory-enhanced or multi-modal).
- Integrate quantum simulations (Qiskit) for hybrid AI agents.
- Deploy as web apps (e.g., Streamlit).
- Add evaluation metrics for agent performance

## About the Author
Asfand Yar  
BSc Computer Science (Minor in Physics)  
Instructor – Introduction to Quantum Computing  
AI Systems & RAG Engineer 

Interests:  
- AI Engineering  
- Retrieval-Augmented Generation  
- ML Infrastructure  
- Quantum Computing  
- AI in Education  

## License
This project is for educational and research purposes. Licensed under the MIT License.
