# H-JEPA_PM_Co-Pilot
# Project Manager Co-Pilot Chatbot

This repository contains the code for a Project Manager Co-Pilot Chatbot using a Hierarchical Joint Embedding Predictive Architecture (H-JEPA) with the Ollama model. The chatbot is trained using self-supervised learning and fine-tuned with task-specific data.

Inspired by the "A Path Towards Autonomous Machine Intelligence" Paper. Link to paper:  https://openreview.net/forum?id=BZ5a1r-kVsf

## Installation

1. Clone the repository:

git clone https://github.com/your-username/project-manager-copilot-chatbot.git

2. Install the required dependencies:

pip install -r requirements.txt

## Usage

1. Prepare your training data and place it in the `data/` directory.

2. Train the model:

python src/train.py

3. Use the trained model for chatbot inference:
```python
from src.model import generate_response

context = "How can I effectively manage project timelines?"
response = generate_response(context)
print(response)
