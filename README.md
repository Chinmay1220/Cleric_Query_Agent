# Cleric_Query_Agent

This project implements an AI agent capable of interacting with a Kubernetes cluster to answer queries about deployed applications. The agent uses GPT-4 for natural language processing and provides a Flask API for query submission.

**Approach for the AI Agent:**
The agent collects information about the Kubernetes cluster, such as configuration details, deployments, pods, etc. This is the phase where the agent gathers data about its environment.

**Reasoning and Decision-Making:**
Once the cluster information is retrieved, you send the data to an LLM (such as OpenAI's LLM) along with the user query. The LLM processes this information, infers answers, and generates a response based on its reasoning. This phase reflects the decision-making or reasoning process of your agent.

**Action:**
The action the AI agent takes is to return the generated response to the user. This is similar to an AI agent taking action in its environment based on its reasoning.

**Goal-Oriented Behavior:**
The AI agent's goal is to assist users by answering queries about the Kubernetes cluster. This aligns with the goal-oriented behavior of AI agents, as it is working towards fulfilling user requests.

---

## Features

- Uses Kubernetes API to gather information about cluster resources.
- Processes natural language queries via GPT-4.
- Provides a REST API for submitting and retrieving query results.
- Logs all activity to `agent.log` for debugging.

---

## Requirements

- Python 3.10 or later
- Kubernetes cluster (configured via `~/.kube/config`)
- OpenAI API key for GPT-4
- Dependencies specified in `requirements.txt`

---

## Steps to Set Up and Run the Cleric Query Agent


1. **Clone the Repository**
   ```bash
   git clone https://github.com/Chinmay1220/Cleric_Query_Agent.git
   cd Cleric_Query_Agent

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt

1. **Set the API Key**
   ```bash
   export OPENAI_API_KEY="your_api_key"

1. **Run the Application**
   ```bash
   python main.py

**Testing use case for example :**

Processed Queries and Answers:  
Q: "How many pods are running?" A: "10"  
Q: "Name one pod which is running?" A: "frontend"  
Q: "How many replicas are set for the 'frontend deployment'?" A: "3"  
Q: "What is the AGE of pod named nginx'?" A: ""2 days""  
Q: "What type of redis-leader deployment is?" A: "Stateful"  
Q: "What is the status of all running pods in the cluster?" A: "Running"  
Q: "How many pods are deployed across?" A: "11"  
Q: "Are there any recent updates or changes to secrets or ConfigMaps?" A: "Yes"  
Q: "What is the status of the pod named 'redis-leader'?" A: "Running"  
Q: "Which pod is spawned by frontend deployment?" A: "php-redis"  

**Conclusion:**
To wrap things up, this assignment really showcases how AI can interact with a Kubernetes cluster to answer queries. My approach involves gathering information from the cluster, using GPT-4 to process it, and delivering responses to the user. This is a good example of an AI agent in action, as it autonomously collects data, reasons through it, and takes action by answering queries.

While the reasoning part currently relies on OpenAI's GPT, which isn't an internal AI model, the agent still operates in a very intelligent way by handling the tasks independently. The agent essentially does everything needed to fulfill the user's requests without needing step-by-step guidance.

To take this a step further, adding the ability for the agent to learn and adapt based on user feedback would make it even smarter. But even without that, it already meets many of the key aspects of an AI agent.

Submission ID: edb01ee7-6d57-4d00-8c36-29355bae209e

In conclusion, this assignment highlights the potential of AI agents in real-world tasks, and I'm confident that with a bit more work, this agent could become even more autonomous and intelligent.


