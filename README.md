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
   git clone https://github.com/Chinmay1220/Query_Agent_Cleric.git
   cd Query_Agent_Cleric

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

 **User Query 1:** How many nodes are there in the cluster?

 **Agent Answer 1:** The cluster has 1 node.

 **User Query 2:** How many pods are currently running in the cluster?

 **Agent Answer 2:** The cluster currently has 16 pods running.

 **User Query 3:** What is the AGE of pod named nginx'?

 **Agent Answer 3:** The age of the pod named "nginx-676b6c5bbc-rt792" is "14:51:24".

 **User Query 4:** What type of redis-leader deployment is?

 **Agent Answer 4:** The type of the "redis-leader" deployment is "docker.io/redis".

 **User Query 5:** What is the status of all running pods in the cluster?

 **Agent Answer 5:** Here is the status of all running pods in the cluster:

1. Pod Name: frontend-6b5f4cf68c-6g5lt, Namespace: default, Status: Running
   
2. Pod Name: frontend-6b5f4cf68c-p84vz, Namespace: default, Status: Running
   
3. Pod Name: frontend-6b5f4cf68c-ws2r9, Namespace: default, Status: Running
   
4. Pod Name: httpd-deployment-bf5c4df59-gpvh9, Namespace: default, Status: Running
   
5. Pod Name: httpd-deployment-bf5c4df59-pqftr, Namespace:

**Conclusion:**
To wrap things up, this assignment really showcases how AI can interact with a Kubernetes cluster to answer queries. My approach involves gathering information from the cluster, using GPT-4 to process it, and delivering responses to the user. This is a good example of an AI agent in action, as it autonomously collects data, reasons through it, and takes action by answering queries.

While the reasoning part currently relies on OpenAI's GPT, which isn't an internal AI model, the agent still operates in a very intelligent way by handling the tasks independently. The agent essentially does everything needed to fulfill the user's requests without needing step-by-step guidance.

To take this a step further, adding the ability for the agent to learn and adapt based on user feedback would make it even smarter. But even without that, it already meets many of the key aspects of an AI agent.

In conclusion, this assignment highlights the potential of AI agents in real-world tasks, and I'm confident that with a bit more work, this agent could become even more autonomous and intelligent.


