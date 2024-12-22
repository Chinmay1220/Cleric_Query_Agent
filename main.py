import os
import json
import openai
from kubernetes import client, config
from pydantic import BaseModel, ValidationError
import logging
from datetime import datetime, timezone
from flask import Flask, request, jsonify


# Set up logs
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)s - %(message)s',
                    filename='agent.log', filemode='a')


# Initialize Flask app
app = Flask(__name__)

# Pydantic model for user query
class UserQuery(BaseModel):
    query: str
    answer: str

# Function for gathering Kubernetes data
def gather_kubernetes_data():
    try:    
        # Load from ~/.kube/config
        config.load_kube_config()

        # Initialize Kubernetes API clients
        core_api = client.CoreV1Api()
        apps_api = client.AppsV1Api()
        batch_api = client.BatchV1Api()
        networking_api = client.NetworkingV1Api()  # For Network Policies
        apiextensions_api = client.ApiextensionsV1Api()  # For CRDs

        cluster_info = {}

        # Cluster Info (API resources available)
        cluster_info["cluster_info"] = [
            {"name": resource.name, "kind": resource.kind} 
            for resource in core_api.get_api_resources().resources
        ]

        # Deployments
        deployments = apps_api.list_deployment_for_all_namespaces()
        cluster_info["deployments"] = [
            {
                "name": dep.metadata.name,
                "namespace": dep.metadata.namespace,
                "replicas": dep.spec.replicas,
                "type": [c.image.split(":")[0] for c in dep.spec.template.spec.containers],  # Container image name as type
                "port": [
                    p.container_port for c in dep.spec.template.spec.containers if c.ports for p in c.ports
                ],
                "age": str(datetime.now(timezone.utc) - dep.metadata.creation_timestamp).split(".")[0],  # Human-readable age
            }
            for dep in deployments.items
        ]
        
        # Pods
        pods = core_api.list_pod_for_all_namespaces()
        cluster_info["pods"] = [
            {
                "name": pod.metadata.name,
                "namespace": pod.metadata.namespace,
                "containers": [c.name for c in pod.spec.containers],
                "status": pod.status.phase,
                "age": str(
                    datetime.now(timezone.utc) - pod.metadata.creation_timestamp
                ).split(".")[0],  # Convert timedelta to a human-readable string
            }
            for pod in pods.items
        ]

        # Nodes
        nodes = core_api.list_node()
        cluster_info["nodes"] = [
            {
                "name": node.metadata.name,
                "status": node.status.conditions[-1].type,
                "addresses": [addr.address for addr in node.status.addresses],
                "roles": ",".join(
                    [label.split("/")[-1] for label in node.metadata.labels.keys() if "role" in label]
                ) or "None",  # Extract roles based on labels
                "age": str(datetime.now(timezone.utc) - node.metadata.creation_timestamp).split(".")[0],  # Node age
                "version": node.status.node_info.kubelet_version,  # Kubernetes version on the node
            }
            for node in nodes.items
        ]

        # ReplicaSets
        replicasets = apps_api.list_replica_set_for_all_namespaces()
        cluster_info["replicasets"] = [
            {
                "name": rs.metadata.name,
                "namespace": rs.metadata.namespace,
                "replicas": rs.spec.replicas,
                "available_replicas": rs.status.available_replicas or 0,  # Default to 0 if not set
                "ready_replicas": rs.status.ready_replicas or 0,  # Default to 0 if not set
                "age": str(datetime.now(timezone.utc) - rs.metadata.creation_timestamp).split(".")[0],  # Human-readable age
                "labels": rs.metadata.labels or {},  # Include labels
                "selector": rs.spec.selector.match_labels or {},  # Label selector
                "owner": rs.metadata.owner_references[0].name if rs.metadata.owner_references else "None",  # Owner (Deployment)
            }
            for rs in replicasets.items
        ]

        # Persistent Volumes (PVs)
        pv_api = client.CoreV1Api()
        pvs = pv_api.list_persistent_volume()
        cluster_info["pvs"] = [
            {
                "name": pv.metadata.name,
                "status": pv.status.phase,
                "capacity": pv.spec.capacity["storage"],  # Storage capacity (e.g., 10Gi)
                "storage_class": pv.spec.storage_class_name,  # Storage class name
                "reclaim_policy": pv.spec.persistent_volume_reclaim_policy,  # Reclaim policy
                "access_modes": pv.spec.access_modes,  # List of access modes
                "volume_mode": pv.spec.volume_mode or "Filesystem",  # Volume mode (default to Filesystem)
                "claim": pv.spec.claim_ref.name if pv.spec.claim_ref else "Unbound",  # Bound PVC name or Unbound
                "age": str(datetime.now(timezone.utc) - pv.metadata.creation_timestamp).split(".")[0],  # Human-readable age
            }   
            for pv in pvs.items
        ]

        # Secrets
        secrets = core_api.list_secret_for_all_namespaces()
        cluster_info["secrets"] = [
            {"name": secret.metadata.name, "namespace": secret.metadata.namespace}
            for secret in secrets.items
        ]

        #HPAs
        hpa_api = client.AutoscalingV1Api()
        hpas = hpa_api.list_horizontal_pod_autoscaler_for_all_namespaces()
        cluster_info["hpas"] = [
            {
                "name": hpa.metadata.name,
                "namespace": hpa.metadata.namespace,
                "min_replicas": hpa.spec.min_replicas,
                "max_replicas": hpa.spec.max_replicas,
            }
            for hpa in hpas.items
        ]

        #ConfigMaps
        configmaps = core_api.list_config_map_for_all_namespaces()
        cluster_info["configmaps"] = [
            {"name": cm.metadata.name, "namespace": cm.metadata.namespace}
            for cm in configmaps.items
        ]

        #CronJobs
        cronjobs = batch_api.list_cron_job_for_all_namespaces()
        cluster_info["cronjobs"] = [
            {
                "name": cronjob.metadata.name,
                "namespace": cronjob.metadata.namespace,
                "schedule": cronjob.spec.schedule,
            }
            for cronjob in cronjobs.items
        ]


        #Resource Quotas
        resource_quotas = core_api.list_resource_quota_for_all_namespaces()
        cluster_info["resource_quotas"] = [
            {
                "name": rq.metadata.name,
                "namespace": rq.metadata.namespace,
                "hard": rq.spec.hard,
            }
            for rq in resource_quotas.items
        ]

        #Events
        events_api = core_api.list_event_for_all_namespaces()
        cluster_info["events"] = [
            {
                "name": event.metadata.name,
                "namespace": event.metadata.namespace,
                "message": event.message,
                "reason": event.reason,
                "type": event.type,
            }
            for event in events_api.items
        ]

        #Network Policies
        network_policies = networking_api.list_network_policy_for_all_namespaces()
        cluster_info["network_policies"] = [
            {
                "name": np.metadata.name,
                "namespace": np.metadata.namespace,
            }
            for np in network_policies.items
        ]   

        #CRDs
        crds = apiextensions_api.list_custom_resource_definition()
        cluster_info["crds"] = [
            {"name": crd.metadata.name} for crd in crds.items
        ]

        #StatefulSets
        statefulsets = apps_api.list_stateful_set_for_all_namespaces()
        cluster_info["statefulsets"] = [
            {
                "name": ss.metadata.name,
                "namespace": ss.metadata.namespace,
                "replicas": ss.spec.replicas,
            }
            for ss in statefulsets.items
        ]

        #Ingresses
        ingresses = networking_api.list_ingress_for_all_namespaces()
        cluster_info["ingresses"] = [
            {
                "name": ingress.metadata.name,
                "namespace": ingress.metadata.namespace,
                "host": ingress.spec.rules[0].host if ingress.spec.rules else "N/A",
            }
            for ingress in ingresses.items
        ]

        #Kubeconfig Details (read config file content)
        kubeconfig_path = os.path.expanduser("~/.kube/config")
        with open(kubeconfig_path) as kube_config_file:
            cluster_info["kubeconfig"] = kube_config_file.read()

        #Current Context
        cluster_info["current_context"] = config.list_kube_config_contexts()[1]

        #All Contexts
        cluster_info["all_contexts"] = [
            context["name"] for context in config.list_kube_config_contexts()[0]
        ]

        #Log and print the gathered data
        #print("\n--- Gathered Kubernetes Cluster Information ---")
        #print(json.dumps(cluster_info, indent=2))
        #Return gathered data
        return cluster_info
    
    except Exception as e:
        raise Exception(f"Error gathering Kubernetes information: {e}")

#Function to send the data to GPT model

def query_llm(cluster_data, user_query):
    try:
        #Construct the prompt with both gathered data and the user query
        prompt = f"""
        You are an expert Kubernetes assistant. Given the following information about a Kubernetes cluster:

        {json.dumps(cluster_data, indent=2)}

        Answer the following query related to Kubernetes. Ensure your response is always a **single word**. 
        If the query is not relevant to Kubernetes or the provided information, respond with "Unknown".

        Query: {user_query}
        """

        openai.api_key = os.getenv('OPENAI_API_KEY')  #OpenAI key

        #Sending the prompt to gpt-3.5-turbo using the v1/chat/completions endpoint
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use the chat model
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=50,  #Setting a low token limit to encourage brevity
            temperature=0.5
        )

        #Extract and return the answer from the LLM response
        return response.choices[0].message["content"].strip()

    except Exception as e:
        raise Exception(f"Error querying the LLM: {e}")

#Flask endpoint to handle multiple user queries
@app.route("/query/", methods=["POST"])
def query_kubernetes():
    try:
        #Retrieve the list of queries from the request
        user_queries = request.json.get('queries')
        
        #Check if the queries exist
        if not user_queries:
            return jsonify({"error": "Queries are required"}), 400
        
        #Ensure that queries are in a list format
        if not isinstance(user_queries, list):
            return jsonify({"error": "Queries should be a list"}), 400
        
        #Gather Kubernetes data once
        cluster_data = gather_kubernetes_data()
        
        #Process each query and store the answers
        formatted_answers = []
        for user_query in user_queries:
            if isinstance(user_query, str):
                answer = query_llm(cluster_data, user_query)
                #Format the answer exactly as requested
                formatted_answer = f'Q: "{user_query}" A: "{answer}"'
                formatted_answers.append(formatted_answer)

                #Log the results
                logging.info(f"Processed Query: {user_query} -> {answer}")
            else:
                formatted_answer = f'Q: "{user_query}" A: "Invalid query format"'
                formatted_answers.append(formatted_answer)

                logging.warning(f"Invalid Query Format: {user_query}")

        #Print the results
        print("\nQueries and Answers:")
        for fa in formatted_answers:
            print(fa)

        #Logging the answers
        logging.info(f"All Agent Answers: {formatted_answers}")

        #Return the answers as a JSON response
        return jsonify({"answers": formatted_answers})

    except Exception as e:
        logging.error(f"Error handling request: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
