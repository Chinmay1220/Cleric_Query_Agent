import os
import json
import openai
from kubernetes import client, config
from pydantic import BaseModel, ValidationError
import logging
from datetime import datetime, timezone
from flask import Flask, request, jsonify
import re

# Set up logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s - %(message)s',
    filename='agent.log',
    filemode='a'
)

# Initialize Flask app
app = Flask(__name__)

#Function to Remove K8s-Style Pod Suffix
def strip_k8s_suffix(pod_name: str) -> str:
    """
    Removes the last two dash-separated segments if they look like alphanumeric
    hashes (e.g., 'frontend-6b5f4cf68c-6g5lt' -> 'frontend').
    Otherwise returns the pod_name unchanged.
    """
    pattern = r'(.*)-[a-z0-9]+-[a-z0-9]+$'
    match = re.match(pattern, pod_name)
    if match:
        return match.group(1)
    return pod_name

#Pydantic Models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    answer: str

def gather_kubernetes_data():
    """
    Load Kubernetes config and retrieve cluster information:
    deployments, pods, nodes, replicasets, etc.
    Returns a dictionary 'cluster_info' with all relevant data.
    """
    try:
        # Load from ~/.kube/config
        config.load_kube_config()

        # Initialize Kubernetes API clients
        core_api = client.CoreV1Api()
        apps_api = client.AppsV1Api()
        batch_api = client.BatchV1Api()
        networking_api = client.NetworkingV1Api()
        apiextensions_api = client.ApiextensionsV1Api()

        cluster_info = {}

        # Example resource data; see original code for the full set
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
        # Pods
        pods = core_api.list_pod_for_all_namespaces()
        cluster_info["pods"] = [
            {
                "name": pod.metadata.name,
                "namespace": pod.metadata.namespace,
                "containers": [
                    {
                        "name": c.name,
                        "image": c.image,
                        "ports": [port.container_port for port in (c.ports or [])],  # Container ports
                        "env": [
                                {"name": env.name, "value": env.value} for env in (c.env or [])  # Environment variables
                                ] if c.env else None,
                    }
                    for c in pod.spec.containers
                                ],
                "status": pod.status.phase,
                "age": str(datetime.now(timezone.utc) - pod.metadata.creation_timestamp).split(".")[0],
                "volumes": [
                    {
                        "name": volumes.name,
                        "mount_path": mount.mount_path,
                    }
                    for volumes in (pod.spec.volumes or [])
                    for container in pod.spec.containers if container.volume_mounts
                    for mount in container.volume_mounts
                          ],
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
                "available_replicas": rs.status.available_replicas or 0,
                "ready_replicas": rs.status.ready_replicas or 0,
                "age": str(datetime.now(timezone.utc) - rs.metadata.creation_timestamp).split(".")[0],
                "labels": rs.metadata.labels or {},
                "selector": rs.spec.selector.match_labels or {},
                "owner": rs.metadata.owner_references[0].name if rs.metadata.owner_references else "None",
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
                "capacity": pv.spec.capacity["storage"],  # e.g., 10Gi
                "storage_class": pv.spec.storage_class_name,
                "reclaim_policy": pv.spec.persistent_volume_reclaim_policy,
                "access_modes": pv.spec.access_modes,
                "volume_mode": pv.spec.volume_mode or "Filesystem",
                "claim": pv.spec.claim_ref.name if pv.spec.claim_ref else "Unbound",
                "namespace": pv.spec.claim_ref.namespace if pv.spec.claim_ref else "N/A",  # Added namespace of the claim
                "age": str(datetime.now(timezone.utc) - pv.metadata.creation_timestamp).split(".")[0],
                "mount_path": pv.metadata.annotations.get("kubernetes.io/mount-path", "N/A"),  # Fetch mount path if annotated
            }
            for pv in pvs.items
        ]

        # Secrets
        secrets = core_api.list_secret_for_all_namespaces()
        cluster_info["secrets"] = [
            {"name": secret.metadata.name, "namespace": secret.metadata.namespace}
            for secret in secrets.items
        ]

        # HPAs
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

        # ConfigMaps
        configmaps = core_api.list_config_map_for_all_namespaces()
        cluster_info["configmaps"] = [
            {
                "name": cm.metadata.name,
                "namespace": cm.metadata.namespace,
                "data": cm.data,  # Include all key-value pairs stored in the ConfigMap
                "creation_time": str(datetime.now(timezone.utc) - cm.metadata.creation_timestamp).split(".")[0],
            }
            for cm in configmaps.items
        ]

        # CronJobs
        cronjobs = batch_api.list_cron_job_for_all_namespaces()
        cluster_info["cronjobs"] = [
            {
                "name": cronjob.metadata.name,
                "namespace": cronjob.metadata.namespace,
                "schedule": cronjob.spec.schedule,
            }
            for cronjob in cronjobs.items
        ]

        # Resource Quotas
        resource_quotas = core_api.list_resource_quota_for_all_namespaces()
        cluster_info["resource_quotas"] = [
            {
                "name": rq.metadata.name,
                "namespace": rq.metadata.namespace,
                "hard": rq.spec.hard,
            }
            for rq in resource_quotas.items
        ]

        # Events
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

        # Network Policies
        network_policies = networking_api.list_network_policy_for_all_namespaces()
        cluster_info["network_policies"] = [
            {
                "name": np.metadata.name,
                "namespace": np.metadata.namespace,
            }
            for np in network_policies.items
        ]

        # CRDs
        crds = apiextensions_api.list_custom_resource_definition()
        cluster_info["crds"] = [
            {"name": crd.metadata.name} for crd in crds.items
        ]

        # StatefulSets
        statefulsets = apps_api.list_stateful_set_for_all_namespaces()
        cluster_info["statefulsets"] = [
            {
                "name": ss.metadata.name,
                "namespace": ss.metadata.namespace,
                "replicas": ss.spec.replicas,
            }
            for ss in statefulsets.items
        ]

        # Ingresses
        ingresses = networking_api.list_ingress_for_all_namespaces()
        cluster_info["ingresses"] = [
            {
                "name": ingress.metadata.name,
                "namespace": ingress.metadata.namespace,
                "host": ingress.spec.rules[0].host if ingress.spec.rules else "N/A",
            }
            for ingress in ingresses.items
        ]

        # Kubeconfig Details
        kubeconfig_path = os.path.expanduser("~/.kube/config")
        with open(kubeconfig_path) as kube_config_file:
            cluster_info["kubeconfig"] = kube_config_file.read()

        # Current Context
        cluster_info["current_context"] = config.list_kube_config_contexts()[1]

        # All Contexts
        cluster_info["all_contexts"] = [
            context["name"] for context in config.list_kube_config_contexts()[0]
        ]

        return cluster_info
    
    except Exception as e:
        raise Exception(f"Error gathering Kubernetes information: {e}")

#Function to send the data to GPT model   
def query_llm(cluster_data, user_query):
    """
    Sends the cluster_data + user_query to OpenAI ChatCompletion
    and returns the LLM's single-word response (or "Unknown").
    """
    try:
        prompt = f"""
        You are an expert Kubernetes assistant. Given the following information about a Kubernetes cluster:

        {json.dumps(cluster_data, indent=2)}

        Answer the following query related to Kubernetes. Ensure your response is always a **single word**.
        If the query is not relevant to Kubernetes or the provided information, respond with "Unknown".

        Query: {user_query}
        """

        openai.api_key = os.getenv('OPENAI_API_KEY')  #OpenAI key

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=50,
            temperature=0.5
        )

        return response.choices[0].message["content"].strip()

    except Exception as e:
        raise Exception(f"Error querying the LLM: {e}")

@app.route("/query", methods=["POST"])
def query_kubernetes():
    """
    Expects a JSON payload with a single 'query' string.
    Returns a JSON response with 'query' and 'answer'.
    """
    try:
        # Parse incoming JSON and validate with Pydantic
        payload = request.get_json(force=True)
        query_req = QueryRequest(**payload)  # Raises ValidationError if missing/invalid

        # Gather cluster data once
        cluster_data = gather_kubernetes_data()

        # Get the raw answer from the LLM
        raw_answer = query_llm(cluster_data, query_req.query)

        # Use the helper function to strip the suffix if it matches the pattern
        # This ensures we don't remove meaningful parts like "redis-docker",
        # but we do remove random suffixes like "6b5f4cf68c-6g5lt".
        cleaned_answer = strip_k8s_suffix(raw_answer)

        # Log the processed query and answer
        logging.info(f"Processed Query: {query_req.query} -> {cleaned_answer}")

        # Construct a typed response
        query_res = QueryResponse(query=query_req.query, answer=cleaned_answer)

        # Return JSON
        return jsonify(query_res.dict())

    except ValidationError as e:
        # If the request body doesn't match the Pydantic schema
        logging.error(f"Validation Error: {e}")
        return jsonify({"error": "Invalid request payload", "details": e.errors()}), 400

    except Exception as e:
        logging.error(f"Error handling request: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

                         

