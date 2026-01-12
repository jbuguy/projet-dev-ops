import time
import os
import mlflow
import pandas as pd
import numpy as np
from core.rag import rag_service

# --- Configuration ---
TEST_SET = [
    {
        "query": "What modules are taught in the first semester of the 1st year?",
        "expected_source_partial": "esprit_curriculum", 
        "expected_fact_keywords": ["Programming 1", "Math 1", "Computer Systems"],
        "type": "factual"
    },
    {
        "query": "How many hours is the Java Programming course?",
        "expected_source_partial": "esprit_curriculum",
        "expected_fact_keywords": ["42H"], 
        "type": "factual"
    },
    {
        "query": "What happens in the 5th year?",
        "expected_source_partial": "esprit_curriculum",
        "expected_fact_keywords": ["Stage ingénieur", "PFE", "Projet de Fin d'Études"],
        "type": "factual"
    },
    {
        "query": "What is the total duration of the engineering cycle?",
        "expected_source_partial": "esprit_curriculum",
        "expected_fact_keywords": ["5 ans", "BAC+5"],
        "type": "factual"
    }
]

def calculate_exactness(generated_answer, expected_keywords):
    """
    Checks what percentage of expected keywords appear in the answer.
    """
    generated_lower = generated_answer.lower()
    matches = [kw for kw in expected_keywords if kw.lower() in generated_lower]
    return len(matches) / len(expected_keywords) if expected_keywords else 0

def evaluate_project_metrics():
    print("--- Starting Project 6 Evaluation Suite (ESPRIT Data) ---")
    
    # Ensure RAG is loaded
    if not rag_service.is_ready:
        rag_service.load_artifacts()

    # Set up MLflow experiment
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("ESPRIT_Curriculum_Evaluation")

    with mlflow.start_run(run_name="curriculum_test"):
        
        results = []
        latencies = []
        retrieval_accuracies = []
        exactness_scores = []
        
        for item in TEST_SET:
            start_time = time.time()
            
            # 1. Execute Search (Single turn, no history)
            retrieved_chunks, _ = rag_service.search(item["query"], history=[])
            
            # 2. Generate Answer
            response = rag_service.generate_answer(item["query"], retrieved_chunks)
            
            end_time = time.time()
            latency = end_time - start_time
            latencies.append(latency)
            
            # --- Metric 1: Retrieval Accuracy ---
            hit = False
            found_sources = []
            for chunk in retrieved_chunks:
                source = chunk['metadata']['source']
                found_sources.append(source)
                if item["expected_source_partial"] in source:
                    hit = True
            
            retrieval_accuracies.append(1 if hit else 0)
            
            # --- Metric 2: Factual Exactness ---
            exactness = calculate_exactness(response["answer"], item["expected_fact_keywords"])
            exactness_scores.append(exactness)
            
            # Console Output
            print(f"\nQuery: {item['query']}")
            print(f" - Latency: {latency:.4f}s")
            print(f" - Source Hit: {hit} (Found: {found_sources})")
            print(f" - Fact Exactness: {exactness:.2f} (Keywords: {item['expected_fact_keywords']})")
            print(f" - Answer Preview: {response['answer'][:100]}...")

            results.append({
                "query": item["query"],
                "latency": latency,
                "source_hit": hit,
                "exactness": exactness,
                "generated_answer": response["answer"]
            })

        # --- Aggregating Metrics ---
        avg_latency = np.mean(latencies)
        avg_retrieval_acc = np.mean(retrieval_accuracies)
        avg_exactness = np.mean(exactness_scores)

        print("\n" + "="*40)
        print("FINAL REPORT")
        print("="*40)
        print(f"Average Latency:      {avg_latency:.4f}s")
        print(f"Retrieval Accuracy:   {avg_retrieval_acc * 100:.2f}%")
        print(f"Factual Exactness:    {avg_exactness * 100:.2f}%")

        # Log to MLflow
        mlflow.log_metric("avg_latency_seconds", avg_latency)
        mlflow.log_metric("retrieval_accuracy", avg_retrieval_acc)
        mlflow.log_metric("factual_exactness", avg_exactness)
        
        # Save CSV
        df = pd.DataFrame(results)
        df.to_csv("esprit_eval_results.csv", index=False)
        mlflow.log_artifact("esprit_eval_results.csv")

if __name__ == "__main__":
    evaluate_project_metrics()