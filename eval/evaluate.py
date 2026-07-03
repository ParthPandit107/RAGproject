import json
import time
import requests

with open("eval/questions.json", "r") as f:
    questions = json.load(f)

correct_answers = 0
correct_retrieval = 0

latencies = []

for item in questions:

    start = time.time()

    response = requests.post(
        "http://127.0.0.1:8000/ask",
        json={
            "question": item["question"]
        }
    )

    latency = time.time() - start

    latencies.append(latency)

    result = response.json()

    answer = result["answer"]

    retrieved_sources = [
        source["source"]
        for source in result["sources"]
    ]

    if item["expected_answer"].lower() in answer.lower():
        correct_answers += 1

    if item["expected_source"] in retrieved_sources:
        correct_retrieval += 1

    print("=" * 60)
    print("Question :", item["question"])
    print("Answer   :", answer)
    print("Sources  :", retrieved_sources)
    print("Latency  :", round(latency, 3), "seconds")

answer_accuracy = (
    correct_answers /
    len(questions)
) * 100

hit_rate = (
    correct_retrieval /
    len(questions)
) * 100

average_latency = (
    sum(latencies) /
    len(latencies)
)
recall = hit_rate

print("\n========== Evaluation Summary ==========")
print(f"Answer Accuracy : {answer_accuracy:.2f}%")
print(f"Hit Rate@3      : {hit_rate:.2f}%")
print(f"Recall@3       : {recall:.2f}%")
print(f"Average Latency : {average_latency:.3f} sec")