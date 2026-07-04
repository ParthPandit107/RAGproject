import os
import json
import time
import requests

ASK_URL = "http://127.0.0.1:8000/ask"
RETRIEVE_URL = "http://127.0.0.1:8000/retrieve"
CACHE_FILE = "eval/cache.json"


def load_cache():

    if os.path.exists(CACHE_FILE):

        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)

        except Exception:
            return {}

    return {}


def save_cache(cache):

    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)

        return True

    except Exception:
        return False


def ask_with_retry(question, cache):

    if question in cache:
        return cache[question], 0.0

    retries = 3
    delay = 2

    for attempt in range(retries):

        try:
            start = time.time()

            response = requests.post(
                ASK_URL,
                json={
                    "question": question
                }
            )

            latency = time.time() - start

            if response.status_code == 200:
                result = response.json()
                answer = result["answer"]

                cache[question] = answer
                save_cache(cache)

                return answer, latency

            elif response.status_code == 502:
                detail = response.json().get("detail", "")

                if "quota" in detail.lower() or "limit" in detail.lower() or "429" in detail:
                    time.sleep(delay)
                    delay *= 2
                    continue

                return None, 0.0

            else:
                return None, 0.0

        except Exception:
            time.sleep(delay)
            delay *= 2

    return None, 0.0


cache = load_cache()

with open("eval/questions.json", "r") as f:
    questions = json.load(f)

correct_answers = 0
correct_retrieval = 0
total_evaluated_answers = 0
latencies = []
quota_exceeded_flag = False

for index, item in enumerate(questions, start=1):

    print(f"\n[{index}/{len(questions)}] Processing...")

    retrieved_sources = []

    try:
        response = requests.post(
            RETRIEVE_URL,
            json={
                "question": item["question"]
            }
        )

        if response.status_code == 200:
            result = response.json()

            retrieved_sources = [
                source["source"]
                for source in result
            ]

    except Exception as e:
        print(f"Retrieval error: {e}")

    if item["expected_source"] in retrieved_sources:
        correct_retrieval += 1

    answer = None
    latency = 0.0

    if not quota_exceeded_flag:

        if item["question"] not in cache:
            time.sleep(3)

        answer, latency = ask_with_retry(item["question"], cache)

        if latency > 0.0:
            latencies.append(latency)

        if answer is None:
            print("Gemini API daily quota exceeded or unavailable. Skipping LLM verification for remaining questions.")
            quota_exceeded_flag = True

    if answer is not None:
        total_evaluated_answers += 1

        if item["expected_answer"].lower() in answer.lower():
            correct_answers += 1

    print("=" * 60)
    print("Question :", item["question"])
    print("Answer   :", answer if answer is not None else "[Skipped/Error]")
    print("Sources  :", retrieved_sources)

    if latency > 0.0:
        print("Latency  :", round(latency, 3), "seconds")

answer_accuracy = (
    (correct_answers / total_evaluated_answers * 100)
    if total_evaluated_answers > 0
    else 0.0
)

hit_rate = (
    correct_retrieval / len(questions)
) * 100

average_latency = (
    (sum(latencies) / len(latencies))
    if len(latencies) > 0
    else 0.0
)

print("\n========== Evaluation Summary ==========")
print(f"Answer Accuracy : {answer_accuracy:.2f}% (Evaluated: {total_evaluated_answers}/{len(questions)})")
print(f"Hit Rate@3      : {hit_rate:.2f}%")
print(f"Recall@3        : {hit_rate:.2f}%")
print(f"Average Latency : {average_latency:.3f} sec")