import json
import asyncio
import aiohttp
from tqdm import tqdm

INPUT_FILE = "prompts.jsonl"
OUTPUT_FILE = "distilled_dataset.jsonl"

API_URL = "http://localhost:8000/v1/chat/completions"  
MODEL_NAME = "Qwen/Qwen1.5-14B-Chat"

HEADERS = {
    "Content-Type": "application/json"
}


async def fetch_completion(session, prompt: str):
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 512
    }
    async with session.post(API_URL, headers=HEADERS, json=data) as resp:
        if resp.status != 200:
            print(f"Error {resp.status}: {await resp.text()}")
            return None
        out = await resp.json()
        return {
            "prompt": prompt,
            "response": out["choices"][0]["message"]["content"]
        }


async def main():
    prompts = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            prompts.append(obj["prompt"])

    results = []
    async with aiohttp.ClientSession() as session:
        for prompt in tqdm(prompts):
            result = await fetch_completion(session, prompt)
            if result:
                results.append(result)
                with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
