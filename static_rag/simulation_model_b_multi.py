#!/usr/bin/env python3
"""
Team 2: Static RAG Simulation (sharded, sync)
- process_index / process_count 로 날짜를 분할 실행
- 각 프로세스는 별도 OPENAI_API_KEY 로 병렬 실행 가능
- 결과는 지정한 output CSV에만 기록 (충돌 방지)
"""

import os
import sys
import json
import argparse
import pandas as pd
import random
from openai import OpenAI

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.persona_generator import generate_balanced_personas, Persona
from utils.search_queries import GAMER_TYPE_QUERIES, GENERAL_QUERY
from utils.llm_config import TEMPERATURE
from static_rag.rag_modules import RAGRetriever

# ---------------------------------------------------------------------------
# LLM 클라이언트 (동기)
# ---------------------------------------------------------------------------

def get_llm_client_sync():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
    client = OpenAI(api_key=api_key)
    model_name = "gpt-4o-mini"
    return client, model_name

client, MODEL_NAME = get_llm_client_sync()
print(f"✅ Using model: {MODEL_NAME} (Team 2 - shard)")

SIMULATION_DATES_FILE = "datasets/simulation_dates.csv"
# 리뷰 데이터가 2023-12-13까지이므로 이후 날짜는 제외
SIMULATION_DATE_CUTOFF = "2023-12-13"

# ---------------------------------------------------------------------------
# Prompt 생성
# ---------------------------------------------------------------------------

def create_prompt(agent: Persona, current_date: str, context: list):
    context_str = "\n".join(context) if context else "(No reviews found.)"
    return f"""[ROLE]
You are a {agent.age} {agent.gender}.
Personality: '{agent.gamer_type_name_display}' ({agent.description})

[DATE]
Today is {current_date}.

[SEARCH RESULTS]
Reviews selected based on your interests and recentness (Time-Weighted):
{context_str}

[TASK]
Decide to buy 'Cyberpunk 2077' or not based strictly on the reviews above.
- The reviews are filtered by relevance and recency.
- Trust these reviews as the most important information available to you.

[OUTPUT]
JSON only:
{{
    "decision": "YES" or "NO",
    "reasoning": "Explain why based on the reviews."
}}
"""

# ---------------------------------------------------------------------------
# LLM 호출
# ---------------------------------------------------------------------------

def call_llm(prompt: str) -> dict:
    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=TEMPERATURE,
        )
        return json.loads(res.choices[0].message.content)
    except Exception as e:
        print(f"[LLM Error] {e}")
        return {"decision": "NO", "reasoning": "Error"}

# ---------------------------------------------------------------------------
# 메인 실행
# ---------------------------------------------------------------------------

def run_sharded(process_index: int, process_count: int, output_file: str, n_per_type: int = 13):
    print("=" * 70)
    print(f"Task 2: Static RAG Simulation (Shard {process_index}/{process_count})")
    print("=" * 70)

    retriever = RAGRetriever()

    dates_df = pd.read_csv(SIMULATION_DATES_FILE)
    simulation_dates = [
        d for d in dates_df['date'].tolist()
        if pd.to_datetime(d) <= pd.to_datetime(SIMULATION_DATE_CUTOFF)
    ]

    # 날짜를 shard별로 분할 (라운드 로빈)
    shard_dates = [d for i, d in enumerate(simulation_dates) if i % process_count == process_index]
    print(f"Total dates: {len(simulation_dates)}, this shard: {len(shard_dates)} dates")

    personas = generate_balanced_personas(n_per_type=n_per_type)
    print(f"Generated {len(personas)} agents.")

    results = []
    total_steps = len(shard_dates) * len(personas)
    step_count = 0
    flush_every = 50  # 50개마다 파일에 append

    for date_str in shard_dates:
        print(f"\n📅 Date: {date_str}")
        for persona in personas:
            step_count += 1

            # 쿼리 선정
            agent_queries = GAMER_TYPE_QUERIES.get(persona.gamer_type, [])
            if len(agent_queries) >= 4:
                selected_queries = random.sample(agent_queries, 4)
            else:
                selected_queries = agent_queries
            selected_queries.append(GENERAL_QUERY)

            # 검색
            candidates = []
            for query in selected_queries:
                reviews = retriever.retrieve_reviews(query, date_str, top_k=2)
                candidates.extend(reviews)
            unique_candidates = list(set(candidates))
            final_docs = unique_candidates[:5]

            # 프롬프트
            prompt = create_prompt(persona, date_str, final_docs)

            # LLM 호출
            print(f"[{step_count}/{total_steps}] {persona.gamer_type_name_display}...", end=" ", flush=True)
            res = call_llm(prompt)
            decision = res.get("decision", "NO").upper()
            decision = "YES" if "YES" in decision else "NO"
            print(f"-> {decision}")

            results.append({
                "Agent_ID": persona.id,
                "Name": persona.name,
                "Persona_Type": persona.gamer_type_name_display,
                "Decision": decision,
                "Simulation_Date": date_str,
                "Reasoning": res.get("reasoning", "")
            })

            if len(results) >= flush_every:
                header = not os.path.exists(output_file) or os.path.getsize(output_file) == 0
                pd.DataFrame(results).to_csv(
                    output_file,
                    mode="a",
                    index=False,
                    header=header,
                    encoding="utf-8-sig"
                )
                print(f"💾 Saved chunk: total rows so far ~{len(results)} (appended)", flush=True)
                results = []

    # 남은 결과 저장
    if results:
        header = not os.path.exists(output_file) or os.path.getsize(output_file) == 0
        pd.DataFrame(results).to_csv(
            output_file,
            mode="a",
            index=False,
            header=header,
            encoding="utf-8-sig"
        )
        print(f"💾 Saved final chunk: rows {len(results)}", flush=True)

    print("\n" + "=" * 70)
    print(f"Simulation completed. Results saved to {output_file}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--process-index", type=int, default=0, help="0-based shard index")
    parser.add_argument("--process-count", type=int, default=1, help="total shards")
    parser.add_argument("--output", type=str, default="static_rag/Team2_StaticRAG_Results_shard.csv")
    args = parser.parse_args()

    run_sharded(
        process_index=args.process_index,
        process_count=args.process_count,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()

