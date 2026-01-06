## Experiment Results & Issues (KO/EN)

### 데이터 / Data
- Reviews: Kaggle [Cyberpunk 2077 Steam Reviews](https://www.kaggle.com/datasets/kamilneumann/cyberpunk-2077-steam-reviews) (English, 2020-12-10 ~ 2023-12-13)
- Ground Truth: `datasets/ground_truth_steam.csv`, `datasets/ground_truth_stock.csv`
- 시뮬레이션 날짜: 원본 `simulation_dates.csv`는 2024-08-01까지 포함 → 2023-12-13 이후는 리뷰 없음

### 주요 이슈 / Key Issues
1) **Team3 all-NO**  
   - 원인: `rag_modules.py`가 `timestamp` 필터 사용 → ChromaDB 메타데이터 `date` 필드와 불일치 → 검색 0건 → 빈 컨텍스트로 NO 쏠림  
   - 조치: 필터를 `date <= YYYYMMDD`로 수정, days_diff 재계산
2) **날짜 범위 불일치**  
   - 리뷰는 2023-12-13까지, 시뮬레이션은 2024-08-01까지 → 후기 구간은 의미 없는 빈 검색  
   - 조치: `simulation_model_c.py`에서 시뮬레이션 날짜를 `<= 2023-12-13`으로 제한 (Team3)  
   - Team2도 동일하게 컷오프 적용 재실행을 권장
3) **비동기 시도**  
   - Async(20 동시, 재시도 포함) 시도 → ChromaDB thread-safe 이슈와 키 레이트리밋으로 불안정  
   - 현재 권장: 동기 실행 또는 날짜 샤드 + 키 분할 실행

### 현재 실행 현황 (요약)
- Team2 동기 실행(b): 진행 중 (PID 77918, 로그 `archive/team02_run.log`)  
- Team2 샤드(b-2): 3개 프로세스, 서로 다른 키, 출력 `static_rag/Team2_StaticRAG_Results_run2_p{0,1,2}.csv` (실행 중)  
- Team3 동기(c): 이전 all-NO 결과 확인 → 코드 수정 완료, 재실행 필요

### 재실행 가이드
- Team2 동기: `python -u static_rag/simulation_model_b.py`  
- Team2 샤드(예시 3분할):  
  ```
  python -u static_rag/simulation_model_b_multi.py --process-index 0 --process-count 3 --output static_rag/Team2_StaticRAG_Results_run2_p0.csv
  python -u static_rag/simulation_model_b_multi.py --process-index 1 --process-count 3 --output static_rag/Team2_StaticRAG_Results_run2_p1.csv
  python -u static_rag/simulation_model_b_multi.py --process-index 2 --process-count 3 --output static_rag/Team2_StaticRAG_Results_run2_p2.csv
  ```
  (키를 프로세스별로 다르게 설정, 완료 후 CSV 병합)
- Team3 동기(수정본, 컷오프 적용):  
  `python -u time_aware_rag/simulation_model_c.py`

### 참고 자료
- 시퀀스/파이프라인 다이어그램: `png/rag.png`  
- README(ko/en): `README_ko.md`, `README_en.md`
- 현재 ChromaDB 용량: 약 1.3GB (`datasets/chroma_db`), 요청 시 압축 가능


