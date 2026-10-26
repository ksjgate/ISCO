# ISCO — 인라인 3D 검사 설정 최적화 시뮬레이터

**ISCO**(Inspection Setup Configuration Optimizer)는 부품의 3D 모델과 결함 종류를 넣으면,
가상 컨베이어 공정 안에서 **3D 카메라를 몇 대, 어디에 둘지 · 회전대를 쓸지 · 벨트를 얼마나 빨리
돌릴지 · 카메라를 어떻게 촬영시킬지**를 직접 시험해 보고, 가장 좋은 검사 설정을 추천하는
시뮬레이터다. 가상 공정은 NVIDIA Isaac Sim으로 만든 디지털 트윈(실제 공정을 컴퓨터 안에 똑같이
만든 것)이다.

2026-2 산학캡스톤디자인, 팀 강화실패.

---

## 1. 이 프로젝트가 하는 일

### 1.1 풀려는 문제
- 눌림·휨·미세 단차처럼 색은 그대로이고 **모양만 달라지는 결함**은 2D 카메라로 잡기 어려워
  거리(깊이)를 재는 3D 카메라 검사를 쓴다.
- 그런데 카메라 수·위치·부품 회전·벨트 속도는 아직 경험으로 정한다. 결함 면이 어느 카메라에도
  안 보이면 검출 모델이 아무리 좋아도 놓치고, 카메라를 무작정 늘리면 비용과 검사 시간이 늘어난다.
- 기존 도구는 정지한 부품의 표면을 얼마나 덮는지(커버리지)로 카메라 위치를 계산하거나 공정
  흐름만 시뮬레이션한다. ISCO는 **움직이는 공정 안에서, 실제 검출기의 성능을 기준으로** 설정을 찾는다.

### 1.2 입력과 출력
| | 내용 |
|---|---|
| 입력 | 부품 3D 모델(설계 CAD 또는 3D 스캔), 결함 사전(결함 종류·크기), 제약(예: 카메라 3대 이하, 한 개당 1.5초 이하) |
| 출력 | 제약 안의 **권장 설정**, 성능–비용 절충 곡선(파레토 전선: 이보다 싸면서 더 잘 잡는 설정이 없는 설정들), 결함 종류별 가시성·미검률, 카메라 배치 그림, 웹 리포트 |

### 1.3 성능과 비용
- **성능 = 과검률 1%에서의 미검률 (FNR@1%FPR).** 정상품 100개 중 1개만 불량으로 잘못 걸리도록
  판정 기준선을 정한 뒤, 결함 부품 중 몇 %를 놓치는지 센다. 낮을수록 좋다.
  95% 신뢰구간은 표본을 1,000번 다시 뽑아(부트스트랩) 계산한다.
- **비용 = 카메라 수 + 사이클 시간.** 사이클 시간 = 부품 간격 0.3 m ÷ 벨트 속도 + 회전대 시간.
  벨트 0.1 / 0.3 / 0.6 m/s → 3.0 / 1.0 / 0.5초, 회전대 90° 한 번 +0.9초, 180° 한 번 +1.1초,
  90°씩 세 번 +1.7초.

### 1.4 바꿔 보는 설정 (탐색 공간)
| 설정 | 값 | 명령에서 쓰는 이름 |
|---|---|---|
| 카메라 수 | 1–6대 | `--slots`에 넣는 슬롯 개수 |
| 카메라 위치 | 검사 지점 중심 반구 위 **96곳** = 방위 0–330°(30° 간격) 12 × 고도 20/40/60/80° 4 × 거리 0.25/0.40 m 2. 같은 방향 두 대 금지, 시선끼리 30° 이상 | 슬롯 번호 `k = (거리번호 × 4 + 고도번호) × 12 + 방위번호` (방위번호 0–11 = 0°부터 30°씩, 고도번호 0–3 = 20/40/60/80°, 거리번호 0–1 = 0.25/0.40 m. 예: `12,16,20` = 고도 40°, 120° 간격, 0.25 m의 3대 기준 배치) |
| 회전대 | 없음 / 90° 한 번 / 180° 한 번 / 90°씩 세 번 (부품을 멈추고 돌려 가며 찍음) | `none`, `r90x1`, `r180x1`, `r90x3` |
| 벨트 속도 | 0.1 / 0.3 / 0.6 m/s | `--speed 0`, `1`, `2` |
| 촬영 방식 | 모든 카메라 동시 / 카메라마다 순서대로 | `simultaneous`, `staggered` |

### 1.5 동작 흐름
```
부품 CAD / 3D 스캔 ─► 기준 자세 메시(가장 낮게 놓이는 자세, 최대 치수 40 mm, 0.25 mm 메시)
                         │
                         ▼
변형 세트: 정상품 140개(검출기 학습 40 + 판정 기준선 100) + 결함 부품 72개(탐색용 36 + 성적용 36)
          + 어떤 카메라로도 볼 수 없는 바닥 결함 대조군 6개
                         │
                         ▼
시점 저장본: 모든 변형 부품을 카메라 자리 96곳에서 한 번씩 미리 촬영(부품당 약 2만 장, 4–5분)
                         │
                         ▼
1단계(빠른 거르기): 결함 표면이 카메라에 보이는 비율(가시성)을 설정 하나당 0.35 ms에 계산
                   → 유전 알고리즘 1,500개 + 무작위 500개 → 후보 약 1,400개
                         │
                         ▼
2단계(정밀 평가): 저장본 조합 → 센서 잡음·벨트 움직임 모델 → 점 합치기·정합(ICP)
                 → FPFH 메모리 뱅크 검출기 → 과검률 1%에서의 미검률 (설정 하나 약 1분)
                 베이지안 최적화가 다음에 시험할 설정을 고름(부품당 40회)
                         │
                         ▼
결과: 성능–비용 곡선 + 제약 안 권장 설정 + 결함 종류별 가시성·미검률 + 웹 리포트
```

- **센서 모델**: 구조광 3D 카메라(1024×1024, 수평 시야 20°, 0.25 m에서 화소당 0.086 mm). 잡음은
  거리 제곱에 비례하고, 비스듬한 면은 측정 실패가 늘며, 물체 경계에 튀는 점이 생긴다. 실제 카메라처럼
  깊이가 2 mm 넘게 뛰는 경계 화소는 지운다. 잡음 수준 4단계(N0 없음, N1 0.05 mm, **N2 0.15 mm 기본**,
  N3 0.30 mm, 0.25 m 기준)와 노출 10배 다중 촬영(N2MS).
- **공정 모델**: 촬영 신호 지연(표준편차 5 ms), 노출 중 번짐(2 ms), 회전대 재안착 오차(0.5 mm, 1°).
  부품이 벨트에 떨어져 놓이는 자세는 Isaac 물리 시뮬레이션에서 측정한 분포를 쓴다.
- **검출기**: 정상품만 배우는 메모리 뱅크(점마다 주변 형상을 33개 숫자로 요약하는 FPFH 특징을 1·2·4 mm
  세 크기로 계산 + 부품 위 위치 정보)가 기본. 비교용으로 설계 CAD까지의 거리만 보는 학습 없는 검출기와
  DINOv2 딥러닝 특징 검출기가 있다.
- **결함 사전**(6종): 눌림, 돌출, 구멍, 절단(모서리 깨짐), 균열, 휨. 크기 3단계 × 보이기 어려운 정도
  3단계로 고르게 섞고, 결함이 있는 삼각형마다 정답 표시를 붙인다.

---

## 2. 실행 환경

| 항목 | 내용 |
|---|---|
| 확인한 장비 | Ubuntu 24.04, NVIDIA RTX 3080 Ti (12 GB), CPU 24코어, RAM 61 GB |
| 시뮬레이터 | NVIDIA Isaac Sim 5.1 (헤드리스 렌더 가능) |
| 파이썬 | conda 환경 **`env_isaaclab`** (Python 3.11, PyTorch 2.7.0+cu128) — **새 가상환경을 만들지 말고 이 환경만 쓴다** |
| 저장 공간 | 원본 데이터셋 약 56 GB(아래 표), 부품당 시점 저장본 1.3–9 GB(현재 14개 부품 58 GB), 변형 세트 4.7 GB |

ROS 2가 설치된 셸은 Python 3.10용 `PYTHONPATH`를 내보내므로 ISCO를 돌리기 전에 비운다.

```bash
conda activate env_isaaclab
export PYTHONPATH=
```

## 3. 설치

```bash
conda activate env_isaaclab && export PYTHONPATH=
bash scripts/install_extra_deps.sh          # open3d, optuna, faiss-cpu, pymeshlab, manifold3d 등
pip install -e . --no-deps                  # isco 패키지와 `isco` 명령 설치
isco --help
```

`scripts/constraints-isaacsim.txt`가 Isaac Sim이 의존하는 버전(`numpy==1.26.0`, `psutil==5.9.8`,
`torch==2.7.0+cu128`)을 고정해, 추가 패키지가 이것들을 올려 버리지 않게 한다.
실험 기록을 Weights & Biases에도 남기려면 `wandb`를 설치하고 `ISCO_WANDB=1`을 켠다(선택).

---

## 4. 사용한 데이터셋

| 데이터셋 | 무엇인가 | 사용한 물체 | ISCO에서의 역할 | 라이선스 | 받는 법 |
|---|---|---|---|---|---|
| **T-LESS** | 질감 없는 전기 부품 30종의 설계 CAD 모델(물체 인식 벤치마크 BOP의 일부) | 13번(나사산 있는 전구 소켓), 18번(원뿔형 하우징), 29번(뚜껑 달린 직육면체 박스), 30번(원형 박스) → `tless_13/18/29/30` | **검증 실험의 주 대상**. 13·18번은 자기 가림이 큰 부품, 29·30번은 단순한 부품 | CC BY 4.0 (재배포 가능) | 자동 (`download_datasets.sh tless`, 34 MB) |
| **MulSen-AD** | 실제 산업 부품을 360° 레이저로 스캔한 다중 센서 이상 탐지 데이터셋(CVPR 2025). 실제 결함 스캔 포함 | 너트, 평판 패드, 캡슐 → `mulsen_nut/flat_pad/capsule` | 실제 스캔 형상의 부품 + **실제 결함 스캔**으로 "시뮬레이션 순위가 실제 결함에서도 유지되는가" 확인 | MIT | 자동 (`download_datasets.sh mulsen`, zip 약 9 GB) |
| **Real3D-AD** | 장난감·식품 모양 물체의 고정밀 360° 스캔 이상 탐지 데이터셋(NeurIPS 2023) | 캔디바, 불가사리 → `real3d_candybar/starfish` | 스캔 점군에서 메시를 만들어 부품으로 준비(데이터 빌드까지 완료, 검증 실험에는 아직 미사용) | CC BY 4.0 | 자동 (`download_datasets.sh real3d`, Google Drive) |
| **Anomaly-ShapeNet v2** | ShapeNet 물체에 결함을 넣은 3D 이상 탐지 데이터셋(CVPR 2024) | — | 결함 사전(결함 종류) 설계 때 참고. **현재 코드와 실험에서는 읽지 않는다** | CC BY 4.0 | 자동 (`download_datasets.sh anomaly_shapenet`, 3.8 GB, 선택) |
| **MVTec 3D-AD** | 실제 산업용 구조광 3D 카메라로 위에서 찍은 산업 물체 10종의 이상 탐지 데이터셋 | **dowel**(지느러미가 달린 플라스틱 벽 앵커), **cable gland**(케이블 글랜드) 두 종류만 | (1) 우리 검출기 구현이 공개 논문 수치를 재현하는지 확인 (2) 실제 3D 카메라와 시뮬레이터 센서 모델의 차이 측정 | CC BY-NC-SA 4.0 (비상업) | **Kaggle에서 받을 수 있다** |
| **Fusion 360 Gallery** | Autodesk Fusion 360 공개 설계 데이터의 부품 | 시제품에서 쓰던 부품 5종 → `fusion_22340/26768/28283/41229/51876` | 데모와 개발 점검 전용(검증 실험에는 쓰지 않음) | 비상업 연구용, **재배포 금지** | 저장소에 포함하지 않음. `data/parts/raw/fusion360/`에 로컬로만 둔다 |

이 밖에 `dowel`이라는 이름의 부품은 데이터셋이 아니라 코드로 만든 원기둥 핀(`isco parts prepare-dowel`)이다.

### 4.1 내려받기

```bash
bash scripts/download_datasets.sh                         # T-LESS, MulSen-AD, Anomaly-ShapeNet, Real3D-AD 전부
bash scripts/download_datasets.sh tless mulsen real3d     # Anomaly-ShapeNet 없이
```
각 데이터셋은 `data/datasets/<이름>/raw/`에 내려받아 같은 곳의 `extracted/`에 풀린다
(Hugging Face `hf` 명령과 `gdown`을 쓴다). 내려받은 압축 파일 안의 어떤 것도 실행하지 않는다.

MVTec 3D-AD는 Kaggle에서 받아 두 물체 폴더만 다음 위치에 둔다.
```
data/datasets/mvtec3d/raw/
├── dowel/
└── cable_gland/
```

### 4.2 `data/` 폴더 (모두 자동 생성, git에서 제외)
```
data/
├── datasets/<이름>/raw/          원본 데이터셋
├── parts/<부품>/                 기준 자세 메시와 부품 정보 (raw/fusion360/ 에 Fusion 360 원본 STL)
├── variants/<부품>/<세트>/        변형 부품 세트(정상·결함·대조군)
├── cache/<부품>/<세트>/           시점 저장본(96곳 × 변형 부품의 깊이·정답 표시)
├── visibility/                   1단계용 결함 가시성 표
├── evalcache/                    2단계 평가 결과 캐시 (중단 후 이어 하기, 실험 간 공유)
├── conveyor/occluder.npz         컨베이어 구조물(카메라 시야를 가리는 물체) 메시
└── settle/<부품>.json            부품이 벨트에 놓이는 자세 분포(Isaac 물리 시뮬레이션)
```
`runs/`(실행 결과), `queue/`(Isaac 작업 대기열)도 자동 생성되며 git에서 제외된다.

---

## 5. 빠르게 돌려 보기

### 5.1 부품 준비 (기준 자세 메시 만들기)
```bash
isco parts prepare-tless --ids 13 --ids 18 --ids 29 --ids 30   # T-LESS 4종
isco parts prepare-mulsen --categories nut --categories flat_pad --categories capsule
isco parts prepare-real3d --categories candybar --categories starfish
isco parts prepare-dowel                                        # 코드로 만든 원기둥 핀
isco parts prepare-fusion                                       # Fusion 360 부품(로컬에 있을 때만)
isco parts list                                                 # 준비된 부품, 출처, 재배포 가능 여부
```

### 5.2 데이터 빌드 (변형 세트 + 시점 저장본 + 가시성 표)
```bash
isco data build tless_29 --preset mvp                 # 부품 하나, 약 4–5분
bash scripts/build_all_mvp.sh                         # 14개 부품 전부
```
- `--preset`: `smoke`(작은 시험용) / `mvp`(검증 실험에 쓴 세트) / `full`.
- `--yaw-mode random --set-id mvp_yawrand`: 부품이 아무 방향으로나 놓이는 세트(놓임 방향 실험용).
- `--rebuild-sets`: 변형 세트만 다시 만들고, 시점 저장본은 바뀐 부품만 다시 찍는다.

### 5.3 설정 하나 평가 ("설정을 넣으면 미검률과 비용이 나온다")
```bash
isco evaluate tless_29 --slots 4,39,59 --rotation r90x1 --speed 2 --trigger simultaneous
```
`--noise`(N0–N3, N2MS), `--process`(`inline_default` 인라인 공정 / `static_capture` 정지 촬영),
`--detector`(`fpfh_bank` / `cad_distance` / `dinov2_bank`)로 조건을 바꿀 수 있다. 저장본이 있으면
설정 하나에 약 1분, 같은 설정을 다시 평가하면 캐시에서 바로 나온다.

### 5.4 설정 찾기 (1단계 + 2단계 + 리포트)
```bash
isco search tless_29 --search full --constraint "n<=3,T<=1.5"
```
- `--search`: `smoke`(1단계 400개, 2단계 12회) / `demo`(2,000개, 20회) / `full`(2,000개, 60회 × 시드 3개).
- `--constraint "n<=3,T<=1.5"`: 카메라 3대 이하, 사이클 1.5초 이하.
- `--methods`: 2단계 탐색 방법. `bo_warm`(1단계 후보로 시작하는 베이지안 최적화, 다목적 TPE, 기본), `bo_cold`(처음부터 하는 베이지안 최적화),
  `nsga2`(유전 알고리즘), `random`(무작위), `greedy_vis`·`greedy_fnr`(가시성 또는 미검률이 가장 좋아지는 카메라를 하나씩 더하기).

### 5.5 데모 흐름 한 번에
```bash
isco run fusion_51876 --constraint "n<=3,T<=1.5" --out runs/demo_fusion_51876
```
데이터가 없으면 빌드부터 하고, 1단계 → 2단계 → 리포트까지 실행한다.
리포트만 다시 만들려면 `isco report runs/demo_fusion_51876 --constraint "n<=3,T<=1.5"`.

### 5.6 결과 폴더
```
runs/<실행 이름>/
├── run.json          모든 설정값
├── env.json          패키지 버전, git 커밋
├── stage1.jsonl      1단계 후보와 가시성
├── stage2.jsonl      2단계 평가 결과(미검률, 신뢰구간, 결함 종류별 값, 비용)
└── report/           index.html, index.md, recommended.json, 성능–비용 곡선·카메라 배치·결함 종류별 그림
```

---

## 6. Isaac Sim 사용

대량 촬영(시점 저장본)은 Isaac과 같은 결과를 내는 빠른 광선 추적(open3d embree)으로 하고,
Isaac Sim은 정확도 검증·물리 시뮬레이션·데모 화면에 쓴다. 두 방식의 깊이 차이는 1 µm 미만이다
(`python scripts/e01_isaac_vs_embree.py`로 확인).

Isaac은 한 번 켜 두고 작업을 받아 처리하는 **렌더 워커**로 돌린다. GPU는 Isaac 프로세스 하나만 쓰게 한다.
```bash
bash scripts/start_worker.sh                     # 백그라운드 워커 시작 (로그: queue/worker.log)
isco worker status
isco sim export-occluder                         # 컨베이어 메시 → data/conveyor/occluder.npz
isco sim settle fusion_51876                     # 벨트에 떨어뜨려 놓임 자세 측정 → data/settle/
isco sim rgb fusion_51876 --slots 12,16,20       # 설정의 카메라 RGB 사진
isco sim check-rig fusion_51876                  # 96곳 모두 렌더링해 설치 가능 여부 확인
isco worker stop
```

`bash scripts/smoke_gate.sh [부품] [슬롯]`은 한 줄 점검이다: 워커를 켜고, Isaac에서 부품 3개를
2곳에서 찍어 시점 저장본과 비교한 뒤, 카메라 2대 설정 하나를 평가해 미검률과 비용을 JSON으로 출력한다.

팀원이 만든 첫 시제품 스크립트도 그대로 동작한다(내부적으로 `isco.sim`을 쓴다).
```bash
cd isaac_sim
python create_conveyor.py                                    # 화면 실행, 부품이 계속 흐름
python create_conveyor.py --headless --video                 # 데모 동영상 저장
python create_conveyor.py --headless --capture-views --views-part 41229_16283ae1_0018.stl   # 카메라 3대 촬영
```
`bash scripts/isaac_regression.sh`가 위 세 명령을 다시 돌려 확인한다.

---

## 7. 실험 돌리기

### 7.1 원칙
- 검증 실험의 부품·평가 횟수·판정 기준은 **실험 전에 고정**해 두었다. `isco exp run`은
  `--budget`/`--seeds`를 주지 않으면 그 고정된 값을 쓴다.
- 설정을 고를 때 쓰는 **탐색용 결함**과 성적을 매길 때만 쓰는 **성적용 결함**을 나눠, 성적은 탐색에
  쓰지 않은 결함으로만 매긴다.
- 모든 2단계 평가는 `data/evalcache/`에 저장되므로, 실험을 멈췄다 다시 돌리면 끝난 평가는 다시 하지
  않고, 같은 설정을 쓰는 실험끼리 결과를 공유한다.
- 결과는 `runs/exp/<이름>/summary.json`(+ 그림, 평가 기록 `evals_*.jsonl`)에 쌓인다.

### 7.2 실험 목록
`이름`은 명령에 넣는 코드상의 실험 이름이다. 시간은 24코어 장비에서 대기열로 돌렸을 때 실제로 걸린 시간이다.

**점검 실험** — 시뮬레이터와 계산 방법을 믿을 수 있는가

| 이름 | 실험 | 묻는 질문 | 부품 | 시간 |
|---|---|---|---|---|
| `e0_2` | 시점 저장본 조합 근사 검증 | 저장본을 조합한 평가가 Isaac에서 공정 그대로 직접 찍은 것과 같은가 (Isaac 워커 필요) | T-LESS 29·13 | 약 8분 |
| `e0_3` | 결함 정답 표시 면적 검증 | 결함 정답 표시 면적이 공칭 크기와 맞는가 | T-LESS 4종, 너트, fusion_51876 | 1분 이내 |
| `e0_4` | 설정 평가 시간 측정 | 저장본 빌드·1단계·2단계에 얼마나 걸리는가 | T-LESS 13, fusion_51876 | 약 6분 |
| `e0_5` | 결함 크기 보정 | 기준 배치에서 미검률이 목표(20–60%) 안에 드는가 | T-LESS 4종, 너트, 평판 패드 | 약 20분 |
| `e0_6` | 검출기 구현 재현 검증 | 공개 기준 검출기(BTF)를 다시 구현하면 공개 수치가 나오는가 | MVTec 3D-AD dowel, cable gland | — |
| `result_sanity` | 카메라 수 기본 점검 | 사람이 정한 기본 배치(1/2/3/4/6대)에서 결과가 상식적인가 | Fusion 3종 | 약 6분 |

**가설 검증 실험**

| 이름 | 실험 | 묻는 질문 | 부품 | 시간 |
|---|---|---|---|---|
| `e1` | 부품 간 최적 설정 교차 적용 실험 | 최적 설정은 부품마다 다른가 | T-LESS 4종, 너트, 평판 패드 | 10.5시간 |
| `e2` | 가시성 대리지표 타당성 실험 | 가시성이 높은 설정일수록 실제로 덜 놓치는가 | T-LESS 4종 | 21시간 |
| `e3` | 카메라 추가 효과 실험 | 카메라를 더하면 나빠지는 것이 검출기 고유의 성질인가, 검출 성능으로 직접 탐색하면 나은가 | T-LESS 4종, 너트, 평판 패드 | 26시간 |
| `e4` | 정지 촬영 대 인라인 공정 비교 실험 | 공정을 반영하면 권장 설정이 달라지는가 | T-LESS 13·29·30 | 7.5시간 |
| `e5` | 실제 스캔 결함에서의 순위 유지 실험 | 합성 결함으로 매긴 설정 순위가 실제 결함에서도 유지되는가 | MulSen-AD 너트·평판 패드·캡슐 | 첫 실행은 6시간 뒤 중단(정렬 수정 후 재실행) |

**견고성 실험**

| 이름 | 실험 | 묻는 질문 | 부품 | 시간 |
|---|---|---|---|---|
| `e5b` | 실제 센서와 시뮬레이터 센서 비교 실험 | 시뮬레이터의 센서 잡음이 실제 3D 카메라와 얼마나 다른가 | MVTec 3D-AD dowel, cable gland | 3시간 |
| `e6` | 검출기 교체 강건성 실험 | 검출기를 바꿔도 설정 순위가 유지되는가 | T-LESS 13·29 | 12.5시간 |
| `e7` | 센서 잡음 수준 비교 실험 | 잡음 수준(N1/N2/N3/N2MS)을 바꾸면 미검률과 순위가 어떻게 변하는가 | T-LESS 13·29 | 20시간 |
| `e7b` | 부품 놓임 방향 실험 | 부품이 아무 방향으로나 놓이면 어떻게 되는가 | T-LESS 29 | 4.5시간 |
| `e8` | 탐색 방법 효율 비교 실험 | 1단계로 시작한 베이지안 최적화가 다른 방법보다 적은 평가로 좋은 설정을 찾는가 | T-LESS 13·29 | 1일 이상 (T-LESS 13번만 약 1일) |

### 7.3 하나씩 실행하기
```bash
T4="--parts tless_13 --parts tless_18 --parts tless_29 --parts tless_30"
P6="$T4 --parts mulsen_nut --parts mulsen_flat_pad"

# 점검 실험
bash scripts/start_worker.sh && isco exp run e0_2 --parts tless_29 --parts tless_13 && isco worker stop
isco exp run e0_3 $T4 --parts mulsen_nut --parts fusion_51876
isco exp run e0_4 --parts tless_13 --parts fusion_51876
isco exp run e0_5 $P6
isco exp run e0_6                                   # MVTec 3D-AD 필요
isco exp run result_sanity --parts fusion_22340 --parts fusion_41229 --parts fusion_51876

# 가설 검증 실험
isco exp run e1 $P6
isco exp run e2 $T4
isco exp run e3 $P6
isco exp run e4 --parts tless_13 --parts tless_29 --parts tless_30
isco exp run e5 --parts mulsen_nut --parts mulsen_flat_pad --parts mulsen_capsule   # 실제 결함 세트 먼저(7.5)

# 견고성 실험
isco exp run e5b                                    # MVTec 3D-AD 필요
isco exp run e6 --parts tless_13 --parts tless_29   # e2 다음에
isco exp run e7 --parts tless_13 --parts tless_29   # e2 다음에
isco data build tless_29 --preset mvp --yaw-mode random --set-id mvp_yawrand
isco exp run e7b --parts tless_29
isco exp run e8 --parts tless_13 --parts tless_29 --seeds 0,1
```
- `--workers`로 병렬 작업 수(기본 20)를, `--out`으로 결과 폴더를, `--budget`·`--seeds`로 평가 횟수와 반복
  횟수를 바꿀 수 있다(바꾸면 미리 고정한 값과 달라지므로 탐색적 결과로만 본다).
- `e6`, `e7`은 `e2`에서 평가한 설정 40개(`runs/exp/e2/evals_*.jsonl`)를 다른 검출기·잡음으로 다시
  평가하므로 `e2`가 끝난 뒤에 돌린다.

### 7.4 전체 대기열 한 번에
```bash
nohup bash scripts/run_experiments.sh > runs/logs/run_experiments.log 2>&1 &      # 처음부터
bash scripts/run_experiments.sh e1                                                 # e1부터 이어서
```
순서: 데이터 빌드 대기 → `e0_3` → `e0_4` → `e0_5` → 보정 확인(미검률이 목표 범위를 벗어나면 멈추고
결함 크기 재설정을 요구) → 데모 → 데모 자료 → 한 줄 점검 → `result_sanity` → `e0_2` → `e1` → `e2` →
`e3` → `e4` → 실제 결함 세트 → `e5` → `e6` → `e7` → `e7b` → `e8` → DINOv2 검출기 비교.
단계별 로그는 `runs/logs/exp_<단계>.log`. 병렬 작업 수는 `ISCO_WORKERS`(기본 22)로 바꾼다.

대기열이 끝난 뒤 이어서 돌리는 후속 단계:
```bash
nohup bash scripts/run_followups.sh > runs/logs/run_followups.log 2>&1 &
```
실제 결함 세트 재생성(정렬 수정판) → `e5` 재실행 → 잡음 없는(N0) 센서 추가 비교 → 교차 적용 확대
재실행(시드 3개 × 60회, `runs/exp/e1_full`). 앞의 대기열이 돌고 있으면 끝날 때까지 기다렸다 시작한다.

### 7.5 따로 실행하는 단계
```bash
# 실제 스캔 결함 세트 (e5 전에 한 번)
python -c "
from isco.eval.real_parts import build_mulsen_real_set
from isco.render.slot_cache import build_slot_cache
from isco.visibility.raycast import build_visibility
for c in ('nut', 'flat_pad', 'capsule'):
    build_mulsen_real_set(c, overwrite=True)
    build_slot_cache('mulsen_' + c, 'real', n_workers=6)
    build_visibility('mulsen_' + c, 'real')
"

# DINOv2 딥러닝 특징 검출기로 검출기 교체 실험
python -c "
from isco.eval.hypotheses import e6
e6(parts=['tless_13', 'tless_29'], out='runs/exp/e6_dino', detectors=('dinov2_bank',))
"
```

### 7.6 결과 정리
```bash
python scripts/summarize_experiments.py          # 모든 실험의 판정·핵심 수치 표 → runs/exp/SUMMARY.md
python scripts/analyze_e3_additions.py           # 카메라 1대 추가 쌍 분석 → runs/exp/e3/additions.json
python scripts/analyze_noise_levels.py           # 잡음 수준별 설정 값 → runs/exp/e7/noise_levels.json
python scripts/make_paper_figures.py             # 실험 결과와 무관한 개요 그림(시스템 구성, 설정 공간) → runs/paper_figures/
```

---

## 8. 데모 영상과 발표 자료 만들기

데모 영상(약 70초, 한국어 자막): 권장 설정의 카메라와 흐르는 부품, 결함 6종 근접, 촬영 순간의
RGB·깊이, 검출기의 점별 점수, 두 단계 탐색.
```bash
python scripts/demo_video_isaac.py              # Isaac Sim 장면 렌더 (헤드리스, 약 2분)
python scripts/make_demo_video.py scores        # 영상에 나오는 부품의 점별 이상 점수
python scripts/make_demo_video.py compose       # 자막 + 탐색 애니메이션 → runs/demo_video/isco_demo.mp4
python scripts/make_report_assets.py            # 정지 화면, 차트, 카메라 배치 체험 데이터
python scripts/make_demo_assets.py              # 기준 배치 비교 표·그림, 결함 예시, 커버리지 지도
```

중간 점검 자료(제품 6종 검사 시뮬레이션을 담은 약 3분 50초 영상, 41장 발표 자료, 그림)는
`scripts/midterm/`(브랜치 `isco/midterm-reports`)로 만든다. 결과는 `runs/midterm_reports/`에 쌓인다.
```bash
python scripts/midterm/scores.py                # 제품별 권장 설정의 점별 이상 점수
bash scripts/midterm/render_isaac.sh            # Isaac 장면(전경, 제품, 결함, 슬롯, 회전대, 제품별 촬영)
python scripts/midterm/figures.py && python scripts/midterm/assets.py && python scripts/midterm/search_data.py
bash scripts/midterm/make_video.sh              # 영상 합성 → 원본 / 고화질(CRF 22) / 압축(CRF 28)
python scripts/midterm/deck_data.py             # 발표 수치와 정지 화면
npm install --prefix runs/midterm_reports/deck pptxgenjs@3.12.0
NODE_PATH=runs/midterm_reports/deck/node_modules node scripts/midterm/deck.js runs/midterm_reports/deck/ISCO_중간발표.pptx
bash scripts/midterm/qa_deck.sh runs/midterm_reports/deck/ISCO_중간발표.pptx    # LibreOffice 렌더 확인(Docker)
```
`PPTX_SKILL_DIR`를 지정하면 발표 자료에 테마 색을 입힌다(`apply_theme.js`). `qa_deck.sh`는 파일을 검사한 뒤
Docker 이미지 `isco-office-qa`(LibreOffice)로 슬라이드를 그림으로 렌더한다.
git worktree처럼 다른 작업 폴더에서 돌릴 때는 `ISCO_DATA_ROOT`로 `data/` 위치를 알려 준다.

---

## 9. 설정 파일 (`configs/`)

| 폴더 | 내용 |
|---|---|
| `sensor/` | 구조광 3D 카메라 모델(`sl_default`: 1024², 수평 시야 20°) |
| `noise/` | 잡음 수준 `N0`–`N3`, `N2MS`(거리 잡음, 비스듬한 면 측정 실패, 튀는 점, 경계 필터, 노출 번짐) |
| `process/` | `inline_default`(속도 0.1/0.3/0.6 m/s, 부품 간격 0.3 m, 회전대 시간), `static_capture`(정지 촬영) |
| `grid/` | 카메라 후보 자리 96곳 정의 |
| `detector/` | `fpfh_bank`(기본), `cad_distance`(학습 없는 설계도 거리), `dinov2_bank`(DINOv2-S/14) |
| `defects/` | 결함 사전 `default_dictionary` |
| `variants/` | 변형 세트 `smoke` / `mvp`(학습 40 + 기준선 100 + 탐색용 36 + 성적용 36 + 대조군 6) / `full` |
| `search/` | 탐색 설정 `smoke` / `demo` / `full` |

## 10. 저장소 구조

```
isco/
  schemas/      설정·센서·공정·부품·워커 작업의 데이터 규약(pydantic)
  geometry/     좌표계, 핀홀 카메라와 슬롯 격자, 정답 표시 메시, 리메싱
  datagen/      부품 등록, 정상품 변이, 결함 생성기, 변형 세트, 센서 잡음
  render/       embree 렌더러, 가림 물체, 시점 저장본, 작업 대기열
  visibility/   슬롯별 결함 가시성, 1단계 지표
  detect/       뷰 조합·정합, FPFH 특징, 메모리 뱅크, 설계도 거리 검출기, 지표, 2단계 파이프라인
  optim/        탐색 공간, 비용 모델, 평가기(+캐시), 탐색 방법, 파레토, 실행 관리
  report/       결과 저장, 그림, 마크다운/HTML 리포트
  eval/         공개 데이터 부품(T-LESS, MulSen-AD, Real3D-AD), 점검·검증 실험, MVTec 3D-AD
  sim/          Isaac Sim: 앱, 장면, 부품, 카메라 리그, 촬영, 안착, 워커
  cli/          `isco` 명령
isaac_sim/      첫 시제품 스크립트(create_conveyor.py)와 그 출력
configs/        YAML 설정         scripts/   설치·다운로드·워커·빌드·실험 대기열·분석·데모
tests/unit/     단위 테스트(Isaac 불필요)      tests/isaac/   Isaac Sim 테스트
```

## 11. 좌표와 단위 규칙

- 월드 좌표: +Z 위, +X 벨트 진행 방향, 벨트 윗면 z = 0, 검사 지점은 원점 위.
- 부품은 기준 자세(무게중심이 가장 낮은 안정 자세, 긴 변을 +X로, 가장 낮은 점 z = 0)로 저장하고
  가장 긴 변을 40 mm로 맞춘다.
- 카메라는 OpenCV 축, 화소 중심 (u + 0.5, v + 0.5), 깊이는 영상면까지의 거리.
- 회전대는 검사 지점을 지나는 수직축으로 부품을 돌린다. 그래서 부품을 90° 돌려 찍은 영상은 방위를
  반대로 90° 옮긴 슬롯의 저장본과 같고, 회전 설정은 저장본을 그대로 다시 쓴다.

## 12. 테스트

```bash
pytest                                   # 단위 테스트, 약 30초, Isaac Sim 불필요
pytest -m isaac tests/isaac              # Isaac Sim 테스트(헤드리스)
```

## 13. 문제 해결

| 증상 | 해결 |
|---|---|
| `ModuleNotFoundError` 또는 Python 3.10 패키지가 섞임 | ROS 2가 `PYTHONPATH`를 내보낸 것. `export PYTHONPATH=` 후 다시 실행 |
| 워커가 시작되지 않음 | 같은 대기열에 살아 있는 워커가 있으면 거부한다. `isco worker status` → `isco worker stop` |
| 한 Isaac 프로세스에서 여러 부품을 렌더하다 죽음 | 부품 묶음마다 프로세스를 새로 띄운다(`scripts/midterm/render_isaac.sh`처럼) |
| 설정 하나 평가가 2분 넘게 걸림 | 다른 무거운 작업과 CPU를 나눠 쓰는 중. `--workers`를 줄이거나 다른 작업을 멈춤 |
| 패키지 설치 후 Isaac Sim이 깨짐 | `scripts/constraints-isaacsim.txt`의 버전(numpy, psutil, torch)으로 되돌림 |

## 14. 라이선스와 데이터 고지

코드는 Apache-2.0. 데이터셋은 각자의 라이선스를 따른다(4장 표). 이 저장소는 데이터를 재배포하지 않고
내려받기 스크립트만 제공한다. Fusion 360 Gallery 부품은 재배포 금지라 저장소에 넣지 않았다.
NVIDIA 컨베이어 에셋은 Isaac Sim 에셋 서버에서 불러오며 복사하지 않는다.
