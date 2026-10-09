# ISCO 프로토타입 — 진행 상황 (Isaac Sim 컨베이어 파트)

마지막 업데이트: 2026-09-18

## 지금까지 한 것

### 1. 컨베이어 벨트 (`create_conveyor.py`)
- NVIDIA 공식 에셋(Nucleus/CDN `Isaac/Props/Conveyors/ConveyorBelt_A06.usd`)을 참조해서 씬에 배치.
  - `--conveyor-asset`으로 47종 중 다른 것 선택 가능, `--list-assets`로 목록 확인.
  - **인터넷 연결 필요** (첫 로드 시 스트리밍, 이후 Kit이 로컬 캐시).
- 벨트 상판(`Belt` 프림)에 `PhysxSurfaceVelocityAPI`를 얹어서 실제로 물체를 밀어내는 컨베이어 동작 구현.
- GUI로 띄우면 자동 재생, 헤드리스로는 지정한 시간만큼 재생 후 USD 저장.

### 2. 부품 스폰 (연속 흐름)
- Autodesk Fusion 360 Gallery Segmentation Dataset(HuggingFace 미러 `maksimko123/fusion360_test_mesh`, CC-BY-NC-4.0)에서 소형 정밀부품처럼 생긴 STL 5종을 골라 `assets/parts/`에 저장.
- `PartSpawner` 클래스: 벨트 입구에 일정 간격(`--spawn-interval`)으로 랜덤 부품을 하나씩 올리고, 최대 치수(`--part-size`, 기본 4cm)로 정규화 + rigid body 물리를 씌워 벨트를 타고 흘러가게 함. 벨트 끝을 지나면 자동 삭제(누적 방지).
- `--video`로 mp4 녹화 가능 (기본 6초, 30fps). **`output/conveyor_belt.mp4`가 최신 결과물임.**

### 3. 다중 카메라 촬영 + Depth 재구성 (`--capture-views`)
- 부품 하나를 벨트 중간 지점까지 흘려보낸 뒤, 그 순간을 카메라 N대(기본 3대, `--views-count`)로 동시 촬영.
  - 각 카메라 RGB → `output/views/view_{i}.png`
  - 각 카메라 depth(포인트클라우드 애노테이터) → 부품 근처만 필터링해서 합침 → `merged_pointcloud.npy` / `.ply`
- 재구성된 포인트클라우드를 정답 3D 모델(변환된 STL)과 두 방향으로 비교:
  - **completeness**: 정답 표면 중 재구성이 닿은 비율 ("실제로 얼마나 봤는가")
  - **accuracy**: 재구성된 점이 정답 표면에 실제로 맞는 비율 ("재구성이 얼마나 정확한가")
  - 결과는 `output/views/summary.txt`, 시각화는 `output/views/coverage.png` (카메라별 포인트 + completeness 히트맵을 2개 각도에서).
- 마지막 테스트 결과(부품 `41229_16283ae1_0018`, 카메라 3대, elevation 30°, radius_factor 45, focal 24mm):
  - completeness 62.4% (평균 거리 1.52mm), accuracy 67.6% (평균 거리 1.50mm)

## 안 된 것 / 알려진 한계

1. **커버리지 지표가 아직 다듬어지는 중이었음** — 방금 막 "벨트 표면이 부품 점에 섞여 들어가는 필터링 버그"와 "정점 기준 비교라 실제로 잘 보여도 커버리지가 0%로 나오는 문제"를 고쳤음. 위 62.4%/67.6% 수치는 파라미터를 한 번 튜닝한 결과라, **다른 부품·다른 카메라 배치에서도 안정적인지는 아직 검증 안 됨**. 여러 부품/설정으로 반복 실행해서 수치가 합리적인 범위에서 나오는지 확인 필요.
2. **카메라 배치가 하드코딩된 3방향 대칭(120도씩)** — ISCO 본래 목표인 "카메라 수·위치를 탐색"은 아직 없음. 지금은 고정된 고각(elevation)·반지름으로 3대를 균등 배치할 뿐, 배치를 바꿔가며 비교하는 기능은 없음.
3. **결함(defect) 개념이 전혀 없음** — 지금 부품은 전부 "정상" 형상. ISCO 제안서의 핵심인 "결함 검출 성능"을 측정하려면 부품에 결함(눌림/균열/절단 등)을 합성으로 넣는 기능이 필요함 (제안서 M2 "결함 생성기").
4. **검출기(이상탐지 모델)가 없음** — 지금은 "포인트가 표면 근처에 있는가"만 봄. 제안서의 메모리 뱅크 기반 이상탐지(M3)는 미착수.
5. **설정 탐색기(M4)가 없음** — 카메라 수/위치/컨베이어 속도를 바꿔가며 자동으로 여러 설정을 평가하고 파레토 전선을 뽑는 최적화 루프는 미착수. 지금은 사람이 CLI 인자로 하나씩 바꿔서 실행해야 함.
6. **Real3D-AD / MVTec 3D-AD 연동 없음** — 제안서의 실물 앵커 데이터셋은 아직 스크립트에 연결 안 됨. 지금 쓰는 부품은 Fusion 360 Gallery(임의의 CAD 바디)이지 저 두 벤치마크가 아님.
7. **coverage.png의 표면 완전 매치가 아직 애매한 지점** — 정답 메시가 저해상도(면 500개 안팎)라 완전 평평한 넓은 면 vs 좁은 슬롯/홈 사이에서 completeness 차이가 실제 형상 특성 때문인지 메시 해상도 때문인지 구분이 아직 덜 됨. 필요하면 더 조밀한 원본 CAD(STEP)로 다시 메싱하는 게 나을 수 있음.
8. **GUI 모드 장시간 안정성 미검증** — 헤드리스는 여러 번 검증했지만, GUI로 몇 분 이상 계속 띄워놓고 부품이 수십~수백 개 스폰/디스폰되는 상황은 오래 테스트 안 해봄.

## 다음에 할 일 (우선순위 순)

1. **커버리지 지표 재검증**: 5개 부품 전부에 대해 `--capture-views`를 돌려서 completeness/accuracy가 부품 형상에 따라 합리적으로 달라지는지 확인 (예: 납작한 부품 vs 원통형 부품).
2. **카메라 배치를 파라미터화**: 대수뿐 아니라 elevation/azimuth 조합을 리스트로 받아서 여러 배치를 한 번에 비교하고, 배치별 completeness를 표로 뽑는 기능.
3. **결함 생성기 최소 버전**: STL 메시에 국소적으로 정점을 눌러 넣거나(sink mark 흉내) 잘라내는(cut) 간단한 변형 함수 하나부터 시작.
4. **설정 vs 커버리지 스윕**: 카메라 수(1~5대) × elevation(3~4단계)를 격자로 돌려서 "카메라를 하나 줄이면 completeness가 얼마나 떨어지는가" 곡선 뽑기 — 제안서 파레토 전선의 가장 단순한 버전.
5. Real3D-AD 실스캔 1개를 받아서 지금 파이프라인(로드→3카메라 촬영→재구성 비교)에 그대로 넣어보기 — 형식 변환(스캔 → 이 스크립트가 쓰는 정규화 STL 형태)이 관건.

## 실행 명령어 모음
```bash
conda activate env_isaaclab
cd ~/Capstone/isaac_sim

# GUI로 계속 보기 (부품이 계속 흘러감)
python create_conveyor.py

# 데모 영상 재생성
python create_conveyor.py --headless --video

# 3카메라 촬영 + depth 재구성 커버리지 계산
python create_conveyor.py --headless --capture-views --views-part 41229_16283ae1_0018.stl
```
