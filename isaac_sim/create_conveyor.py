"""
ISCO - 검사 공정 시뮬레이션용 컨베이어 벨트 생성 스크립트 (NVIDIA Isaac Sim)

무엇을 만드는가
----------------
검사 셀(inspection cell)의 물류 이동부인 컨베이어 벨트 하나를 USD 스테이지에
생성한다. 지오메트리는 NVIDIA 공식 에셋 서버(Nucleus/CDN)의
`Isaac/Props/Conveyors/` 컬렉션에서 참조해온다 (isaacsim.asset.gen.conveyor.ui
확장의 "Conveyor Track Builder"가 쓰는 것과 같은 에셋 — 롤러/볼트/패널까지
모델링된 실제 산업용 컨베이어 모양). 47종류(직선/커브/T분기/Y분기/경사로,
벨트·롤러·듀얼 스타일)가 있으며 `--conveyor-asset`로 고를 수 있다.
기본값은 가장 단순한 직선 벨트형인 ConveyorBelt_A06.

** 인터넷 연결이 필요함 ** (첫 실행 시 에셋을 스트리밍해오고, 이후엔 Kit이
로컬에 캐시해둔다). 완전 오프라인/절차적 생성이 필요하면 git 히스토리의
이전 버전(박스로 직접 조립한 버전)을 참고할 것.

에셋의 "Belt" 하위 프림은 이미 kinematic rigid body + 충돌까지 갖춰서 오므로,
여기에 PhysX의 "Surface Velocity" 기법만 얹는다: 프림 자체는 움직이지 않지만
접촉 콜백에서 표면 속도를 주입해 그 위에 놓인 물체가 실제로 밀려나가는 것처럼
시뮬레이션된다 (PhysxSchema.PhysxSurfaceVelocityAPI — omni.physx.demos의
ConveyorBeltDemo와 동일한 원리).

실행 방법 (env_isaaclab conda 환경에서)
--------------------------------------
    conda activate env_isaaclab

    # GUI를 띄워 눈으로 확인 (자동 재생됨, 창을 닫으면 저장 후 종료)
    python create_conveyor.py

    # 헤드리스로 5초 시뮬레이션 후 USD 저장하고 종료 (서버/무헤드 환경용)
    python create_conveyor.py --headless --run-seconds 5

    # 다른 종류의 컨베이어로 바꾸기 (예: 롤러형, 커브형)
    python create_conveyor.py --conveyor-asset ConveyorBelt_A02
    python create_conveyor.py --list-assets   # 사용 가능한 47종류 이름 출력

    # 헤드리스로 시뮬레이션을 재생하며 mp4 영상까지 저장
    python create_conveyor.py --headless --video --video-seconds 6

산출물
------
    isaac_sim/assets/conveyor_belt.usd    (기본 저장 경로, --save로 변경 가능)
    isaac_sim/output/conveyor_belt.mp4    (--video 지정 시, --video-out으로 변경 가능)

부품 스폰
--------
벨트 입구에 일정 간격으로 작은 부품을 계속 올려서, 실제 검사 라인처럼 새
부품이 끊임없이 들어와 벨트를 타고 흘러가는 느낌을 만든다. 부품 지오메트리는
Autodesk Fusion 360 Gallery Segmentation Dataset(CC-BY-NC-4.0, HuggingFace
미러 maksimko123/fusion360_test_mesh)에서 가져온 실제 CAD B-rep 바디 STL을
쓴다 — `isaac_sim/assets/parts/*.stl`. 각 부품은 소형 정밀부품 크기(기본
최대 치수 4cm)로 정규화해서 로드하고, convexHull 충돌 + rigid body를 씌워
벨트 위에서 실제로 밀려가게 한다. 벨트 끝을 지나면 자동으로 지운다(누적 방지).

다중 카메라 촬영 (--capture-views)
----------------------------------
ISCO의 핵심 아이디어인 "여러 각도의 검사 카메라"를 가장 단순하게 시연하는
모드. 벨트 위에 부품 하나를 올리고, 벨트 중간 지점(초점) 둘레에 카메라
N대(기본 3대)를 고른 각도로 배치한 뒤, 부품이 그 지점을 지나는 순간 모든
카메라에서 동시에 한 장씩 찍어 PNG로 저장한다.

    python create_conveyor.py --headless --capture-views
    python create_conveyor.py --headless --capture-views --views-count 4

다음 단계(이번 스크립트 범위 밖)
--------------------------------
    - 검사 카메라 배치/시점 계획 (ISCO의 view planning)
    - 결함이 있는 검사 대상물 변형(스크래치/찌그러짐 등)
    - 사이클 타임 측정을 위한 트리거 볼륨
"""

import argparse

parser = argparse.ArgumentParser(description="Isaac Sim 컨베이어 벨트 생성 스크립트")
parser.add_argument("--headless", action="store_true", help="헤드리스로 실행 (GUI 없이 저장만)")
parser.add_argument(
    "--conveyor-asset",
    type=str,
    default="ConveyorBelt_A06",
    help="Isaac/Props/Conveyors/ 아래 에셋 이름 (기본: 직선 벨트형 A06). --list-assets로 목록 확인",
)
parser.add_argument("--list-assets", action="store_true", help="사용 가능한 컨베이어 에셋 이름 47종을 출력하고 종료")
parser.add_argument("--speed", type=float, default=0.3, help="벨트 표면 속도(m/s)")
parser.add_argument("--no-parts", dest="spawn_parts", action="store_false", help="부품 스폰 없이 빈 벨트만 생성")
parser.add_argument("--spawn-interval", type=float, default=1.5, help="새 부품을 올리는 간격(초)")
parser.add_argument("--part-size", type=float, default=0.04, help="부품 하나의 최대 치수(m)")
parser.add_argument(
    "--parts-dir",
    type=str,
    default=None,
    help="부품 STL이 들어있는 폴더 (기본: isaac_sim/assets/parts/)",
)
parser.add_argument("--run-seconds", type=float, default=0.0, help="저장 전 시뮬레이션을 n초 재생(헤드리스 검증용)")
parser.add_argument("--save", type=str, default=None, help="저장할 USD 경로 (기본: isaac_sim/assets/conveyor_belt.usd)")
parser.add_argument("--video", action="store_true", help="시뮬레이션을 mp4 영상으로 녹화")
parser.add_argument("--video-out", type=str, default=None, help="저장할 mp4 경로 (기본: isaac_sim/output/conveyor_belt.mp4)")
parser.add_argument("--video-seconds", type=float, default=6.0, help="녹화 길이(초). --run-seconds가 더 크면 그 값을 따름")
parser.add_argument("--video-fps", type=int, default=30, help="영상 프레임레이트")
parser.add_argument("--video-width", type=int, default=1280, help="영상 가로 해상도")
parser.add_argument("--video-height", type=int, default=720, help="영상 세로 해상도")
parser.add_argument("--camera-distance", type=float, default=1.6, help="카메라를 컨베이어 크기 대비 얼마나 멀리 둘지 배율")
parser.add_argument(
    "--camera-focal-length",
    type=float,
    default=1.8,
    help="카메라 초점거리(mm). 이 Camera 센서의 기본 조리개가 매우 작아(~2.1mm) "
    "일반 카메라와 스케일이 다름 - 1.8 정도가 대략 60도 화각(광각)에 해당함",
)
parser.add_argument("--capture-views", action="store_true", help="부품 하나를 여러 각도 카메라로 동시 촬영하고 종료")
parser.add_argument("--views-count", type=int, default=3, help="배치할 카메라 대수")
parser.add_argument("--views-elevation", type=float, default=30.0, help="카메라 고각(도) - 클수록 위에서 내려다봄")
parser.add_argument(
    "--views-radius-factor",
    type=float,
    default=45.0,
    help="부품 크기 대비 카메라 거리 배율. 컨베이어 프레임(다리·지지대)에 "
    "카메라가 파묻히지 않으려면 부품 크기(4cm)보다 훨씬 멀리(수십 배) 둬야 함",
)
parser.add_argument("--views-focal-length", type=float, default=24.0, help="촬영 카메라 초점거리(mm) - 멀리서 확대해서 찍음")
parser.add_argument("--views-width", type=int, default=1024, help="촬영 이미지 가로 해상도")
parser.add_argument("--views-height", type=int, default=1024, help="촬영 이미지 세로 해상도")
parser.add_argument("--views-part", type=str, default=None, help="촬영할 특정 부품 STL 파일명 (기본: 랜덤 선택)")
parser.add_argument(
    "--views-out-dir", type=str, default=None, help="촬영 이미지 저장 폴더 (기본: isaac_sim/output/views/)"
)
args = parser.parse_args()

if args.list_assets:
    import json
    import os as _os

    _here = _os.path.dirname(_os.path.abspath(__file__))
    # 로컬 pip 설치본에 번들된 트랙 목록 정의 파일 (isaacsim.asset.gen.conveyor.ui 확장)
    _candidates = [
        _os.path.expanduser(
            "~/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/exts/"
            "isaacsim.asset.gen.conveyor.ui/data/track_types.json"
        )
    ]
    for _p in _candidates:
        if _os.path.isfile(_p):
            with open(_p) as f:
                _assets = json.load(f)["assets"]
            for _name, _cfg in sorted(_assets.items()):
                print(f"{_name:20s} style={_cfg.get('style'):6s} type={_cfg.get('type')}")
            raise SystemExit(0)
    print("track_types.json을 찾지 못함 (Isaac Sim 설치 경로가 다를 수 있음)")
    raise SystemExit(1)

from isaacsim import SimulationApp

# create_new_stage=False: 기본값(True)으로 두면 GUI 모드에서 Kit이 앱 시작
# 직후 빈 스테이지를 한 번 더 비동기로 열어버려서, 우리가 만든 씬이 다 지어진
# 뒤에 그 스테이지가 통째로 무효화되는 레이스가 생김 (Isaac Lab의
# AppLauncher도 같은 이유로 이 옵션을 끔). World()가 알아서 스테이지를
# 만들도록 우리가 직접 끈다.
simulation_app = SimulationApp({"headless": args.headless, "create_new_stage": False})

# 아래 import들은 SimulationApp 생성 이후에만 가능함 (Kit 확장이 이때 로드됨)
import glob
import os
import random

import numpy as np
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

from isaacsim.core.api import World
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.utils.bounds import compute_aabb, create_bbox_cache

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SAVE_PATH = os.path.join(SCRIPT_DIR, "assets", "conveyor_belt.usd")
DEFAULT_VIDEO_PATH = os.path.join(SCRIPT_DIR, "output", "conveyor_belt.mp4")
DEFAULT_PARTS_DIR = os.path.join(SCRIPT_DIR, "assets", "parts")
DEFAULT_VIEWS_DIR = os.path.join(SCRIPT_DIR, "output", "views")

# 부품마다 눈으로 구분하기 쉽게 고정 색상을 하나씩 배정 (파일명 기준)
PART_COLORS = [
    (0.75, 0.55, 0.15),  # 황동색
    (0.55, 0.15, 0.15),  # 벽돌색
    (0.20, 0.45, 0.65),  # 청록
    (0.55, 0.35, 0.60),  # 자주
    (0.35, 0.60, 0.30),  # 녹색
    (0.70, 0.70, 0.70),  # 회색(강철)
]


def look_at_quat_wxyz(eye, target, world_up=(0.0, 0.0, 1.0)):
    """eye 위치에서 target을 바라보는 카메라 orientation을 scalar-first 쿼터니언
    (w, x, y, z) numpy 배열로 반환한다. (USD 카메라 컨벤션: 로컬 -Z를 바라봄, +Y가 위)
    """
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(world_up, dtype=float)

    z_axis = eye - target
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    # 행 = 카메라 로컬 축의 월드 좌표 (pxr.Gf는 행벡터(v*M) 컨벤션을 쓰므로
    # 열이 아니라 행에 축을 채워야 ExtractRotation() 결과가 맞음)
    rot = np.stack([x_axis, y_axis, z_axis], axis=0)
    quat = Gf.Matrix3d(
        rot[0, 0], rot[0, 1], rot[0, 2],
        rot[1, 0], rot[1, 1], rot[1, 2],
        rot[2, 0], rot[2, 1], rot[2, 2],
    ).ExtractRotation().GetQuat()
    imag = quat.GetImaginary()
    return np.array([quat.GetReal(), imag[0], imag[1], imag[2]])


def apply_physics_material(stage, prim_path, static_friction=0.7, dynamic_friction=0.6, restitution=0.0):
    """정지/동적 마찰 계수를 지정한 물리 재질을 만들어 prim에 바인딩한다.

    surface velocity 기법은 접촉 마찰을 통해 물체를 밀어내므로, 마찰이 0에
    가까우면 물체가 벨트 위에서 미끄러지기만 하고 실제로는 움직이지 않는다.
    """
    mat_path = f"{prim_path}/PhysicsMaterial"
    material = UsdShade.Material.Define(stage, mat_path)
    physics_mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_mat.CreateStaticFrictionAttr().Set(static_friction)
    physics_mat.CreateDynamicFrictionAttr().Set(dynamic_friction)
    physics_mat.CreateRestitutionAttr().Set(restitution)

    prim = stage.GetPrimAtPath(prim_path)
    binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
    binding_api.Bind(material, bindingStrength=UsdShade.Tokens.weakerThanDescendants, materialPurpose="physics")
    return material


def find_part_stl_files(parts_dir):
    files = sorted(glob.glob(os.path.join(parts_dir, "*.stl")))
    if not files:
        raise RuntimeError(
            f"{parts_dir}에 부품 STL이 없음 - Fusion 360 Gallery 데이터셋에서 받아둔 "
            "isaac_sim/assets/parts/*.stl 파일이 필요함"
        )
    return files


def load_and_scale_mesh(stl_path, target_max_dim):
    """STL을 읽어 자기 중심을 원점으로 맞추고 최대 변 길이가 target_max_dim(m)이
    되도록 스케일한 (verts, faces)를 반환한다. load_part_mesh(USD 생성)와
    포인트클라우드 커버리지 비교(정답 메시)에서 공유해서 쓴다 - 둘 다 "부품이
    실제로 어느 위치/스케일에 있는가"에 대해 같은 답을 써야 하기 때문.
    """
    import trimesh

    mesh = trimesh.load(stl_path, force="mesh")
    verts = np.array(mesh.vertices, dtype=float)
    bbox_min, bbox_max = verts.min(axis=0), verts.max(axis=0)
    verts -= (bbox_min + bbox_max) / 2.0  # 자기 중심을 원점으로
    extent = float((bbox_max - bbox_min).max())
    scale = target_max_dim / extent if extent > 0 else 1.0
    verts *= scale
    faces = np.array(mesh.faces, dtype=int)
    return verts, faces


def load_part_mesh(stage, prim_path, stl_path, target_max_dim, color):
    """Fusion 360 Gallery의 STL 바디 하나를 UsdGeom.Mesh + rigid body로 로드한다.

    STL 데이터셋 원본 좌표는 임의 스케일이라 신뢰하지 않고, 매번 자기 bbox의
    최대 변 길이를 target_max_dim(m)에 맞춰 다시 스케일한다 (그래야 어떤
    바디를 골라도 "작은 정밀부품" 크기로 일관되게 보임). convexHull 충돌을
    쓰는 이유: PhysX는 오목한(non-convex) 동적 메시의 충돌을 직접 지원하지
    않기 때문.
    """
    verts, faces = load_and_scale_mesh(stl_path, target_max_dim)

    usd_mesh = UsdGeom.Mesh.Define(stage, prim_path)
    usd_mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in verts])
    usd_mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    usd_mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
    usd_mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    usd_mesh.CreateSubdivisionSchemeAttr("none")

    prim = usd_mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_collision.CreateApproximationAttr("convexHull")
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(0.02)
    apply_physics_material(stage, prim_path, static_friction=0.8, dynamic_friction=0.7)
    return prim_path


class PartSpawner:
    """벨트 입구에 주기적으로 부품을 하나씩 올리고, 벨트 끝을 지난 부품은
    지워서(despawn) 씬이 무한히 무거워지는 것을 막는다.

    prim의 현재 월드 위치는 SingleRigidPrim의 텐서 기반 get_world_pose()
    대신 순수 USD 쿼리(ComputeLocalToWorldTransform)로 읽는다 - 텐서
    물리 뷰는 rigid body 개수가 바뀔 때(=매번 스폰/디스폰할 때) 캐시가
    깨져서 "Simulation view object is invalidated" 예외가 나기 때문.
    """

    def __init__(self, stage, part_files, entry_pos, exit_x, part_size, spawn_interval, physics_fps):
        self.stage = stage
        self.part_files = part_files
        self.entry_pos = np.asarray(entry_pos, dtype=float)
        self.exit_x = exit_x
        self.part_size = part_size
        self.spawn_every_steps = max(1, int(round(spawn_interval * physics_fps)))
        self.step_count = 0
        self.spawn_count = 0
        self.active = []  # [prim_path, ...]

    def _spawn_one(self):
        stl_path = random.choice(self.part_files)
        color = PART_COLORS[self.spawn_count % len(PART_COLORS)]
        prim_path = f"/World/Parts/Part_{self.spawn_count:04d}"
        self.spawn_count += 1

        jitter_y = random.uniform(-0.6, 0.6) * 0.15  # 벨트 폭 내에서 살짝 좌우로
        pos = self.entry_pos + np.array([0.0, jitter_y, 0.0])
        load_part_mesh(self.stage, prim_path, stl_path, self.part_size, color)

        prim = self.stage.GetPrimAtPath(prim_path)
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(*pos.tolist()))
        yaw = random.uniform(0, 360)
        w, x, y, z = _euler_z_to_quat_wxyz(yaw)
        xformable.AddOrientOp().Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))

        self.active.append(prim_path)

    def step(self):
        """매 물리 스텝마다 호출. 스폰 주기가 되면 새 부품을 올리고, 벨트
        끝을 지난 기존 부품은 지운다."""
        self.step_count += 1
        if self.step_count % self.spawn_every_steps == 0:
            self._spawn_one()

        still_active = []
        for prim_path in self.active:
            prim = self.stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            world_xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            translation = world_xf.ExtractTranslation()
            if translation[0] > self.exit_x or translation[2] < -0.5:
                self.stage.RemovePrim(prim_path)
            else:
                still_active.append(prim_path)
        self.active = still_active


def _euler_z_to_quat_wxyz(yaw_degrees):
    half = np.radians(yaw_degrees) / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


def add_official_conveyor(stage, prim_path, asset_name, speed):
    """NVIDIA 공식 컨베이어 에셋을 Nucleus/CDN에서 참조해와 씬에 배치한다.

    반환값의 "belt_top_z"/"aabb"는 이후 카메라·테스트 큐브 배치에 쓰인다
    (에셋마다 실제 치수/높이가 다르므로 하드코딩하지 않고 로드 후 직접 잰다).
    """
    from isaacsim.storage.native import nucleus

    root = nucleus.get_assets_root_path()
    if not root:
        raise RuntimeError(
            "NVIDIA 에셋 서버(Nucleus)에 연결하지 못함 - 인터넷 연결을 확인할 것. "
            "(이 스크립트는 공식 컨베이어 에셋을 온라인에서 참조해온다)"
        )
    asset_path = f"{root}/Isaac/Props/Conveyors/{asset_name}.usd"

    conveyor_xform = stage.DefinePrim(prim_path, "Xform")
    if not conveyor_xform.GetReferences().AddReference(asset_path):
        raise RuntimeError(f"에셋을 불러오지 못함: {asset_path}")

    belt_path = f"{prim_path}/Belt"
    belt_prim = stage.GetPrimAtPath(belt_path)
    if not belt_prim.IsValid():
        raise RuntimeError(
            f"{asset_name}에 'Belt' 프림이 없음 - 이 스크립트는 conveyor_nodes에 "
            "'Belt'가 있는(스타일 BELT/DUAL) 에셋만 지원함. --list-assets로 다른 걸 골라볼 것."
        )

    # 이미 kinematic rigid body + collision은 에셋에 포함돼 있음. surface
    # velocity와 마찰 재질만 추가로 얹는다. (실측 결과 이 에셋 계열은 로컬
    # +X가 벨트 진행 방향)
    surface_vel = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(belt_prim)
    surface_vel.CreateSurfaceVelocityEnabledAttr().Set(True)
    surface_vel.CreateSurfaceVelocityLocalSpaceAttr().Set(True)
    surface_vel.CreateSurfaceVelocityAttr().Set(Gf.Vec3f(speed, 0.0, 0.0))
    apply_physics_material(stage, belt_path, static_friction=0.8, dynamic_friction=0.7)

    return prim_path, belt_path


def measure_conveyor(prim_path, belt_path):
    """로드된 컨베이어의 전체 바운딩박스와 벨트 상판 높이를 실측한다."""
    cache = create_bbox_cache()
    whole_aabb = compute_aabb(cache, prim_path)  # [xmin,ymin,zmin,xmax,ymax,zmax]
    belt_aabb = compute_aabb(cache, belt_path)
    return {
        "whole_aabb": whole_aabb,
        "belt_aabb": belt_aabb,
        "belt_top_z": belt_aabb[5],
        "center": np.array(
            [
                (whole_aabb[0] + whole_aabb[3]) / 2.0,
                (whole_aabb[1] + whole_aabb[4]) / 2.0,
                (whole_aabb[2] + whole_aabb[5]) / 2.0,
            ]
        ),
        "size": np.array(
            [
                whole_aabb[3] - whole_aabb[0],
                whole_aabb[4] - whole_aabb[1],
                whole_aabb[5] - whole_aabb[2],
            ]
        ),
    }


def add_lights(stage):
    """돔 라이트 + 보조 디스턴트 라이트로 씬을 밝힌다 (렌더/영상 품질용)."""
    from pxr import UsdGeom, UsdLux

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
    dome.CreateIntensityAttr(900.0)

    key_light = UsdLux.DistantLight.Define(stage, "/World/Lights/KeyLight")
    key_light.CreateIntensityAttr(3000.0)
    key_light.CreateAngleAttr(1.0)
    UsdGeom.Xformable(key_light).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 30.0, 0.0))


def add_conveyor_camera(center, size, resolution, distance_factor=1.6, focal_length=1.8, span=None):
    """컨베이어 전체가 잘 보이는 3/4 부감 시점 카메라를 생성한다.

    center/size는 measure_conveyor()로 실측한 바운딩박스 기준이라 어떤
    컨베이어 에셋(직선/커브/경사로)을 골라도 자동으로 프레임에 맞는다.
    distance_factor를 키우면 더 멀리서, focal_length를 줄이면 더 넓은 화각으로
    잡는다. span을 직접 지정하면 자동 계산(전체가 다 보이게) 대신 그 값을
    기준으로 확대/축소한다 - 부품처럼 작은 물체를 크게 보이게 하고 싶을 때 씀
    (부품은 벨트 길이의 1/50 수준이라 전체를 다 담으면 점으로만 보임).
    """
    from isaacsim.sensors.camera import Camera

    if span is None:
        span = float(np.max(size[:2]))  # 수평 방향 크기 기준
    target = center
    offset = np.array([span * 0.9, -span * 1.05, span * 0.75]) * distance_factor
    eye = target + offset
    orientation = look_at_quat_wxyz(eye, target)

    camera = Camera(
        prim_path="/World/Camera",
        name="conveyor_camera",
        resolution=resolution,
    )
    camera.initialize()
    # camera_axes="usd": look_at_quat_wxyz가 만든 쿼터니언은 USD 카메라 컨벤션
    # (로컬 -Z를 바라보고 +Y가 위) 기준임. Camera의 기본 world 컨벤션(+X 전방,
    # +Z 위)과 다르므로 반드시 명시해야 함.
    camera.set_world_pose(position=eye, orientation=orientation, camera_axes="usd")
    camera.set_focal_length(focal_length)
    return camera


def add_inspection_cameras(center, num_cameras, part_size, radius_factor, elevation_deg, resolution, focal_length=24.0):
    """center(부품 위치) 둘레에 카메라 num_cameras대를 방위각으로 고르게
    돌려가며 배치한다 (ISCO의 다중 시점 검사 카메라 배치를 가장 단순화한 형태).

    모든 카메라가 같은 고각(elevation_deg)에서 center를 내려다보고, 방위각만
    360/num_cameras씩 차이나게 돈다. 반지름은 part_size * radius_factor로
    정해서 부품 크기에 맞게 자동으로 가까워지거나 멀어진다.
    """
    from isaacsim.sensors.camera import Camera

    radius = part_size * radius_factor
    elev = np.radians(elevation_deg)
    cameras = []
    for i in range(num_cameras):
        az = np.radians(360.0 * i / num_cameras)
        offset = radius * np.array(
            [np.cos(elev) * np.cos(az), np.cos(elev) * np.sin(az), np.sin(elev)]
        )
        eye = center + offset
        orientation = look_at_quat_wxyz(eye, center)

        cam_path = f"/World/InspectionCameras/Cam_{i}"
        camera = Camera(prim_path=cam_path, name=f"inspection_cam_{i}", resolution=resolution)
        camera.initialize()
        camera.set_world_pose(position=eye, orientation=orientation, camera_axes="usd")
        camera.set_focal_length(focal_length)
        camera.add_pointcloud_to_frame()  # depth 기반 3D 포인트클라우드 (world frame)
        cameras.append(camera)
    return cameras


def compare_reconstruction(gt_verts_world, gt_faces, merged_points, threshold, num_samples=8000):
    """3대 카메라의 depth를 합쳐 만든 포인트클라우드("재구성 3D 모형")를 실제
    정답 3D 모형(정답 메시)과 두 방향으로 비교한다 - 표준 3D 재구성 평가
    지표(완전성/정확도, Chamfer distance 계열)와 같은 구조:

      completeness(정답→재구성): 정답 표면을 촘촘히 샘플링한 각 점에서 재구성
          포인트클라우드까지의 최단거리. "실제 표면 중 얼마나 놓쳤는가".
          다 놓친 곳(먼 거리)이 있으면 "이 부분은 3대 중 어느 카메라에도
          안 보였다"는 뜻 - 카메라 배치의 사각지대.
      accuracy(재구성→정답): 재구성된 각 점에서 정답 표면까지의 최단거리.
          "재구성한 점이 실제 표면에서 얼마나 벗어났는가" - 노이즈 없는
          합성 depth라서 이상적으로는 threshold 이내에 다 들어와야 함
          (많이 벗어나면 카메라 파라미터/좌표 변환이 잘못됐다는 신호).

    정점(vertex) 기준이 아니라 표면적 비례 샘플을 쓰는 이유: 저해상도
    메시(면 500개 안팎)의 정점은 대부분 모서리/꼭짓점이라 depth 픽셀이
    정확히 그 자리에 찍힐 확률이 낮아, 실제로는 잘 보인 면인데도
    "커버리지 0"으로 나오는 착시가 생기기 때문.
    """
    import trimesh
    from scipy.spatial import cKDTree

    mesh = trimesh.Trimesh(vertices=gt_verts_world, faces=gt_faces, process=False)
    gt_surface_points, _ = trimesh.sample.sample_surface(mesh, num_samples)

    result = {
        "gt_surface_points": gt_surface_points,
        "completeness": 0.0,
        "completeness_dists": np.full(len(gt_surface_points), np.inf),
        "accuracy": 0.0,
        "accuracy_dists": np.full(len(merged_points), np.inf),
        "mean_completeness_dist": float("inf"),
        "mean_accuracy_dist": float("inf"),
    }
    if len(merged_points) == 0:
        return result

    recon_tree = cKDTree(merged_points)
    completeness_dists, _ = recon_tree.query(gt_surface_points)
    result["completeness_dists"] = completeness_dists
    result["completeness"] = float(np.mean(completeness_dists < threshold))
    result["mean_completeness_dist"] = float(np.mean(completeness_dists))

    gt_tree = cKDTree(gt_surface_points)
    accuracy_dists, _ = gt_tree.query(merged_points)
    result["accuracy_dists"] = accuracy_dists
    result["accuracy"] = float(np.mean(accuracy_dists < threshold))
    result["mean_accuracy_dist"] = float(np.mean(accuracy_dists))

    return result


def visualize_coverage(gt_verts_world, gt_faces, per_camera_points, comparison, threshold, out_path):
    """3패널 시각화:
      1) 정답 메시(연회색) + 카메라별 재구성 포인트(색상 구분) - 어느
         카메라가 어디를 봤는지
      2) 정답 표면 샘플을 completeness 거리로 색칠한 히트맵(초록=재구성이
         가까이 있음=봤다, 빨강=멀다=3대 중 아무도 못 본 사각지대) - "실제
         모형과 비교했을 때 어디가 비었는지"를 바로 보여줌
      3) 1)과 2)를 위에서 내려다본 각도로 반복 (반대편 사각지대 확인용)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # 한글 라벨이 깨지지 않도록 시스템에 있는 CJK 폰트를 지정 (없으면 기본 폰트로
    # 조용히 넘어감 - 라벨이 네모로 보이는 정도는 치명적이지 않음)
    for _font in ["Noto Sans CJK KR", "NanumGothic", "Malgun Gothic"]:
        if _font in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
            matplotlib.rcParams["font.family"] = _font
            break

    mins, maxs = gt_verts_world.min(axis=0), gt_verts_world.max(axis=0)
    pad = float((maxs - mins).max()) * 0.6 + 1e-4
    center = (mins + maxs) / 2.0
    cam_colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]
    views = [(20, -60), (75, -90)]

    fig = plt.figure(figsize=(15, 5.5 * len(views)))
    for row, (elev, azim) in enumerate(views):
        # 패널 1: 카메라별 재구성 포인트
        ax1 = fig.add_subplot(len(views), 2, row * 2 + 1, projection="3d")
        ax1.add_collection3d(
            Poly3DCollection(gt_verts_world[gt_faces], alpha=0.15, facecolor="gray", edgecolor="gray", linewidths=0.1)
        )
        for i, pts in enumerate(per_camera_points):
            if len(pts) == 0:
                continue
            ax1.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                s=2, alpha=0.5, color=cam_colors[i % len(cam_colors)], label=f"Cam {i}",
            )
        if row == 0:
            ax1.legend(loc="upper left", fontsize=8, markerscale=3)
            ax1.set_title("카메라별 재구성 포인트", fontsize=10)

        # 패널 2: 정답 표면의 completeness 히트맵 (초록=봄, 빨강=사각지대)
        ax2 = fig.add_subplot(len(views), 2, row * 2 + 2, projection="3d")
        gt_pts = comparison["gt_surface_points"]
        dists = np.clip(comparison["completeness_dists"], 0, threshold * 4)
        sc = ax2.scatter(
            gt_pts[:, 0], gt_pts[:, 1], gt_pts[:, 2],
            c=dists, cmap="RdYlGn_r", s=4, vmin=0, vmax=threshold * 4,
        )
        if row == 0:
            ax2.set_title("정답 표면 vs 재구성 거리 (초록=봄 / 빨강=사각지대)", fontsize=10)
            cbar = fig.colorbar(sc, ax=ax2, shrink=0.6, pad=0.02)
            cbar.set_label("정답 표면 → 가장 가까운 재구성 점까지 거리 (m)", fontsize=8)

        for ax in (ax1, ax2):
            ax.set_xlim(center[0] - pad, center[0] + pad)
            ax.set_ylim(center[1] - pad, center[1] + pad)
            ax.set_zlim(center[2] - pad, center[2] + pad)
            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def run_capture_views(world, stage, info, part_files, args):
    """부품 하나를 벨트에 올려서 초점 지점까지 흘려보낸 뒤, 그 순간을 여러
    각도의 카메라로 동시에 촬영한다. 각 카메라의 RGB 이미지를 PNG로 저장하고,
    depth(포인트클라우드 애노테이터)로 부품 표면을 재구성해서 정답 메시와
    비교한 커버리지를 계산·시각화한다. 저장된 파일 경로 목록을 반환.
    """
    import imageio

    whole = info["whole_aabb"]
    entry_x = whole[0] + 0.15
    focus_x = whole[0] + (whole[3] - whole[0]) * 0.4
    entry_pos = np.array([entry_x, 0.0, info["belt_top_z"] + args.part_size / 2.0 + 0.02])

    stl_path = (
        os.path.join(args.parts_dir or DEFAULT_PARTS_DIR, args.views_part)
        if args.views_part
        else random.choice(part_files)
    )
    local_verts, local_faces = load_and_scale_mesh(stl_path, args.part_size)

    part_path = "/World/Parts/InspectedPart"
    load_part_mesh(stage, part_path, stl_path, args.part_size, PART_COLORS[0])
    prim = stage.GetPrimAtPath(part_path)
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*entry_pos.tolist()))
    xformable.AddOrientOp().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

    physics_fps = round(1.0 / world.get_physics_dt())
    travel_time = max(0.1, (focus_x - entry_x) / max(args.speed, 1e-3))
    travel_steps = int(round(travel_time * physics_fps))

    world.reset()
    for _ in range(travel_steps):
        world.step(render=False)

    # 부품이 벨트 위에서 실제로 자리잡은(약간 구르거나 튄) 최종 포즈를 읽어서
    # 정답 메시를 월드 좌표로 옮긴다 - 카메라가 보는 것과 같은 기준이어야
    # 커버리지 비교가 의미 있음.
    final_xf = UsdGeom.Xformable(stage.GetPrimAtPath(part_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    gt_verts_world = np.array([np.array(final_xf.Transform(Gf.Vec3d(*v.tolist()))) for v in local_verts])
    part_center = gt_verts_world.mean(axis=0)

    cameras = add_inspection_cameras(
        center=part_center,
        num_cameras=args.views_count,
        part_size=args.part_size,
        radius_factor=args.views_radius_factor,
        elevation_deg=args.views_elevation,
        resolution=(args.views_width, args.views_height),
        focal_length=args.views_focal_length,
    )

    for _ in range(10):  # 렌더 파이프라인 워밍업
        world.step(render=True)

    out_dir = args.views_out_dir or DEFAULT_VIEWS_DIR
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    per_camera_points = []
    # 배경(벨트)까지 다 잡히는 포인트클라우드에서 부품 근처만 남기는 필터.
    # 구 형태(중심까지의 거리)로 자르면 안 됨 - 벨트가 평평해서 part_center와
    # 거의 같은 높이의 벨트 면이 반지름 안에서 원반 모양으로 통째로 걸러지지
    # 않고 남아버림 (실제로 이 문제로 처음엔 벨트 표면이 다 섞여 나왔음).
    # 대신 부품의 실제 bbox를 약간의 여유만 두고 그대로 쓰는 axis-aligned
    # 박스 필터를 쓴다 - 벨트 길이/폭 방향으로는 부품 바로 옆만 남고,
    # 벨트가 넓게 퍼진 나머지 영역은 다 잘려나간다.
    pad = args.part_size * 0.12
    bbox_min = gt_verts_world.min(axis=0) - pad
    bbox_max = gt_verts_world.max(axis=0) + pad

    for i, camera in enumerate(cameras):
        frame = camera.get_rgba()
        retries = 0
        while (frame is None or frame.size == 0) and retries < 30:
            world.step(render=True)
            frame = camera.get_rgba()
            retries += 1
        if frame is None or frame.size == 0:
            print(f"[create_conveyor] warning: camera {i} RGB 프레임을 못 얻음")
            per_camera_points.append(np.empty((0, 3)))
            continue
        out_path = os.path.join(out_dir, f"view_{i}.png")
        imageio.imwrite(out_path, np.asarray(frame[:, :, :3], dtype=np.uint8))
        saved.append(out_path)

        pc = np.asarray(camera.get_pointcloud(world_frame=True))
        if pc.size > 0:
            inside = np.all((pc >= bbox_min) & (pc <= bbox_max), axis=1)
            pc = pc[inside]
        per_camera_points.append(pc)
        print(f"[create_conveyor] camera {i}: {len(pc)} points near part")

    merged_points = (
        np.concatenate([p for p in per_camera_points if len(p) > 0], axis=0)
        if any(len(p) > 0 for p in per_camera_points)
        else np.empty((0, 3))
    )

    threshold = args.part_size * 0.05  # 부품 크기의 5% 이내면 "일치"로 침
    comparison = compare_reconstruction(gt_verts_world, local_faces, merged_points, threshold)
    print(
        f"[create_conveyor] 재구성 포인트 합계: {len(merged_points)}개 "
        f"(카메라별: {[len(p) for p in per_camera_points]})\n"
        f"[create_conveyor] completeness(정답 표면 중 재구성된 비율): "
        f"{comparison['completeness']*100:.1f}% (평균 거리 {comparison['mean_completeness_dist']*1000:.2f}mm)\n"
        f"[create_conveyor] accuracy(재구성된 점이 정답 표면에 맞는 비율): "
        f"{comparison['accuracy']*100:.1f}% (평균 거리 {comparison['mean_accuracy_dist']*1000:.2f}mm)"
    )

    pc_path = os.path.join(out_dir, "merged_pointcloud.npy")
    np.save(pc_path, merged_points)
    saved.append(pc_path)
    try:
        import trimesh

        ply_path = os.path.join(out_dir, "merged_pointcloud.ply")
        trimesh.points.PointCloud(merged_points).export(ply_path)
        saved.append(ply_path)

        gt_ply_path = os.path.join(out_dir, "ground_truth_mesh.ply")
        trimesh.Trimesh(vertices=gt_verts_world, faces=local_faces, process=False).export(gt_ply_path)
        saved.append(gt_ply_path)
    except Exception as e:
        print(f"[create_conveyor] warning: ply 저장 실패 ({e})")

    coverage_img_path = os.path.join(out_dir, "coverage.png")
    try:
        visualize_coverage(gt_verts_world, local_faces, per_camera_points, comparison, threshold, coverage_img_path)
        saved.append(coverage_img_path)
    except Exception as e:
        print(f"[create_conveyor] warning: 커버리지 시각화 실패 ({e})")

    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"part: {os.path.basename(stl_path)}\n")
        f.write(f"num_cameras: {len(cameras)}\n")
        f.write(f"points_per_camera: {[len(p) for p in per_camera_points]}\n")
        f.write(f"merged_points: {len(merged_points)}\n")
        f.write(f"match_threshold_m: {threshold}\n")
        f.write(f"completeness_ratio: {comparison['completeness']}\n")
        f.write(f"completeness_mean_dist_m: {comparison['mean_completeness_dist']}\n")
        f.write(f"accuracy_ratio: {comparison['accuracy']}\n")
        f.write(f"accuracy_mean_dist_m: {comparison['mean_accuracy_dist']}\n")
    saved.append(summary_path)

    return saved


def main():
    world = World(stage_units_in_meters=1.0)
    stage = world.stage

    GroundPlane(prim_path="/World/GroundPlane", z_position=0.0, size=10.0)
    add_lights(stage)

    conveyor_path, belt_path = add_official_conveyor(
        stage,
        prim_path="/World/Conveyor",
        asset_name=args.conveyor_asset,
        speed=args.speed,
    )

    world.reset()
    for _ in range(5):  # 에셋 레퍼런스가 완전히 컴포즈되도록 워밍업
        world.step(render=False)

    info = measure_conveyor(conveyor_path, belt_path)
    physics_fps = round(1.0 / world.get_physics_dt())

    if args.capture_views:
        parts_dir = args.parts_dir or DEFAULT_PARTS_DIR
        part_files = find_part_stl_files(parts_dir)
        saved = run_capture_views(world, stage, info, part_files, args)
        for p in saved:
            print(f"[create_conveyor] saved view: {p}")

        import omni.usd

        save_path = args.save or DEFAULT_SAVE_PATH
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        omni.usd.get_context().save_as_stage(save_path)
        print(f"[create_conveyor] saved stage to: {save_path}")
        simulation_app.close()
        return

    spawner = None
    if args.spawn_parts:
        whole = info["whole_aabb"]
        parts_dir = args.parts_dir or DEFAULT_PARTS_DIR
        part_files = find_part_stl_files(parts_dir)
        entry_pos = np.array([whole[0] + 0.15, 0.0, info["belt_top_z"] + args.part_size / 2.0 + 0.02])
        spawner = PartSpawner(
            stage=stage,
            part_files=part_files,
            entry_pos=entry_pos,
            exit_x=whole[3] - 0.05,
            part_size=args.part_size,
            spawn_interval=args.spawn_interval,
            physics_fps=physics_fps,
        )

    camera = None
    video_writer = None
    if args.video:
        cam_center = info["center"]
        cam_size = info["size"]
        cam_span = None
        if spawner is not None:
            # 부품은 벨트 길이의 1/50 수준이라 전체 뷰로는 점처럼 보임.
            # 벨트를 타고 흘러가는 구간(입구에서 40% 지점)에 초점을 맞추고
            # 폭 기준으로 확대해서 부품 모양이 실제로 보이게 한다.
            whole = info["whole_aabb"]
            focus_x = whole[0] + (whole[3] - whole[0]) * 0.4
            cam_center = np.array([focus_x, info["center"][1], info["belt_top_z"]])
            cam_span = max(0.55, whole[4] - whole[1])
        camera = add_conveyor_camera(
            center=cam_center,
            size=cam_size,
            span=cam_span,
            resolution=(args.video_width, args.video_height),
            distance_factor=args.camera_distance,
            focal_length=args.camera_focal_length,
        )

    world.reset()

    # 카메라 렌더 파이프라인 워밍업 (첫 몇 프레임은 유효한 이미지가 아닐 수 있음)
    if camera is not None:
        for _ in range(10):
            world.step(render=True)

        import imageio

        video_path = args.video_out or DEFAULT_VIDEO_PATH
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        video_writer = imageio.get_writer(video_path, fps=args.video_fps, codec="libx264", quality=8)

    save_path = args.save or DEFAULT_SAVE_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    run_seconds = max(args.run_seconds, args.video_seconds if args.video else 0.0)
    if run_seconds > 0:
        do_render = (video_writer is not None) or not args.headless

        if video_writer is not None:
            # 요청한 영상 길이(초)를 프레임 단위로 정확히 채운다. 물리 스텝을
            # video_fps 간격으로 묶어 진행하고, 렌더 애노테이터가 간헐적으로
            # 한 스텝 늦게 준비되는 경우를 대비해 몇 번 더 스텝을 밟아 재시도한다.
            # (재시도 없이 바로 스킵하면 영상 길이가 요청보다 짧아짐)
            sub_steps = max(1, round(physics_fps / args.video_fps))
            frames_wanted = int(round(run_seconds * args.video_fps))
            frames_written = 0
            for _ in range(frames_wanted):
                for _ in range(sub_steps):
                    world.step(render=True)
                    if spawner is not None:
                        spawner.step()
                frame = camera.get_rgba()
                retries = 0
                while (frame is None or frame.size == 0) and retries < 30:
                    world.step(render=True)
                    frame = camera.get_rgba()
                    retries += 1
                if frame is not None and frame.size > 0:
                    video_writer.append_data(np.asarray(frame[:, :, :3], dtype=np.uint8))
                    frames_written += 1
            print(f"[create_conveyor] video frames written: {frames_written}/{frames_wanted}")
        else:
            total_steps = max(1, int(round(run_seconds * physics_fps)))
            for _ in range(total_steps):
                world.step(render=do_render)
                if spawner is not None:
                    spawner.step()

    if video_writer is not None:
        video_writer.close()
        print(f"[create_conveyor] saved video to: {video_path}")

    import omni.usd

    if args.headless:
        # 헤드리스는 이 시점 이후로 더 이상 World를 쓰지 않으므로 바로 저장해도 안전함.
        omni.usd.get_context().save_as_stage(save_path)
        print(f"[create_conveyor] saved stage to: {save_path}")
    else:
        # GUI 모드에서는 창을 닫기 전까지 계속 재생 상태로 둔다.
        # 주의: save_as_stage()는 내부적으로 "새 스테이지가 열림" 이벤트를
        # 발생시키고, World의 SimulationContext가 이 이벤트를 구독해 자기
        # 자신을 무효화(_scene 등 내부 상태 삭제)해버린다. 그래서 재생 루프
        # *이전*에 저장하면 곧바로 이어지는 world.step() 호출이
        # AttributeError로 죽는다. 그러니 재생 루프가 끝난 뒤(창을 닫은 뒤)에
        # 저장해야 한다.
        world.play()
        try:
            while simulation_app.is_running():
                world.step(render=True)
                if spawner is not None:
                    spawner.step()
        except Exception:
            pass

        try:
            omni.usd.get_context().save_as_stage(save_path)
            print(f"[create_conveyor] saved stage to: {save_path}")
        except Exception:
            print(f"[create_conveyor] warning: could not save stage to {save_path} on exit")

    simulation_app.close()


if __name__ == "__main__":
    main()
