"""
ISCO - 검사 공정 시뮬레이션용 컨베이어 벨트 생성 스크립트 (NVIDIA Isaac Sim)

이 스크립트는 이제 `isco.sim` 패키지 위의 얇은 CLI 래퍼다 (PoC 939줄을
`isco/sim/{app,scene,part_prim,rig,capture}.py`로 리팩터; CLI 플래그·동작은 그대로).

무엇을 만드는가
----------------
검사 셀(inspection cell)의 물류 이동부인 컨베이어 벨트 하나를 USD 스테이지에
생성한다. 지오메트리는 NVIDIA 공식 에셋 서버(Nucleus/CDN)의
`Isaac/Props/Conveyors/` 컬렉션에서 참조해온다 (isaacsim.asset.gen.conveyor.ui
확장의 "Conveyor Track Builder"가 쓰는 것과 같은 에셋). 47종류(직선/커브/T분기/
Y분기/경사로, 벨트·롤러·듀얼 스타일)가 있으며 `--conveyor-asset`로 고를 수 있다.
기본값은 가장 단순한 직선 벨트형인 ConveyorBelt_A06.

** 인터넷 연결이 필요함 ** (첫 실행 시 에셋을 스트리밍해오고, 이후엔 Kit이
로컬에 캐시해둔다).

에셋의 "Belt" 하위 프림은 이미 kinematic rigid body + 충돌까지 갖춰서 오므로,
여기에 PhysX의 "Surface Velocity" 기법만 얹는다 (PhysxSchema.PhysxSurfaceVelocityAPI).

실행 방법 (env_isaaclab conda 환경에서, 레포 루트 또는 isaac_sim/ 어디서든)
-----------------------------------------------------------------------
    conda activate env_isaaclab

    # GUI를 띄워 눈으로 확인 (자동 재생됨, 창을 닫으면 저장 후 종료)
    python isaac_sim/create_conveyor.py

    # 헤드리스로 5초 시뮬레이션 후 USD 저장하고 종료
    python isaac_sim/create_conveyor.py --headless --run-seconds 5

    # 다른 종류의 컨베이어로 바꾸기 / 목록 보기
    python isaac_sim/create_conveyor.py --conveyor-asset ConveyorBelt_A02
    python isaac_sim/create_conveyor.py --list-assets

    # 헤드리스로 시뮬레이션을 재생하며 mp4 영상까지 저장
    python isaac_sim/create_conveyor.py --headless --video --video-seconds 6

    # 부품 하나를 3카메라로 동시 촬영 + depth 재구성 커버리지 계산
    python isaac_sim/create_conveyor.py --headless --capture-views --views-part 41229_16283ae1_0018.stl

산출물
------
    isaac_sim/assets/conveyor_belt.usd    (기본 저장 경로, --save로 변경 가능)
    isaac_sim/output/conveyor_belt.mp4    (--video 지정 시, --video-out으로 변경 가능)
    isaac_sim/output/views/               (--capture-views, --views-out-dir로 변경 가능)
        view_{i}.png, merged_pointcloud.npy/.ply, ground_truth_mesh.ply, coverage.png,
        summary.txt, summary.json (completeness/accuracy @2mm 및 @0.2mm)

다중 카메라 촬영 (--capture-views)과 ADR-0008
--------------------------------------------
원래 PoC는 부품 포즈(정답)를 읽은 뒤 카메라 워밍업 렌더 스텝 동안 벨트가 계속
부품을 밀어서 렌더와 정답이 어긋나는 레이스가 있었다. 기본 동작은 이제 워밍업 전에
부품을 고정(벨트 속도 0 + kinematic)하고, 촬영 시점의 포즈를 정답으로 쓰며, AABB
필터 대신 semantic 마스크로 부품 점만 뽑는다. `--legacy-race`는 원래 PoC 로직을
그대로 재현하고(비교용), `--views-aabb-filter`는 포인트클라우드 애노테이터 + AABB
필터를 fallback으로 쓴다.

부품 STL은 Fusion 360 Gallery(재배포 금지) - `data/parts/raw/fusion360/*.stl`.
"""

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_SAVE_PATH = os.path.join(SCRIPT_DIR, "assets", "conveyor_belt.usd")
DEFAULT_VIDEO_PATH = os.path.join(SCRIPT_DIR, "output", "conveyor_belt.mp4")
# Fusion 360 Gallery STL은 재배포 금지 라이선스라 git에서 빼고 gitignore된 data/로 옮겼다.
DEFAULT_PARTS_DIR = os.path.join(REPO_ROOT, "data", "parts", "raw", "fusion360")
DEFAULT_VIEWS_DIR = os.path.join(SCRIPT_DIR, "output", "views")

try:  # `pip install -e .` 없이 실행해도 동작하도록 레포 루트를 경로에 추가
    import isco  # noqa: F401
except ImportError:
    sys.path.insert(0, REPO_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Isaac Sim 컨베이어 벨트 생성 스크립트 (isco.sim 래퍼)")
    parser.add_argument("--headless", action="store_true", help="헤드리스로 실행 (GUI 없이 저장만)")
    parser.add_argument("--conveyor-asset", type=str, default="ConveyorBelt_A06",
                        help="Isaac/Props/Conveyors/ 아래 에셋 이름 (기본: 직선 벨트형 A06). --list-assets로 목록 확인")
    parser.add_argument("--list-assets", action="store_true", help="사용 가능한 컨베이어 에셋 이름 47종을 출력하고 종료")
    parser.add_argument("--speed", type=float, default=0.3, help="벨트 표면 속도(m/s)")
    parser.add_argument("--no-parts", dest="spawn_parts", action="store_false", help="부품 스폰 없이 빈 벨트만 생성")
    parser.add_argument("--spawn-interval", type=float, default=1.5, help="새 부품을 올리는 간격(초)")
    parser.add_argument("--part-size", type=float, default=0.04, help="부품 하나의 최대 치수(m)")
    parser.add_argument("--parts-dir", type=str, default=None,
                        help="부품 STL이 들어있는 폴더 (기본: data/parts/raw/fusion360/)")
    parser.add_argument("--run-seconds", type=float, default=0.0, help="저장 전 시뮬레이션을 n초 재생(헤드리스 검증용)")
    parser.add_argument("--save", type=str, default=None, help="저장할 USD 경로 (기본: isaac_sim/assets/conveyor_belt.usd)")
    parser.add_argument("--video", action="store_true", help="시뮬레이션을 mp4 영상으로 녹화")
    parser.add_argument("--video-out", type=str, default=None, help="저장할 mp4 경로 (기본: isaac_sim/output/conveyor_belt.mp4)")
    parser.add_argument("--video-seconds", type=float, default=6.0, help="녹화 길이(초). --run-seconds가 더 크면 그 값을 따름")
    parser.add_argument("--video-fps", type=int, default=30, help="영상 프레임레이트")
    parser.add_argument("--video-width", type=int, default=1280, help="영상 가로 해상도")
    parser.add_argument("--video-height", type=int, default=720, help="영상 세로 해상도")
    parser.add_argument("--camera-distance", type=float, default=1.6, help="카메라를 컨베이어 크기 대비 얼마나 멀리 둘지 배율")
    parser.add_argument("--camera-focal-length", type=float, default=1.8,
                        help="카메라 초점거리(stage 단위, Camera.set_focal_length). 기본 조리개 기준 1.8 ~ 60도 화각")
    parser.add_argument("--capture-views", action="store_true", help="부품 하나를 여러 각도 카메라로 동시 촬영하고 종료")
    parser.add_argument("--views-count", type=int, default=3, help="배치할 카메라 대수")
    parser.add_argument("--views-elevation", type=float, default=30.0, help="카메라 고각(도) - 클수록 위에서 내려다봄")
    parser.add_argument("--views-radius-factor", type=float, default=45.0,
                        help="부품 크기 대비 카메라 거리 배율 (컨베이어 프레임에 파묻히지 않도록 수십 배)")
    parser.add_argument("--views-focal-length", type=float, default=24.0, help="촬영 카메라 초점거리(stage 단위)")
    parser.add_argument("--views-width", type=int, default=1024, help="촬영 이미지 가로 해상도")
    parser.add_argument("--views-height", type=int, default=1024, help="촬영 이미지 세로 해상도")
    parser.add_argument("--views-part", type=str, default=None, help="촬영할 특정 부품 STL 파일명 (기본: 랜덤 선택)")
    parser.add_argument("--views-out-dir", type=str, default=None, help="촬영 이미지 저장 폴더 (기본: isaac_sim/output/views/)")
    # --- isco.sim 리팩터에서 추가된 플래그 ---
    parser.add_argument("--legacy-race", action="store_true",
                        help="(비교용) 원래 PoC 캡처 로직 재현: 워밍업 중 부품이 계속 이동, 정답 포즈를 워밍업 전에 읽음, AABB 필터")
    parser.add_argument("--views-aabb-filter", action="store_true",
                        help="semantic 마스크 대신 포인트클라우드 애노테이터 + AABB 필터로 부품 점 추출 (fallback)")
    parser.add_argument("--views-seed", type=int, default=None, help="--views-part 미지정 시 부품 랜덤 선택 시드")
    parser.add_argument("--gui-loop-seconds", type=float, default=0.0,
                        help="GUI 재생 루프를 n초(sim time) 후 종료. --headless와 함께 주면 창 없이 GUI 코드 경로"
                             "(world.play 루프 + 종료 시 저장)를 검증")
    return parser


def find_track_types_json():
    """isaacsim.asset.gen.conveyor.ui 확장에 번들된 track_types.json 위치 (앱을 띄우지 않고)."""
    import importlib.util

    spec = importlib.util.find_spec("isaacsim")
    roots = list(spec.submodule_search_locations or []) if spec else []
    for root in roots:
        p = os.path.join(root, "exts", "isaacsim.asset.gen.conveyor.ui", "data", "track_types.json")
        if os.path.isfile(p):
            return p
    return None


def list_assets() -> int:
    path = find_track_types_json()
    if path is None:
        print("track_types.json을 찾지 못함 (isaacsim 패키지가 설치된 환경인지 확인할 것)")
        return 1
    with open(path) as f:
        assets = json.load(f)["assets"]
    for name, cfg in sorted(assets.items()):
        print(f"{name:20s} style={cfg.get('style'):6s} type={cfg.get('type')}")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_assets:
        return list_assets()

    from isco.sim import app as isco_app

    # SimulationApp은 다른 Isaac/pxr import보다 먼저 만들어야 한다 (create_new_stage=False, isco.sim.app 참고)
    simulation_app = isco_app.launch(headless=args.headless)

    import numpy as np
    import omni.usd

    from isco.sim.capture import LegacyCaptureOptions, legacy_capture_views
    from isco.sim.rig import add_conveyor_camera
    from isco.sim.scene import IscoScene, PartSpawner, find_part_stl_files

    scene = IscoScene.build(conveyor_asset=args.conveyor_asset, speed=args.speed)
    world, stage, info = scene.world, scene.stage, scene.info
    physics_fps = scene.physics_fps
    parts_dir = args.parts_dir or DEFAULT_PARTS_DIR
    save_path = args.save or DEFAULT_SAVE_PATH

    if args.capture_views:
        part_files = find_part_stl_files(parts_dir)
        opts = LegacyCaptureOptions(
            part_size=args.part_size, speed=args.speed, parts_dir=parts_dir, views_part=args.views_part,
            views_count=args.views_count, views_elevation=args.views_elevation,
            views_radius_factor=args.views_radius_factor, views_focal_length=args.views_focal_length,
            views_width=args.views_width, views_height=args.views_height, seed=args.views_seed)
        res = legacy_capture_views(scene, part_files, opts, args.views_out_dir or DEFAULT_VIEWS_DIR,
                                   race_fix=not args.legacy_race, aabb_filter=args.views_aabb_filter,
                                   log=lambda m: print(m.replace("[capture]", "[create_conveyor]"), flush=True))
        for p in res["saved"]:
            print(f"[create_conveyor] saved view: {p}", flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        omni.usd.get_context().save_as_stage(save_path)
        print(f"[create_conveyor] saved stage to: {save_path}", flush=True)
        isco_app.close()
        return 0

    spawner = None
    whole = info["whole_aabb"]
    if args.spawn_parts:
        part_files = find_part_stl_files(parts_dir)
        entry_pos = np.array([whole[0] + 0.15, 0.0, info["belt_top_z"] + args.part_size / 2.0 + 0.02])
        spawner = PartSpawner(stage=stage, part_files=part_files, entry_pos=entry_pos, exit_x=whole[3] - 0.05,
                              part_size=args.part_size, spawn_interval=args.spawn_interval, physics_fps=physics_fps)

    camera = None
    video_writer = None
    video_path = args.video_out or DEFAULT_VIDEO_PATH
    if args.video:
        cam_center, cam_size, cam_span = info["center"], info["size"], None
        if spawner is not None:
            # 부품은 벨트 길이의 1/50 수준이라 전체 뷰로는 점처럼 보임 -> 입구에서 40% 지점에 초점, 폭 기준 확대
            focus_x = whole[0] + (whole[3] - whole[0]) * 0.4
            cam_center = np.array([focus_x, info["center"][1], info["belt_top_z"]])
            cam_span = max(0.55, whole[4] - whole[1])
        camera = add_conveyor_camera(center=cam_center, size=cam_size, span=cam_span,
                                     resolution=(args.video_width, args.video_height),
                                     distance_factor=args.camera_distance, focal_length=args.camera_focal_length)

    world.reset()
    if camera is not None:
        for _ in range(10):  # 렌더 파이프라인 워밍업
            world.step(render=True)
        import imageio

        os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)
        video_writer = imageio.get_writer(video_path, fps=args.video_fps, codec="libx264", quality=8)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    gui_loop = (not args.headless) or args.gui_loop_seconds > 0
    run_seconds = max(args.run_seconds, args.video_seconds if args.video else 0.0)
    if run_seconds > 0:
        do_render = (video_writer is not None) or not args.headless
        if video_writer is not None:
            # 요청한 영상 길이를 프레임 단위로 정확히 채운다 (애노테이터가 한 스텝 늦을 때 재시도)
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
            print(f"[create_conveyor] video frames written: {frames_written}/{frames_wanted}", flush=True)
        else:
            for _ in range(max(1, int(round(run_seconds * physics_fps)))):
                world.step(render=do_render)
                if spawner is not None:
                    spawner.step()

    if video_writer is not None:
        video_writer.close()
        print(f"[create_conveyor] saved video to: {video_path}", flush=True)

    if not gui_loop:
        # 헤드리스는 이 시점 이후로 World를 쓰지 않으므로 바로 저장해도 안전함
        omni.usd.get_context().save_as_stage(save_path)
        print(f"[create_conveyor] saved stage to: {save_path}", flush=True)
    else:
        # GUI 모드: 창을 닫기 전까지 계속 재생. save_as_stage()는 "새 스테이지 열림" 이벤트로 World를
        # 무효화하므로 반드시 재생 루프가 끝난 뒤에 저장한다 (PoC 주석 참고).
        world.play()
        max_steps = int(round(args.gui_loop_seconds * physics_fps)) if args.gui_loop_seconds > 0 else None
        steps = 0
        try:
            while simulation_app.is_running():
                world.step(render=True)
                if spawner is not None:
                    spawner.step()
                steps += 1
                if max_steps is not None and steps >= max_steps:
                    break
        except Exception as e:
            print(f"[create_conveyor] GUI loop stopped: {e}")
        n_active = len(spawner.active) if spawner is not None else 0
        n_spawned = spawner.spawn_count if spawner is not None else 0
        print(f"[create_conveyor] GUI loop ran {steps} steps ({steps / physics_fps:.1f}s sim), "
              f"spawned {n_spawned} parts, {n_active} active", flush=True)
        try:
            omni.usd.get_context().save_as_stage(save_path)
            print(f"[create_conveyor] saved stage to: {save_path}", flush=True)
        except Exception:
            print(f"[create_conveyor] warning: could not save stage to {save_path} on exit")
    isco_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
