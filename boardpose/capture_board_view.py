#!/usr/bin/env python3
"""
capture_board_view.py - capture one board view for the pose pipeline.

You jog the arm to a pose, press ENTER, and this saves everything needed to
localise the board and, later, to calibrate the robot chain:

    <root>/pose_N/<HHMMSS_uuuuuu>/
        color.png       full resolution, unresampled, lossless
        depth.png       16-bit, registered to colour if the driver publishes it
        pose.txt        T_base_tcp, same format the existing pipeline parses
        joints.json     JOINT ANGLES, velocities and efforts   <-- new
        meta.json       settling, sharpness, frame ages, topic names
    <root>/camera_info.json

Why joints.json matters: the board pose is good to 0.25 mm in the camera frame,
but the same static board scatters by 2-5 mm once it is pushed through
T_base_tcp @ X. Fixing that means calibrating the robot -- joint offsets and
joint compliance -- and every such model is a function of the joint angles.
The existing dataset records only the FK output, so none of it can be fitted;
handeye_board.py's droop model has to guess the arm geometry from the TCP pose
alone and, unsurprisingly, fails. Efforts are recorded too: they are a direct
measurement of the gravity load a compliance model is supposed to explain.

Run it once and leave it running. A persistent node keeps ROS discovery, the
camera stream and the auto-exposure loop warm; restarting per view costs
several seconds of discovery and gives you the first, worst-exposed frame.

    python3 capture_board_view.py --root images/images_depth --pose 4

    [ENTER]   capture the current pose
    p <n>     switch to pose_<n> (a new static board position)
    d         delete the last capture
    s         coverage statistics -- how identifiable a chain model would be
    q         quit

Capture protocol this script enforces (see README, "Capture protocol"):

  * waits --settle seconds after you press ENTER, so the stick stops ringing
  * then watches for --window seconds and measures how much the TCP and the
    image actually moved; a view that was not static is flagged, not silently
    saved
  * refuses frames older than the keypress, so a stale pre-move frame from the
    subscriber buffer can never be saved
  * never resizes or re-encodes lossily -- an interpolated pixel is a corrupted
    subpixel edge, and subpixel edges are the entire measurement
  * reports focus sharpness, so a defocused pocket floor is caught at capture
    time rather than three days later in the bundle
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_ros import Buffer, TransformListener

# Frames, as in record_teleop_dataset.py. BASE_FRAME is the one every capture
# MUST be expressed in; PLANNING_FRAMES is only a fallback if it does not resolve.
# Do not rely on the fallback: mixing frames across folders is silent corruption.
BASE_FRAME = "base_link"
PLANNING_FRAMES = ["base_link", "right_arm_base_link", "base_footprint", "world"]
EE_LINK = "right_arm_tool0"

# The camera on the stick is an Orbbec Femto driven by femto_4k.launch.py, whose
# camera_name is wrist_camera. The Orbbec driver publishes <camera_name>/<stream>,
# not the azure_kinect_ros_driver names these once had. Override with
# --color / --depth / --info if camera_name is changed.
CAMERA = "/wrist_camera"
COLOR_TOPIC = f"{CAMERA}/color/image_raw"
DEPTH_TOPIC = f"{CAMERA}/depth/image_raw"     # already registered to colour
INFO_TOPIC = f"{CAMERA}/color/camera_info"
JOINTS_TOPIC = "/joint_states"

# Focus measurement: on by default, ~0.1 s. --no-focus turns it off.
ENABLE_FOCUS = True
BUDGET_S = 2.0


def decode_color(msg):
    """Decode to BGR at native resolution. No resize -- see module docstring."""
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    enc = msg.encoding.lower()
    if enc in ("bgra8", "rgba8"):
        img = buf.reshape(msg.height, msg.width, 4)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR if enc == "bgra8" else cv2.COLOR_RGBA2BGR)
    elif enc in ("bgr8", "rgb8"):
        img = buf.reshape(msg.height, msg.width, 3)
        if enc == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif enc in ("mono8",):
        img = cv2.cvtColor(buf.reshape(msg.height, msg.width), cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError(f"unsupported colour encoding {msg.encoding!r}")
    return img


def decode_depth(msg):
    """Decode depth at native resolution, as uint16 millimetres."""
    enc = msg.encoding.lower()
    if enc in ("16uc1", "mono16"):
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
    if enc == "32fc1":
        d = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        return np.nan_to_num(d * 1000.0).clip(0, 65535).astype(np.uint16)
    raise ValueError(f"unsupported depth encoding {msg.encoding!r}")


def stamp_sec(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def sharpness(img):
    """Variance of the Laplacian on a centre crop. Higher is "sharper".

    KEPT ONLY FOR CONTINUITY with the older captures. It does NOT measure focus:
    it scales with how much high-contrast content lands in the crop, so framing
    the smooth transparent board tightly makes it collapse while focus is
    unchanged. Two frames at identical focus measured 54.2 and 94.4 here (1.74x)
    while their busy-pixel fractions were 6.20% and 11.43% (1.84x) -- it was
    tracking clutter, essentially one-for-one. Read focus() instead.
    """
    h, w = img.shape[:2]
    c = img[h // 3:2 * h // 3, w // 3:2 * w // 3]
    return float(cv2.Laplacian(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def focus(img):
    """Median 10-90% intensity rise across the pocket rims, in pixels.

    The honest focus measure: a rise width belongs to the optics and the
    sampling, so it does not care how much contrast an edge has or how much else
    is in frame. On the two frames above it returned 1.79 and 1.71 px, correctly
    reporting that focus had not changed. LOWER IS SHARPER. Costs ~2 s at 4K,
    which is affordable next to the settling wait, and it is the number that
    catches a genuinely soft pocket floor -- the 22 mm baseline that makes tilt
    observable at all is the first thing to go when focus drifts.

    Returns NaN when no rims are found (board out of frame, or too oblique), or
    when it takes longer than BUDGET_S -- a capture loop must never be able to
    block on a diagnostic.
    """
    if not ENABLE_FOCUS:
        return float("nan")
    try:
        import tune_lighting
    except ImportError:
        # Almost always because this file was COPIED somewhere without its
        # sibling tune_lighting.py. Say so instead of returning a bare NaN that
        # looks like a framing problem -- that mistake has already cost one
        # 43-view session here, via a stale copy nobody knew was stale.
        if not getattr(focus, "_warned", False):
            focus._warned = True
            print("    NOTE: tune_lighting.py not importable, so focus cannot be "
                  "measured.\n          Run the original at "
                  f"{Path(__file__).resolve().parent}/capture_board_view.py "
                  "rather than a copy.")
        return float("nan")
    t0 = time.time()
    v = tune_lighting.focus(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    dt = time.time() - t0
    if dt > BUDGET_S and not getattr(focus, "_slow", False):
        focus._slow = True
        print(f"    NOTE: focus took {dt:.1f}s, over the {BUDGET_S}s budget. "
              "Pass --no-focus to skip it.")
    return v


class BoardCaptureNode(Node):
    def __init__(self, topics):
        super().__init__("board_capture_node")
        self.topics = topics
        self.cb = ReentrantCallbackGroup()
        self.color = None      # (msg_stamp, wall_time, image)
        self.depth = None
        self.joints = None
        self.info = None
        self.ee_link = EE_LINK
        self.base_frame = BASE_FRAME
        self.planning_frame = None

        self.create_subscription(Image, topics["color"], self._color_cb, 1, callback_group=self.cb)
        self.create_subscription(Image, topics["depth"], self._depth_cb, 1, callback_group=self.cb)
        self.create_subscription(JointState, topics["joints"], self._joints_cb, 10, callback_group=self.cb)
        self.create_subscription(CameraInfo, topics["info"], self._info_cb, 1, callback_group=self.cb)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def _color_cb(self, msg):
        try:
            self.color = (stamp_sec(msg), time.time(), decode_color(msg))
        except Exception as e:
            self.get_logger().error(f"colour decode: {e}")

    def _depth_cb(self, msg):
        try:
            self.depth = (stamp_sec(msg), time.time(), decode_depth(msg))
        except Exception as e:
            self.get_logger().error(f"depth decode: {e}")

    def _joints_cb(self, msg):
        self.joints = (stamp_sec(msg), time.time(), {
            "name": list(msg.name),
            "position": [float(x) for x in msg.position],
            "velocity": [float(x) for x in msg.velocity],
            "effort": [float(x) for x in msg.effort],
        })

    def _info_cb(self, msg):
        self.info = {
            "width": msg.width, "height": msg.height,
            "distortion_model": msg.distortion_model,
            "K": [float(x) for x in msg.k], "D": [float(x) for x in msg.d],
            "P": [float(x) for x in msg.p], "frame_id": msg.header.frame_id,
        }

    def tcp_pose(self):
        """T_base_tcp as (frame, translation, quaternion xyzw, stamp).

        The frame is PINNED once chosen, and BASE_FRAME pins it across runs.
        The original first-match-wins over PLANNING_FRAMES silently picked a
        different frame per session depending on which TF arrived first: one
        capture run landed in base_footprint, another in base_link (10 mm apart,
        harmless) and another in right_arm_base_link -- 1.4 m and 135 deg away.
        Per-folder T_base_board absorbs a constant frame change, so hand-eye
        survived it, but anything reading t_base_tcp directly does not: the droop
        model, the reach and azimuth coverage gates, all silently wrong.
        """
        frames = ([self.planning_frame] if self.planning_frame else []) + \
                 ([self.base_frame] if self.base_frame else []) + PLANNING_FRAMES
        for frame in frames:
            try:
                tr = self.tf_buffer.lookup_transform(
                    frame, self.ee_link, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.05))
            except Exception:
                continue
            self.planning_frame = frame
            t, q = tr.transform.translation, tr.transform.rotation
            return (frame, np.array([t.x, t.y, t.z]),
                    np.array([q.x, q.y, q.z, q.w]), stamp_sec(tr))
        return None


def wait_ready(node, timeout=15.0):
    """Block until every required stream has produced at least one message."""
    deadline = time.time() + timeout
    need = {"colour": lambda: node.color, "depth": lambda: node.depth,
            "joint states": lambda: node.joints, "TF": node.tcp_pose}
    while time.time() < deadline:
        missing = [k for k, f in need.items() if not f()]
        if not missing:
            return []
        time.sleep(0.2)
    return [k for k, f in need.items() if not f()]


def observe(node, settle, window, t_press):
    """Wait out the ringing, then measure whether the view is actually static.

    Returns (sample, motion) or (None, reason). Every returned frame is
    guaranteed to postdate the keypress, so a buffered pre-move frame can never
    be saved as if it were the new pose.
    """
    time.sleep(settle)
    t_end = time.time() + window
    poses, small, first = [], [], None
    while time.time() < t_end:
        p = node.tcp_pose()
        if p:
            poses.append(p[1])
        if node.color and node.color[1] > t_press:
            img = node.color[2]
            g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (320, 180))
            small.append(g.astype(np.float32))
            first = node.color
        time.sleep(0.05)

    if first is None:
        return None, "no colour frame arrived after the keypress"
    if not node.depth or node.depth[1] <= t_press:
        return None, "no depth frame arrived after the keypress"
    if not node.joints or node.joints[1] <= t_press:
        return None, "no joint state arrived after the keypress"
    pose = node.tcp_pose()
    if pose is None:
        return None, "TF lookup failed"

    P = np.array(poses)
    motion = {
        "tcp_range_mm": float(np.linalg.norm(P.max(0) - P.min(0)) * 1000) if len(P) > 1 else 0.0,
        "image_rms_levels": float(np.sqrt(((np.array(small) - small[0]) ** 2).mean()))
        if len(small) > 1 else 0.0,
        "n_pose_samples": len(P), "n_image_samples": len(small),
        "joint_speed_max": float(np.abs(node.joints[2]["velocity"]).max())
        if node.joints[2]["velocity"] else 0.0,
    }
    sample = {"color": node.color, "depth": node.depth,
              "joints": node.joints, "pose": pose}
    return sample, motion


POSE_TXT = """# Translation (x, y, z)
translation_x: {t[0]:.15f}
translation_y: {t[1]:.15f}
translation_z: {t[2]:.15f}

# Rotation Quaternion (x, y, z, w)
rotation_x: {q[0]:.15f}
rotation_y: {q[1]:.15f}
rotation_z: {q[2]:.15f}
rotation_w: {q[3]:.15f}

# Timestamp
secs: {secs}
nsecs: {nsecs}
"""


def save(root, pose_idx, sample, motion, topics, info, ee_link):
    """Write one view. Layout matches images/images_depth/pose_N/<view>/."""
    now = time.time()
    d = Path(root) / f"pose_{pose_idx}" / (time.strftime("%H%M%S", time.localtime(now))
                                          + f"_{int(now % 1 * 1e6):06d}")
    d.mkdir(parents=True, exist_ok=True)

    color, depth, joints, pose = sample["color"], sample["depth"], sample["joints"], sample["pose"]
    frame, t, q, pose_stamp = pose

    cv2.imwrite(str(d / "color.png"), color[2], [cv2.IMWRITE_PNG_COMPRESSION, 3])
    cv2.imwrite(str(d / "depth.png"), depth[2], [cv2.IMWRITE_PNG_COMPRESSION, 3])
    (d / "pose.txt").write_text(POSE_TXT.format(
        t=t, q=q, secs=int(pose_stamp), nsecs=int((pose_stamp % 1) * 1e9)))

    joints[2].update({"stamp": joints[0], "base_frame": frame, "ee_link": ee_link})
    (d / "joints.json").write_text(json.dumps(joints[2], indent=2))

    sharp = sharpness(color[2])
    foc = focus(color[2])
    meta = {
        "captured_at": now,
        "base_frame": frame, "ee_link": ee_link,
        "topics": topics,
        "color_shape": list(color[2].shape), "depth_shape": list(depth[2].shape),
        "sharpness": sharp,
        "focus_rise_px": foc,
        "motion": motion,
        # how stale each stream was relative to the moment of writing
        "age_s": {"color": now - color[1], "depth": now - depth[1],
                  "joints": now - joints[1], "tf": now - pose_stamp},
        "tcp": {"translation": t.tolist(), "quaternion_xyzw": q.tolist()},
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2))

    if info:
        p = Path(root) / "camera_info.json"
        if not p.exists():
            p.write_text(json.dumps(info, indent=2))
    return d, sharp, foc


def coverage(root):
    """What the captured set can and cannot identify.

    A chain model is only fittable if the captures excite it. Shoulder droop is
    a function of the horizontal reach and its azimuth; if either barely varies,
    the deflection is near-constant within a board position and gets absorbed
    wholesale into that position's free 6-DoF pose. That is exactly what
    happened to the existing 47 views, whose horizontal reach spans only
    143-516 mm while |t| looks like a healthy 1445-1691 mm.
    """
    root = Path(root)
    rows = {}
    for pose_dir in sorted(root.glob("pose_*")):
        vs = []
        for v in sorted(pose_dir.iterdir()):
            f = v / "meta.json"
            if f.is_file():
                vs.append(json.loads(f.read_text()))
        if vs:
            rows[pose_dir.name] = vs
    if not rows:
        print("  nothing captured yet")
        return

    def span(a, fmt="{:.0f}"):
        return (fmt.format(a.min()) + ".." + fmt.format(a.max())).rjust(16)

    print(f"\n  {'set':10s} {'n':>3s} {'horiz reach mm':>16s} {'azimuth deg':>16s} "
          f"{'tcp z mm':>16s} {'sharpness':>16s}")
    allt = []
    for name, vs in rows.items():
        T = np.array([v["tcp"]["translation"] for v in vs]) * 1000
        allt.append(T)
        print(f"  {name:10s} {len(vs):3d} {span(np.hypot(T[:, 0], T[:, 1]))} "
              f"{span(np.degrees(np.arctan2(T[:, 1], T[:, 0])), '{:+.0f}')} "
              f"{span(T[:, 2])} {span(np.array([v['sharpness'] for v in vs]))}")

    T = np.vstack(allt)
    r = np.hypot(T[:, 0], T[:, 1])
    az = np.degrees(np.arctan2(T[:, 1], T[:, 0]))
    print(f"\n  overall: {len(T)} views, horizontal reach spread {np.ptp(r):.0f} mm, "
          f"azimuth spread {np.ptp(az):.0f} deg, z spread {np.ptp(T[:, 2]):.0f} mm")
    for ok, msg in [
        (np.ptp(r) > 400, f"horizontal reach spread {np.ptp(r):.0f} mm (want > 400) -- "
                          "without it a load model is unidentifiable"),
        (np.ptp(az) > 60, f"azimuth spread {np.ptp(az):.0f} deg (want > 60)"),
        (len(rows) >= 3, f"{len(rows)} board positions (want >= 3)"),
        (len(T) >= 30, f"{len(T)} views (want >= 30)"),
    ]:
        print(f"    [{'ok' if ok else '  '}] {msg}")
    print("  also capture, at each board position: the same camera pose with the\n"
          "  wrist rolled 90/180 deg (constrains the hand-eye translation), and the\n"
          "  same camera pose from a different arm configuration (separates FK error\n"
          "  from hand-eye error). Neither is visible in the numbers above.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="images/images_depth", help="dataset root")
    ap.add_argument("--pose", type=int, default=1,
                    help="board position index -> pose_<N>/. Bump it whenever you "
                         "physically move the board.")
    ap.add_argument("--settle", type=float, default=1.0,
                    help="seconds to wait after ENTER before looking, so the stick "
                         "stops ringing")
    ap.add_argument("--window", type=float, default=0.6,
                    help="seconds to watch for residual motion")
    ap.add_argument("--max-tcp-motion-mm", type=float, default=0.10,
                    help="flag the view if the TCP moved more than this while watching")
    ap.add_argument("--once", action="store_true", help="capture one view and exit")
    ap.add_argument("--no-focus", action="store_true",
                    help="skip the rim-rise focus measurement (~0.1 s per capture)")
    ap.add_argument("--color", default=COLOR_TOPIC)
    ap.add_argument("--depth", default=DEPTH_TOPIC)
    ap.add_argument("--info", default=INFO_TOPIC)
    ap.add_argument("--joints", default=JOINTS_TOPIC)
    ap.add_argument("--ee-link", default=EE_LINK)
    ap.add_argument("--base-frame", default=BASE_FRAME,
                    help="frame every pose is expressed in. Keep it identical "
                         "across every session; mixing frames corrupts the "
                         "reach/azimuth stats and any deflection model.")
    args = ap.parse_args()

    global ENABLE_FOCUS
    ENABLE_FOCUS = not args.no_focus

    topics = {"color": args.color, "depth": args.depth,
              "info": args.info, "joints": args.joints}

    rclpy.init()
    node = BoardCaptureNode(topics)
    node.ee_link = args.ee_link
    node.base_frame = args.base_frame
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    print(f"waiting for {topics['color']}, {topics['depth']}, {topics['joints']}, TF ...")
    missing = wait_ready(node)
    if missing:
        print(f"\nERROR: no data on: {', '.join(missing)}")
        print("  list what is actually published with:  ros2 topic list")
        print("  then pass --color/--depth/--info/--joints/--ee-link")
        executor.shutdown(); node.destroy_node(); rclpy.shutdown()
        return 1

    h, w = node.color[2].shape[:2]
    print(f"ready: colour {w}x{h} {node.color[2].dtype}, depth "
          f"{node.depth[2].shape[1]}x{node.depth[2].shape[0]}, "
          f"{len(node.joints[2]['name'])} joints, base frame {node.tcp_pose()[0]}")
    if node.joints[2]["name"]:
        print(f"  joints: {', '.join(node.joints[2]['name'][:8])}"
              f"{' ...' if len(node.joints[2]['name']) > 8 else ''}")
    if not node.joints[2]["effort"]:
        print("  NOTE: no effort field published -- compliance models lose their "
              "most direct load measurement, but joint angles alone are still enough "
              "for joint-offset calibration.")
    print("\nLock focus, exposure, gain and white balance before you start. A refocus "
          "mid-session\nchanges fx by ~0.5%, which is 2.4 mm of range error at 478 mm, "
          "and invalidates the bundle.\n")
    print("  [ENTER] capture   p <n> switch board position   d delete last   "
          "s coverage   q quit")

    pose_idx, last, count = args.pose, None, 0
    focus_hist = []
    try:
        while rclpy.ok():
            if args.once:
                cmd = ""
            else:
                try:
                    cmd = input(f"\n[pose_{pose_idx}] {count} captured > ").strip()
                except EOFError:
                    break
            if cmd == "q":
                break
            if cmd == "s":
                coverage(args.root); continue
            if cmd == "d":
                if last is None:
                    print("  nothing to delete")
                else:
                    for f in last.iterdir():
                        f.unlink()
                    last.rmdir()
                    print(f"  deleted {last}")
                    last, count = None, count - 1
                continue
            if cmd.startswith("p "):
                try:
                    pose_idx = int(cmd.split()[1])
                    print(f"  now writing to pose_{pose_idx} -- move the board, and "
                          f"keep it dead still for every view in this set")
                except (ValueError, IndexError):
                    print("  usage: p <n>")
                continue
            if cmd:
                print(f"  unknown command {cmd!r}")
                continue

            t_press = time.time()
            print(f"  settling {args.settle:.1f}s, watching {args.window:.1f}s ...")
            sample, motion = observe(node, args.settle, args.window, t_press)
            if sample is None:
                print(f"  FAILED: {motion}")
                if args.once:
                    break
                continue

            d, sharp, foc = save(args.root, pose_idx, sample, motion, topics,
                                 node.info, args.ee_link)
            last, count = d, count + 1
            flags = []
            if motion["tcp_range_mm"] > args.max_tcp_motion_mm:
                flags.append(f"TCP MOVED {motion['tcp_range_mm']:.2f} mm while watching")
            if motion["joint_speed_max"] > 1e-3:
                flags.append(f"joints still moving ({motion['joint_speed_max']:.4f} rad/s)")
            print(f"  saved {d}")
            print(f"    tcp drift {motion['tcp_range_mm']:.3f} mm | image rms "
                  f"{motion['image_rms_levels']:.2f} levels | "
                  f"focus {foc:.2f} px rise | sharpness {sharp:.0f} (clutter)")
            # Judge focus against THIS session, not an absolute number. The rise
            # width shifts with the OpenCV build (4.6 and 4.14 differ by ~0.35 px
            # on the same frame, via different Hough circles) and with standoff,
            # so a hardcoded threshold produces false alarms. A frame well above
            # the running median of its own session is the real signal.
            if np.isfinite(foc):
                if len(focus_hist) >= 4:
                    med = float(np.median(focus_hist))
                    if foc > 1.25 * med:
                        flags.append(f"SOFT: rim rise {foc:.2f} px vs {med:.2f} "
                                     "median for this session -- defocused or "
                                     "motion blurred")
                focus_hist.append(foc)
            else:
                flags.append("no rims found, so focus could not be measured -- "
                             "board out of frame, or too oblique")
            for f in flags:
                print(f"    WARNING: {f} -- consider 'd' and a longer --settle")
            if args.once:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if count:
            coverage(args.root)
        print(f"\n{count} views captured under {args.root}")
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
