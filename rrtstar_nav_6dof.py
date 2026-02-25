
#!/usr/bin/env python3


import math
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time as RclpyTime

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from tf2_ros import Buffer, TransformListener
from tf_transformations import euler_from_quaternion


def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def quat_to_rpy(q) -> Tuple[float, float, float]:
    r, p, y = euler_from_quaternion([q.x, q.y, q.z, q.w])
    return float(r), float(p), float(y)


def rotate_point_by_quat(px, py, pz, qx, qy, qz, qw):
    """
    Rotate point p by quaternion q (x,y,z,w), return rotated (rx,ry,rz).
    Optimized quaternion-vector rotation:
      t = 2*cross(qvec, p)
      p' = p + qw*t + cross(qvec, t)
    """
    t1 = 2.0 * (qy * pz - qz * py)
    t2 = 2.0 * (qz * px - qx * pz)
    t3 = 2.0 * (qx * py - qy * px)
    rx = px + qw * t1 + (qy * t3 - qz * t2)
    ry = py + qw * t2 + (qz * t1 - qx * t3)
    rz = pz + qw * t3 + (qx * t2 - qy * t1)
    return rx, ry, rz


class OccGrid3D:
    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax, res):
        self.xmin = float(xmin); self.xmax = float(xmax)
        self.ymin = float(ymin); self.ymax = float(ymax)
        self.zmin = float(zmin); self.zmax = float(zmax)
        self.res  = float(res)

        self.nx = int(math.ceil((self.xmax - self.xmin) / self.res))
        self.ny = int(math.ceil((self.ymax - self.ymin) / self.res))
        self.nz = int(math.ceil((self.zmax - self.zmin) / self.res))
        if self.nx <= 0 or self.ny <= 0 or self.nz <= 0:
            raise ValueError("Invalid bounds/res -> non-positive grid size.")

        # occ[z,y,x]
        self.occ = np.zeros((self.nz, self.ny, self.nx), dtype=np.uint8)

    def in_bounds(self, x, y, z) -> bool:
        return (self.xmin <= x < self.xmax and
                self.ymin <= y < self.ymax and
                self.zmin <= z < self.zmax)

    def world_to_idx(self, x, y, z) -> Tuple[int, int, int]:
        ix = int((x - self.xmin) / self.res)
        iy = int((y - self.ymin) / self.res)
        iz = int((z - self.zmin) / self.res)
        return ix, iy, iz

    def idx_ok(self, ix, iy, iz) -> bool:
        return (0 <= ix < self.nx and 0 <= iy < self.ny and 0 <= iz < self.nz)

    def is_free(self, x, y, z) -> bool:
        if not self.in_bounds(x, y, z):
            return False
        ix, iy, iz = self.world_to_idx(x, y, z)
        if not self.idx_ok(ix, iy, iz):
            return False
        return self.occ[iz, iy, ix] == 0

    def set_occupied(self, x, y, z):
        if not self.in_bounds(x, y, z):
            return
        ix, iy, iz = self.world_to_idx(x, y, z)
        if self.idx_ok(ix, iy, iz):
            self.occ[iz, iy, ix] = 1

    def segment_free(self, p0, p1, step) -> bool:
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        dist = math.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)
        if dist < 1e-9:
            return self.is_free(x0, y0, z0)

        n = max(2, int(dist / max(step, 1e-3)) + 1)
        for i in range(n + 1):
            a = i / n
            x = x0 + a * (x1 - x0)
            y = y0 + a * (y1 - y0)
            z = z0 + a * (z1 - z0)
            if not self.is_free(x, y, z):
                return False
        return True


@dataclass
class Node3:
    x: float
    y: float
    z: float
    parent: int
    cost: float


class RRTStar3D:
    def __init__(self, occ: OccGrid3D):
        self.occ = occ
        self.edge_step = max(0.5 * occ.res, 0.05)

    def _sample_free(self) -> Tuple[float, float, float]:
        for _ in range(6000):
            x = random.uniform(self.occ.xmin, self.occ.xmax)
            y = random.uniform(self.occ.ymin, self.occ.ymax)
            z = random.uniform(self.occ.zmin, self.occ.zmax)
            if self.occ.is_free(x, y, z):
                return x, y, z
        return (0.5*(self.occ.xmin+self.occ.xmax),
                0.5*(self.occ.ymin+self.occ.ymax),
                0.5*(self.occ.zmin+self.occ.zmax))

    @staticmethod
    def _dist(a, b) -> float:
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

    def _nearest(self, tree: List[Node3], p) -> int:
        d2 = [(n.x-p[0])**2 + (n.y-p[1])**2 + (n.z-p[2])**2 for n in tree]
        return int(np.argmin(d2))

    def _near(self, tree: List[Node3], p, r) -> List[int]:
        r2 = r*r
        px, py, pz = p
        out = []
        for i, n in enumerate(tree):
            if (n.x-px)**2 + (n.y-py)**2 + (n.z-pz)**2 <= r2:
                out.append(i)
        return out

    def _steer(self, from_n: Node3, p, step_len) -> Tuple[float, float, float]:
        dx = p[0]-from_n.x
        dy = p[1]-from_n.y
        dz = p[2]-from_n.z
        d = math.sqrt(dx*dx+dy*dy+dz*dz)
        if d <= step_len:
            return p
        ux, uy, uz = dx/d, dy/d, dz/d
        return (from_n.x + step_len*ux, from_n.y + step_len*uy, from_n.z + step_len*uz)

    def plan(self, start_xyz, goal_xyz,
             *, max_iter=25000, step_len=0.4, goal_sample_rate=0.25,
             search_radius=1.2, goal_tol=0.45) -> Optional[List[Tuple[float, float, float]]]:

        sx, sy, sz = start_xyz
        gx, gy, gz = goal_xyz

        if not self.occ.is_free(sx, sy, sz):
            return None
        if not self.occ.is_free(gx, gy, gz):
            return None

        tree: List[Node3] = [Node3(sx, sy, sz, parent=-1, cost=0.0)]
        goal_idx: Optional[int] = None

        for _ in range(max_iter):
            rp = (gx, gy, gz) if random.random() < goal_sample_rate else self._sample_free()
            ni = self._nearest(tree, rp)
            newp = self._steer(tree[ni], rp, step_len)

            if not self.occ.is_free(*newp):
                continue
            if not self.occ.segment_free((tree[ni].x, tree[ni].y, tree[ni].z), newp, self.edge_step):
                continue

            base = self._dist((tree[ni].x, tree[ni].y, tree[ni].z), newp)
            new = Node3(newp[0], newp[1], newp[2], parent=ni, cost=tree[ni].cost + base)

            near = self._near(tree, newp, search_radius)
            best_parent = new.parent
            best_cost = new.cost

            # choose best parent
            for j in near:
                cand = tree[j]
                if not self.occ.segment_free((cand.x, cand.y, cand.z), newp, self.edge_step):
                    continue
                c = cand.cost + self._dist((cand.x, cand.y, cand.z), newp)
                if c < best_cost:
                    best_cost = c
                    best_parent = j

            new.parent = best_parent
            new.cost = best_cost
            tree.append(new)
            new_idx = len(tree) - 1

            # rewire
            for j in near:
                if j == new.parent:
                    continue
                cand = tree[j]
                if not self.occ.segment_free((new.x, new.y, new.z), (cand.x, cand.y, cand.z), self.edge_step):
                    continue
                c2 = new.cost + self._dist((new.x, new.y, new.z), (cand.x, cand.y, cand.z))
                if c2 < cand.cost:
                    cand.parent = new_idx
                    cand.cost = c2

            # connect to goal
            if self._dist((new.x, new.y, new.z), (gx, gy, gz)) <= goal_tol:
                if self.occ.segment_free((new.x, new.y, new.z), (gx, gy, gz), self.edge_step):
                    tree.append(Node3(gx, gy, gz, parent=new_idx,
                                      cost=new.cost + self._dist((new.x, new.y, new.z), (gx, gy, gz))))
                    goal_idx = len(tree) - 1
                    break

        if goal_idx is None:
            return None

        path = []
        i = goal_idx
        while i != -1:
            n = tree[i]
            path.append((n.x, n.y, n.z))
            i = n.parent
        path.reverse()
        return path


class RRTStarNav6DoF(Node):
    def __init__(self):
        super().__init__("rrtstar_nav_6dof")

        # Frames/topics
        self.declare_parameter("frame_world", "world")
        self.declare_parameter("frame_base", "base_footprint")
        self.declare_parameter("voxel_topic", "/dose_voxels_blocked")
        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("path_topic", "/planned_path")

        # bounds
        self.declare_parameter("xmin", -5.2)
        self.declare_parameter("xmax",  24.5)
        self.declare_parameter("ymin", -6.0)
        self.declare_parameter("ymax",  6.0)
        self.declare_parameter("zmin", -3.0)
        self.declare_parameter("zmax",  3.0)

        self.declare_parameter("res", 0.2)
        self.declare_parameter("stride", 1)

        # planner params
        self.declare_parameter("max_iter", 25000)
        self.declare_parameter("step_len", 0.4)
        self.declare_parameter("search_radius", 1.2)
        self.declare_parameter("goal_sample_rate", 0.25)
        self.declare_parameter("goal_tol", 0.45)
        self.declare_parameter("replan_period_s", 1.0)

        # control
        self.declare_parameter("lookahead", 0.8)
        self.declare_parameter("vxyz_max", 0.25)
        self.declare_parameter("vz_max", 0.12)
        self.declare_parameter("wz_max", 1.0)
        self.declare_parameter("wy_max", 0.6)
        self.declare_parameter("kp_yaw", 2.2)
        self.declare_parameter("kp_pitch", 1.6)
        self.declare_parameter("stop_dist", 0.40)
        self.declare_parameter("pitch_limit_rad", 0.6)  # ~34 deg

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pose_world = None  # (x,y,z, roll,pitch,yaw)
        self.goal_world = None  # (x,y,z)

        self.occ: Optional[OccGrid3D] = None
        self.planner: Optional[RRTStar3D] = None
        self.path: Optional[List[Tuple[float,float,float]]] = None

        self.map_updated = False
        self.goal_updated = False
        self.last_plan_t = 0.0

        self.pub_cmd = self.create_publisher(Twist, self.get_parameter("cmd_vel_topic").value, 10)
        self.pub_path = self.create_publisher(Path, self.get_parameter("path_topic").value, 10)

        self.sub_goal = self.create_subscription(PoseStamped, self.get_parameter("goal_topic").value, self.cb_goal, 10)
        self.sub_vox  = self.create_subscription(PointCloud2,  self.get_parameter("voxel_topic").value, self.cb_vox, 1)

        self.timer = self.create_timer(0.05, self.tick)

        self._last_pose_warn = 0.0
        self.get_logger().info("rrtstar_nav_6dof READY (3D RRT* + blocked voxels).")

    
    def update_pose(self):
        w = self.get_parameter("frame_world").value
        b = self.get_parameter("frame_base").value
        try:
            tr = self.tf_buffer.lookup_transform(w, b, RclpyTime())
            x = float(tr.transform.translation.x)
            y = float(tr.transform.translation.y)
            z = float(tr.transform.translation.z)
            q = tr.transform.rotation
            r, p, yaw = quat_to_rpy(q)
            self.pose_world = (x, y, z, r, p, yaw)
        except Exception:
            self.pose_world = None

            # warn max 1 Hz
            now = time.time()
            if now - self._last_pose_warn > 1.0:
                self._last_pose_warn = now
                self.get_logger().warn(
                    f"No TF pose yet for {w}->{b}. Is cmdvel_6dof_to_tf running and using same frame_world?"
                )

    def cb_goal(self, msg: PoseStamped):
        """
        Preferred: publish goal in frame_world directly (world/map).
        If not, we do a manual TF transform (no tf2_geometry_msgs needed).
        """
        w = self.get_parameter("frame_world").value
        src = msg.header.frame_id if msg.header.frame_id else w

        # fast path: already in world frame
        if (not msg.header.frame_id) or (msg.header.frame_id == w):
            gx = float(msg.pose.position.x)
            gy = float(msg.pose.position.y)
            gz = float(msg.pose.position.z)
            self.goal_world = (gx, gy, gz)
            self.goal_updated = True
            self.path = None
            self.get_logger().info(f"Goal({w}): ({gx:.2f},{gy:.2f},{gz:.2f})")
            return

        # manual transform using TF lookup: w <- src
        try:
            tr = self.tf_buffer.lookup_transform(w, src, RclpyTime())
            tx = float(tr.transform.translation.x)
            ty = float(tr.transform.translation.y)
            tz = float(tr.transform.translation.z)

            q = tr.transform.rotation
            qx, qy, qz, qw = float(q.x), float(q.y), float(q.z), float(q.w)

            px = float(msg.pose.position.x)
            py = float(msg.pose.position.y)
            pz = float(msg.pose.position.z)

            rx, ry, rz = rotate_point_by_quat(px, py, pz, qx, qy, qz, qw)
            gx, gy, gz = rx + tx, ry + ty, rz + tz

            self.goal_world = (gx, gy, gz)
            self.goal_updated = True
            self.path = None
            self.get_logger().info(f"Goal({w}) from {src}: ({gx:.2f},{gy:.2f},{gz:.2f})")
        except Exception as e:
            self.get_logger().error(
                f"Goal TF failed. Publish goal in frame '{w}'. src='{src}' err={e}"
            )

    def cb_vox(self, msg: PointCloud2):
        res   = float(self.get_parameter("res").value)
        xmin  = float(self.get_parameter("xmin").value); xmax = float(self.get_parameter("xmax").value)
        ymin  = float(self.get_parameter("ymin").value); ymax = float(self.get_parameter("ymax").value)
        zmin  = float(self.get_parameter("zmin").value); zmax = float(self.get_parameter("zmax").value)
        stride = max(1, int(self.get_parameter("stride").value))

        occ = OccGrid3D(xmin, xmax, ymin, ymax, zmin, zmax, res)

        used = 0
        i = 0
        for p in pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True):
            i += 1
            if (i % stride) != 0:
                continue
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            occ.set_occupied(x, y, z)
            used += 1

        self.occ = occ
        self.planner = RRTStar3D(self.occ)
        self.map_updated = True

        occ_ratio = float(np.mean(self.occ.occ > 0))
        self.get_logger().info(
            f"Occ3D updated: used={used} stride={stride} grid={occ.nx}x{occ.ny}x{occ.nz} res={res:.2f} "
            f"occ_ratio={occ_ratio*100:.2f}%"
        )

  
    def publish_path(self, path_xyz: List[Tuple[float,float,float]]):
        w = self.get_parameter("frame_world").value
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = w
        for x,y,z in path_xyz:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = float(z)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.pub_path.publish(msg)

    def plan_if_needed(self):
        if self.pose_world is None or self.goal_world is None:
            return
        if self.occ is None or self.planner is None:
            return

        now = time.time()
        if now - self.last_plan_t < float(self.get_parameter("replan_period_s").value):
            return

        # only replan if new goal/map OR no path yet
        if (self.path is not None) and (not self.goal_updated) and (not self.map_updated):
            return

        self.last_plan_t = now
        self.goal_updated = False
        self.map_updated = False

        sx, sy, sz, *_ = self.pose_world
        gx, gy, gz = self.goal_world

        if not self.occ.is_free(sx, sy, sz):
            self.get_logger().warn("Start blocked/out-of-bounds in 3D.")
            return
        if not self.occ.is_free(gx, gy, gz):
            self.get_logger().warn("Goal blocked/out-of-bounds in 3D.")
            return

        path = self.planner.plan(
            (sx,sy,sz), (gx,gy,gz),
            max_iter=int(self.get_parameter("max_iter").value),
            step_len=float(self.get_parameter("step_len").value),
            search_radius=float(self.get_parameter("search_radius").value),
            goal_sample_rate=float(self.get_parameter("goal_sample_rate").value),
            goal_tol=float(self.get_parameter("goal_tol").value),
        )

        if path is None:
            self.get_logger().warn("RRT*: no 3D path.")
            return

        self.path = path
        self.get_logger().info(f"RRT*: 3D path with {len(path)} points.")
        self.publish_path(path)

    def follow(self):
        if self.pose_world is None or self.goal_world is None or not self.path:
            return

        x, y, z, _r, pitch, yaw = self.pose_world
        gx, gy, gz = self.goal_world

        stop_dist = float(self.get_parameter("stop_dist").value)
        if math.sqrt((gx-x)**2 + (gy-y)**2 + (gz-z)**2) < stop_dist:
            self.pub_cmd.publish(Twist())
            return

        # pick lookahead target
        lookahead = float(self.get_parameter("lookahead").value)
        tx, ty, tz = self.path[-1]
        for px, py, pz in self.path:
            if math.sqrt((px-x)**2 + (py-y)**2 + (pz-z)**2) >= lookahead:
                tx, ty, tz = px, py, pz
                break

        dx, dy, dz = (tx-x), (ty-y), (tz-z)
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist < 1e-6:
            self.pub_cmd.publish(Twist())
            return

        # desired yaw/pitch
        yaw_des = math.atan2(dy, dx)
        pitch_des = math.atan2(-dz, math.hypot(dx, dy))  # dz up => pitch negative to go down

        # clamp desired pitch to avoid flips
        pitch_lim = float(self.get_parameter("pitch_limit_rad").value)
        pitch_des = clamp(pitch_des, -pitch_lim, pitch_lim)

        # yaw control
        err_yaw = wrap_pi(yaw_des - yaw)
        kp_yaw = float(self.get_parameter("kp_yaw").value)
        wz_max = float(self.get_parameter("wz_max").value)
        wz = clamp(kp_yaw * err_yaw, -wz_max, wz_max)

        # pitch control (ERROR, not absolute)
        err_pitch = clamp(pitch_des - pitch, -pitch_lim, pitch_lim)
        kp_pitch = float(self.get_parameter("kp_pitch").value)
        wy_max = float(self.get_parameter("wy_max").value)
        wy = clamp(kp_pitch * err_pitch, -wy_max, wy_max)

        # linear velocity towards target (world direction)
        vxyz_max = float(self.get_parameter("vxyz_max").value)
        vz_max = float(self.get_parameter("vz_max").value)

        ux, uy, uz = dx/dist, dy/dist, dz/dist
        v = vxyz_max * max(0.0, 1.0 - abs(err_yaw)/1.4)  # slow down if yaw error big

        vx_w = v * ux
        vy_w = v * uy
        vz_w = clamp(v * uz, -vz_max, vz_max)

        # Convert world linear to body linear (assume cmdvel_6dof_to_tf expects BODY-frame linear)
        cy = math.cos(yaw); sy = math.sin(yaw)
        vx_b =  cy*vx_w + sy*vy_w
        vy_b = -sy*vx_w + cy*vy_w
        vz_b = vz_w

        cmd = Twist()
        cmd.linear.x = float(vx_b)
        cmd.linear.y = float(vy_b)
        cmd.linear.z = float(vz_b)

        cmd.angular.x = 0.0        # roll locked
        cmd.angular.y = float(wy)  # pitch rate
        cmd.angular.z = float(wz)  # yaw rate
        self.pub_cmd.publish(cmd)

    def tick(self):
        self.update_pose()
        self.plan_if_needed()
        self.follow()


def main():
    rclpy.init()
    node = RRTStarNav6DoF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.pub_cmd.publish(Twist())
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()