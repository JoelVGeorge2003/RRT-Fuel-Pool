
#!/usr/bin/env python3


import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from tf2_ros import TransformBroadcaster


def quat_mul(q1, q2):
    # q = q1*q2, both as (x,y,z,w)
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def quat_norm(q):
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / n, y / n, z / n, w / n)


def quat_from_omega_dt(wx, wy, wz, dt):
    # small-angle exponential map: rotation magnitude = |w|*dt
    ang = math.sqrt(wx * wx + wy * wy + wz * wz) * dt
    if ang < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    ax = wx / math.sqrt(wx * wx + wy * wy + wz * wz)
    ay = wy / math.sqrt(wx * wx + wy * wy + wz * wz)
    az = wz / math.sqrt(wx * wx + wy * wy + wz * wz)
    s = math.sin(ang * 0.5)
    c = math.cos(ang * 0.5)
    return (ax * s, ay * s, az * s, c)


def rotate_vec_by_quat(v, q):
    # rotate vector v (x,y,z) by quaternion q (x,y,z,w)
    vx, vy, vz = v
    qx, qy, qz, qw = q
    # v' = q * (v,0) * q_conj
    # optimized form
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    vxp = vx + qw * tx + (qy * tz - qz * ty)
    vyp = vy + qw * ty + (qz * tx - qx * tz)
    vzp = vz + qw * tz + (qx * ty - qy * tx)
    return (vxp, vyp, vzp)


class CmdVel6DoFToTF(Node):
    def __init__(self):
        super().__init__("cmdvel_6dof_to_tf")

        self.declare_parameter("frame_world", "map")
        self.declare_parameter("frame_base", "base_footprint")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("publish_hz", 50.0)

        self.declare_parameter("max_dt", 0.05)          # clamp dt
        self.declare_parameter("vel_lin_max", 2.0)      # m/s clamp magnitude
        self.declare_parameter("vel_ang_max", 6.0)      # rad/s clamp magnitude

        self.frame_world = self.get_parameter("frame_world").value
        self.frame_base = self.get_parameter("frame_base").value
        self.cmd_topic = self.get_parameter("cmd_vel_topic").value
        self.hz = float(self.get_parameter("publish_hz").value)

        self.max_dt = float(self.get_parameter("max_dt").value)
        self.lin_max = float(self.get_parameter("vel_lin_max").value)
        self.ang_max = float(self.get_parameter("vel_ang_max").value)

        self.br = TransformBroadcaster(self)
        self.sub = self.create_subscription(Twist, self.cmd_topic, self.cb_cmd, 10)

        # state: position in world, orientation as quaternion world->base
        self.p = [0.0, 0.0, 0.0]
        self.q = (0.0, 0.0, 0.0, 1.0)  # (x,y,z,w)

        # last commanded twist (body frame)
        self.vb = [0.0, 0.0, 0.0]
        self.wb = [0.0, 0.0, 0.0]

        # quaternion sign continuity
        self.prev_q = self.q

        self.last_t = self.get_clock().now()
        self.timer = self.create_timer(1.0 / max(1.0, self.hz), self.tick)

        self.get_logger().info(
            f"cmdvel_6dof_to_tf: publishing TF {self.frame_world}->{self.frame_base} from {self.cmd_topic}"
        )

    def cb_cmd(self, msg: Twist):
        vx, vy, vz = float(msg.linear.x), float(msg.linear.y), float(msg.linear.z)
        wx, wy, wz = float(msg.angular.x), float(msg.angular.y), float(msg.angular.z)

        # clamp magnitudes (simple)
        vmag = math.sqrt(vx * vx + vy * vy + vz * vz)
        if vmag > self.lin_max and vmag > 1e-9:
            s = self.lin_max / vmag
            vx, vy, vz = vx * s, vy * s, vz * s

        wmag = math.sqrt(wx * wx + wy * wy + wz * wz)
        if wmag > self.ang_max and wmag > 1e-9:
            s = self.ang_max / wmag
            wx, wy, wz = wx * s, wy * s, wz * s

        self.vb = [vx, vy, vz]
        self.wb = [wx, wy, wz]

    def tick(self):
        now = self.get_clock().now()
        dt = (now - self.last_t).nanoseconds * 1e-9
        self.last_t = now

        # clamp dt (prevents huge jumps if timer hiccups)
        if dt <= 0.0:
            dt = 1.0 / max(1.0, self.hz)
        dt = min(dt, self.max_dt)

        # integrate orientation: q <- q * dq  (body rates)
        dq = quat_from_omega_dt(self.wb[0], self.wb[1], self.wb[2], dt)
        self.q = quat_norm(quat_mul(self.q, dq))

        # enforce quaternion sign continuity (prevents yaw/pitch/roll "flip")
        if sum(a * b for a, b in zip(self.q, self.prev_q)) < 0.0:
            self.q = tuple(-x for x in self.q)
        self.prev_q = self.q

        # integrate position: body linear -> world linear via current q
        vw = rotate_vec_by_quat(tuple(self.vb), self.q)  # rotate body vel into world
        self.p[0] += vw[0] * dt
        self.p[1] += vw[1] * dt
        self.p[2] += vw[2] * dt

        t = TransformStamped()
        t.header.stamp = now.to_msg()
        t.header.frame_id = self.frame_world
        t.child_frame_id = self.frame_base
        t.transform.translation.x = float(self.p[0])
        t.transform.translation.y = float(self.p[1])
        t.transform.translation.z = float(self.p[2])
        t.transform.rotation.x = float(self.q[0])
        t.transform.rotation.y = float(self.q[1])
        t.transform.rotation.z = float(self.q[2])
        t.transform.rotation.w = float(self.q[3])

        self.br.sendTransform(t)


def main():
    rclpy.init()
    node = CmdVel6DoFToTF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()