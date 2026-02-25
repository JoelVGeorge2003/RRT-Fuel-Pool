
#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2


def _pick(names, options):
    low = {n.lower(): n for n in names}
    for opt in options:
        if opt.lower() in low:
            return low[opt.lower()]
    return None


def _rgba_from_unit(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    r = np.zeros_like(u); g = np.zeros_like(u); b = np.zeros_like(u)

    m1 = u < 1/3
    t = u[m1] * 3.0
    g[m1] = t
    b[m1] = 1.0 - t

    m2 = (u >= 1/3) & (u < 2/3)
    t = (u[m2] - 1/3) * 3.0
    r[m2] = t
    g[m2] = 1.0

    m3 = u >= 2/3
    t = (u[m3] - 2/3) * 3.0
    r[m3] = 1.0
    g[m3] = 1.0 - t

    R = (r * 255).astype(np.uint32)
    G = (g * 255).astype(np.uint32)
    B = (b * 255).astype(np.uint32)
    A = np.full_like(R, 255, dtype=np.uint32)
    rgba_u32 = (A << 24) | (R << 16) | (G << 8) | B
    return rgba_u32.astype(np.uint32)


def _load_csv_points(csv_path: str, xyz_scale: float) -> np.ndarray:
    
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    cols = list(data.dtype.names)

    cx = _pick(cols, ["x_cm", "x"])
    cy = _pick(cols, ["y_cm", "y"])
    cz = _pick(cols, ["z_cm", "z"])
    cd = _pick(cols, ["dose_gy_per_source_particle", "dose", "dose_gy"])
    ct = _pick(cols, ["temp_c", "temp"])

    if cx is None or cy is None or cz is None or cd is None or ct is None:
        raise RuntimeError(f"{csv_path}: missing columns. Found: {cols}")

    x = data[cx].astype(np.float32) * xyz_scale
    y = data[cy].astype(np.float32) * xyz_scale
    z = data[cz].astype(np.float32) * xyz_scale
    dose = data[cd].astype(np.float64)
    temp = data[ct].astype(np.float32)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(dose) & np.isfinite(temp)
    x, y, z, dose, temp = x[m], y[m], z[m], dose[m], temp[m]

    # color by log-dose for visibility
    eps = 1e-30
    val = np.log10(np.maximum(dose, eps))
    vmin = float(np.min(val)) if len(val) else 0.0
    vmax = float(np.max(val)) if len(val) else 1.0
    u = np.zeros_like(val, dtype=np.float32) if math.isclose(vmin, vmax) else ((val - vmin) / (vmax - vmin)).astype(np.float32)

    rgba_u32 = _rgba_from_unit(u)
    rgba_f32 = rgba_u32.view(np.float32)

    pts = np.zeros((len(x), 6), dtype=np.float32)
    pts[:, 0] = x
    pts[:, 1] = y
    pts[:, 2] = z
    pts[:, 3] = rgba_f32
    pts[:, 4] = dose.astype(np.float32)
    pts[:, 5] = temp
    return pts


class CombinedDoseCloudCsvPublisher(Node):
    
    def __init__(self):
        super().__init__("combined_dose_cloud_csv_publisher")

        self.declare_parameter("core_csv", "/home/joel/ros2_ws/core_gamma_temp.csv")
        self.declare_parameter("spent_csv", "/home/joel/ros2_ws/spentfuel_gamma_temp.csv")

        self.declare_parameter("topic", "/dose_cloud_combined")
        self.declare_parameter("frame_id", "world")
        self.declare_parameter("publish_hz", 1.0)

        self.declare_parameter("xyz_scale", 0.01)  # cm->m

        # Placement control (THIS is what you asked for)
        self.declare_parameter("spent_start_mode", "after_core")  # after_core | fixed
        self.declare_parameter("gap_m", 1.0)                      # used by after_core
        self.declare_parameter("spent_start_x", 15.2)             # used by fixed
        self.declare_parameter("y_offset", -3)   # meters
        core_csv = self.get_parameter("core_csv").value
        spent_csv = self.get_parameter("spent_csv").value
        topic = self.get_parameter("topic").value
        self.frame_id = self.get_parameter("frame_id").value
        hz = float(self.get_parameter("publish_hz").value)
        xyz_scale = float(self.get_parameter("xyz_scale").value)

        self.core = _load_csv_points(core_csv, xyz_scale)
        self.spent = _load_csv_points(spent_csv, xyz_scale)
        y_offset = float(self.get_parameter("y_offset").value)
        self.core[:, 1] += y_offset
        self.spent[:, 1] += y_offset

        self._place_spent()

        self.pub = self.create_publisher(PointCloud2, topic, 10)

        self.fields = [
            PointField(name="x",    offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y",    offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z",    offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name="rgba", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="dose", offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name="temp", offset=20, datatype=PointField.FLOAT32, count=1),
        ]

        self.timer = self.create_timer(1.0 / max(hz, 0.1), self.tick)

        self.get_logger().info(f"Loaded core  pts: {len(self.core)}")
        self.get_logger().info(f"Loaded spent pts: {len(self.spent)}")
        self.get_logger().info(f"Publishing combined cloud: {topic} frame={self.frame_id}")

    def _place_spent(self):
        mode = str(self.get_parameter("spent_start_mode").value)
        gap = float(self.get_parameter("gap_m").value)
        spent_start_x = float(self.get_parameter("spent_start_x").value)

        core_minx, core_maxx = float(np.min(self.core[:, 0])), float(np.max(self.core[:, 0]))
        spent_minx = float(np.min(self.spent[:, 0]))

        if mode == "fixed":
            target_start = spent_start_x
        else:  # after_core
            target_start = core_maxx + gap

        dx = target_start - spent_minx
        self.spent[:, 0] += dx  # shift spent along +X only

        self.split_x = core_maxx + 0.5 * gap  # a useful boundary for the map node
        self.get_logger().info(f"Spent placed: mode={mode} dx={dx:.3f} m | core_x=[{core_minx:.2f},{core_maxx:.2f}] spent_start={target_start:.2f} split_x~{self.split_x:.2f}")

    def tick(self):
        # allow live updates of placement parameters by restarting node (simple & reliable)
        pts = np.vstack([self.core, self.spent])

        hdr = Header()
        hdr.stamp = self.get_clock().now().to_msg()
        hdr.frame_id = self.frame_id

        msg = pc2.create_cloud(hdr, self.fields, pts.tolist())
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = CombinedDoseCloudCsvPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()