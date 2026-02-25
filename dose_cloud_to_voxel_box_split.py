
#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node

from std_msgs.msg import Header
from geometry_msgs.msg import Point, Pose
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2


class DoseCloudToVoxelBoxSplit(Node):
    
    def __init__(self):
        super().__init__("dose_cloud_to_voxel_box_split")

        self.declare_parameter("cloud_topic", "/dose_cloud_combined")
        self.declare_parameter("blocked_topic", "/dose_voxels_blocked")
        self.declare_parameter("box_marker_topic", "/dose_box")
        self.declare_parameter("frame_id", "world")

        self.declare_parameter("voxel", 0.2)

        # 3D box bounds (meters)
        self.declare_parameter("xmin", -5.2)
        self.declare_parameter("xmax",  24.5)
        self.declare_parameter("ymin", -3.0)
        self.declare_parameter("ymax",  3.0)
        self.declare_parameter("zmin",  -6.0)
        self.declare_parameter("zmax",   6.0)

        # split plane along X
        self.declare_parameter("split_x", 15.0)

        # thresholds (<=0 disables that check)
        self.declare_parameter("core_dose_thresh", 1.0)
        self.declare_parameter("spent_dose_thresh", 1.0)
        self.declare_parameter("core_temp_thresh", -1.0)
        self.declare_parameter("spent_temp_thresh", -1.0)

        # reduce publish size
        self.declare_parameter("max_blocked_points", 250000)

        self.cloud_topic = self.get_parameter("cloud_topic").value
        self.blocked_topic = self.get_parameter("blocked_topic").value
        self.box_topic = self.get_parameter("box_marker_topic").value
        self.frame_id = self.get_parameter("frame_id").value

        self.pub_blocked = self.create_publisher(PointCloud2, self.blocked_topic, 1)
        self.pub_box = self.create_publisher(Marker, self.box_topic, 1)

        self.sub = self.create_subscription(PointCloud2, self.cloud_topic, self.cb, 1)

        self.fields_out = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        self.get_logger().info(f"Listening: {self.cloud_topic}")
        self.get_logger().info(f"Publishing blocked voxels: {self.blocked_topic} | box marker: {self.box_topic}")

        # publish box marker at 2 Hz
        self.timer = self.create_timer(0.5, self.publish_box)

        self.last_blocked = None  # cache last blocked points (Nx3 float32)

    def publish_box(self):
        xmin = float(self.get_parameter("xmin").value)
        xmax = float(self.get_parameter("xmax").value)
        ymin = float(self.get_parameter("ymin").value)
        ymax = float(self.get_parameter("ymax").value)
        zmin = float(self.get_parameter("zmin").value)
        zmax = float(self.get_parameter("zmax").value)

        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "dose_box"
        m.id = 1
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = 0.5 * (xmin + xmax)
        m.pose.position.y = 0.5 * (ymin + ymax)
        m.pose.position.z = 0.5 * (zmin + zmax)
        m.pose.orientation.w = 1.0
        m.scale.x = (xmax - xmin)
        m.scale.y = (ymax - ymin)
        m.scale.z = (zmax - zmin)
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 1.0
        m.color.a = 0.10  # translucent
        self.pub_box.publish(m)

        
    def cb(self, msg: PointCloud2):
        names = [f.name for f in msg.fields]
        low = {n.lower(): n for n in names}

        need = ["x", "y", "z"]
        for n in need:
            if n not in low:
                self.get_logger().error(f"Cloud missing '{n}' field. Fields: {names}")
                return
        if "dose" not in low:
            self.get_logger().error(f"Cloud missing numeric 'dose' field. Fields: {names}")
            return

        dose_name = low["dose"]
        temp_name = low.get("temp", None)

        voxel = float(self.get_parameter("voxel").value)

        xmin = float(self.get_parameter("xmin").value)
        xmax = float(self.get_parameter("xmax").value)
        ymin = float(self.get_parameter("ymin").value)
        ymax = float(self.get_parameter("ymax").value)
        zmin = float(self.get_parameter("zmin").value)
        zmax = float(self.get_parameter("zmax").value)

        split_x = float(self.get_parameter("split_x").value)

        core_dose = float(self.get_parameter("core_dose_thresh").value)
        spent_dose = float(self.get_parameter("spent_dose_thresh").value)
        core_temp = float(self.get_parameter("core_temp_thresh").value)
        spent_temp = float(self.get_parameter("spent_temp_thresh").value)

        max_pts = int(self.get_parameter("max_blocked_points").value)

        # voxel index bounds
        nx = int(math.ceil((xmax - xmin) / voxel))
        ny = int(math.ceil((ymax - ymin) / voxel))
        nz = int(math.ceil((zmax - zmin) / voxel))

        if nx <= 0 or ny <= 0 or nz <= 0:
            self.get_logger().error("Invalid voxel grid dimensions. Check bounds and voxel size.")
            return

        blocked = np.zeros((nz, ny, nx), dtype=np.uint8)

        # Decide read fields
        if temp_name is not None:
            fields = ("x", "y", "z", dose_name, temp_name)
        else:
            fields = ("x", "y", "z", dose_name)

        # Fill occupancy
        count_in = 0
        for p in pc2.read_points(msg, field_names=fields, skip_nans=True):
            if temp_name is not None:
                x, y, z, dose, temp = float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
            else:
                x, y, z, dose = float(p[0]), float(p[1]), float(p[2]), float(p[3])
                temp = -1.0

            # inside box only
            if x < xmin or x >= xmax or y < ymin or y >= ymax or z < zmin or z >= zmax:
                continue

            count_in += 1

            is_core = (x < split_x)
            dth = core_dose if is_core else spent_dose
            tth = core_temp if is_core else spent_temp

            hot = False
            if dth > 0.0 and dose >= dth:
                hot = True
            if tth > 0.0 and temp >= tth:
                hot = True
            if not hot:
                continue

            ix = int((x - xmin) / voxel)
            iy = int((y - ymin) / voxel)
            iz = int((z - zmin) / voxel)
            if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                blocked[iz, iy, ix] = 1

        # Extract blocked voxel centers
        idx = np.argwhere(blocked > 0)
        if idx.size == 0:
            self.last_blocked = np.zeros((0, 3), dtype=np.float32)
            self.get_logger().info(f"Voxelized: points_in_box={count_in} blocked_voxels=0")
            return

        pts = np.zeros((idx.shape[0], 3), dtype=np.float32)
        # iz,iy,ix -> center coordinate
        pts[:, 0] = xmin + (idx[:, 2].astype(np.float32) + 0.5) * voxel
        pts[:, 1] = ymin + (idx[:, 1].astype(np.float32) + 0.5) * voxel
        pts[:, 2] = zmin + (idx[:, 0].astype(np.float32) + 0.5) * voxel

        # limit points if huge
        if pts.shape[0] > max_pts:
            keep = np.random.choice(pts.shape[0], size=max_pts, replace=False)
            pts = pts[keep]

        self.last_blocked = pts

        # publish
        hdr = Header()
        hdr.stamp = self.get_clock().now().to_msg()
        hdr.frame_id = self.frame_id
        out = pc2.create_cloud(hdr, self.fields_out, pts.tolist())
        self.pub_blocked.publish(out)

        self.get_logger().info(
            f"Voxelized: points_in_box={count_in} blocked_voxels={idx.shape[0]} published={pts.shape[0]} "
            f"| split_x={split_x:.2f} voxel={voxel:.2f}"
        )


def main():
    rclpy.init()
    node = DoseCloudToVoxelBoxSplit()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()