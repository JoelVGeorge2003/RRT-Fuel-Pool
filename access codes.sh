
keyboard control

cd ~/uuv_fresh_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run uuv_fresh teleop_3d_keyboard



launch robot

cd ~/uuv_fresh_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch uuv_fresh view_drive.launch.py



combined cloud

cd ~/uuv_fresh_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run uuv_fresh dose_cloud_combined_csv_pub --ros-args   -p core_csv:=/home/joel/ros2_ws/core_gamma_temp.csv   -p spent_csv:=/home/joel/ros2_ws/spentfuel_gamma_temp.csv   -p topic:=/dose_cloud_combined   -p frame_id:=world   -p xyz_scale:=0.01   -p spent_start_mode:=after_core   -p gap_m:=1.0 



voxel split

cd ~/uuv_fresh_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run uuv_fresh dose_cloud_to_voxel_box_split --ros-args   -p cloud_topic:=/dose_cloud_combined   -p frame_id:=world   -p voxel:=0.20   -p split_x:=15.0   -p core_dose_thresh:=3.979390e-32   -p spent_dose_thresh:=1.956586e-32   -p core_temp_thresh:=55.0   -p spent_temp_thresh:=55.0



automation

cd ~/uuv_fresh_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run uuv_fresh rrtstar_nav_6dof --ros-args -p frame_world:=world



goal setting

cd ~/uuv_fresh_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 topic pub /goal_pose geometry_msgs/msg/PoseStamped "{
  header: {frame_id: world},
  pose: {position: {x: -5.0, y: -1.0, z: -1.0}, orientation: {w: 1.0}}
}" -1
