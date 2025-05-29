import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Int16
from visualization_msgs.msg import MarkerArray, Marker
from rcl_interfaces.srv import SetParameters
from custom_interfaces.srv import SwitchPublisher
from rclpy.parameter import Parameter as RclpyParameter
import json
import os
import yaml
import subprocess
import threading
import time

class TrajectoryCommander(Node):
    def __init__(self):
        super().__init__('trajectory_commander')

        self.current_goal = None
        self.saved_trajectory = []
        self.data_path = "src/msr_simulation/data/trajectories"

        os.makedirs(self.data_path, exist_ok=True)

        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/trajectory_markers', 10)
        self.line_follow_pub = self.create_publisher(Int16, '/line_follow_enable', 10)
        self.detection_fsm_pub = self.create_publisher(Int16, '/detection_fsm_enable', 10)

        self.set_param_cli = self.create_client(SetParameters, '/inverse_kinematics/set_parameters')
        self.switch_cmd_cli = self.create_client(SwitchPublisher, '/switch_cmd_source')

        # Wait for services
        self.set_param_cli.wait_for_service(timeout_sec=5.0)
        self.switch_cmd_cli.wait_for_service(timeout_sec=5.0)

        self.detection_fsm_msg = Int16()
        self.detection_fsm_msg.data = 0

        self.line_follow_msg = Int16()
        self.line_follow_msg.data = 0

    def goal_callback(self, msg: PoseStamped):
        orientation_q = msg.pose.orientation
        yaw = self.get_yaw_from_quaternion(orientation_q)
        self.current_goal = {
            'x': msg.pose.position.x,
            'y': msg.pose.position.y,
            'yaw': yaw
        }

        # Log the received goal
        self.get_logger().info(
            f"Received goal pose: x={self.current_goal['x']:.3f}, y={self.current_goal['y']:.3f}, yaw={self.current_goal['yaw']:.3f} rad"
        )


    def get_yaw_from_quaternion(self, q):
        import math
        # Assuming quaternion is normalized
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def menu_loop(self):
        while rclpy.ok():
            print("""
Trajectory Commander - Menu:
1. Send next goal pose to robot 🎯
2. Save trajectory 💾
3. Send specific trajectory 🔄
4. Toggle IK/Teleop/Line 🎮
5. Toggle Line Following Mode 🛣️
6. Toggle Semaphore Detection FSM 🚦
-1. Exit
""")
            choice = input("Enter your choice: ")
            if choice == '1':
                self.send_next_goal()
            elif choice == '2':
                self.save_trajectory()
            elif choice == '3':
                self.send_specific_trajectory()
            elif choice == '4':
                self.toggle_cmd_source()
            elif choice == '5':
                self.toggle_line_following()
            elif choice == '6':
                self.toggle_detection_fsm()

            elif choice == '-1':
                print("Exiting...")
                break
            else:
                print("Invalid option.")

    def send_next_goal(self):
        print("Waiting for next goal_pose message...")
        self.current_goal = None
        start_time = time.time()
        timeout = 10.0  # seconds to wait for a goal

        while self.current_goal is None and (time.time() - start_time) < timeout and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.current_goal is None:
            print("Timeout waiting for new goal_pose! ❌")
            return

        # Once received, send it
        params = [
            RclpyParameter('desired_x', RclpyParameter.Type.DOUBLE, self.current_goal['x']).to_parameter_msg(),
            RclpyParameter('desired_y', RclpyParameter.Type.DOUBLE, self.current_goal['y']).to_parameter_msg(),
            RclpyParameter('desired_yaw', RclpyParameter.Type.DOUBLE, self.current_goal['yaw']).to_parameter_msg()
        ]

        req = SetParameters.Request()
        req.parameters = params

        future = self.set_param_cli.call_async(req)

        while not future.done():
            rclpy.spin_once(self, timeout_sec=0.1)

        try:
            result = future.result()
            print("Sent freshly received goal successfully ✅")
        except Exception as e:
            print(f"Failed to send goal: {e}")

    def save_trajectory(self):
        try:
            n_points = int(input("How many points to save? "))
        except ValueError:
            print("Invalid number.")
            return

        self.saved_trajectory = []  # Clear previous trajectory

        print(f"Waiting for {n_points} goal poses... Publish each goal when ready.")

        collected = 0
        timeout = 30.0  # optional: timeout seconds for each point

        while collected < n_points and rclpy.ok():
            previous_goal = self.current_goal  # Remember last known goal
            start_time = time.time()

            # Wait for a NEW goal different from the previous one
            while (self.current_goal == previous_goal) and (time.time() - start_time < timeout) and rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.1)

            if self.current_goal != previous_goal:
                self.saved_trajectory.append(self.current_goal.copy())
                collected += 1
                print(f"✅ Saved point {collected}: {self.current_goal}")
            else:
                print("⚠️ Timeout waiting for new goal. Skipping...")
                break

        if collected < n_points:
            print(f"Only collected {collected} points. Saving what was received.")

        # Ask for filename
        filename = input("Enter filename to save (without .yaml): ")
        filepath = os.path.join(self.data_path, f"{filename}.yaml")

        # Create YAML dictionary
        yaml_data = {
            '/path_generator': {
                'ros__parameters': {
                    'sample_time': 0.018,  # or read dynamically if you want
                    'waypoints_json': json.dumps(self.saved_trajectory, indent=4)
                }
            }
        }

        # Save to YAML
        with open(filepath, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False)

        print(f"Trajectory saved to {filepath} ✅")


    def send_specific_trajectory(self):
        files = [f for f in os.listdir(self.data_path) if f.endswith('.yaml')]
        if not files:
            print("No trajectory YAML files found.")
            return

        print("Available trajectories:")
        for idx, f in enumerate(files):
            print(f"[{idx}] {f}")

        try:
            choice = int(input("Enter the index of the trajectory to load: "))
            if choice < 0 or choice >= len(files):
                print("Invalid index selected.")
                return
        except ValueError:
            print("Invalid input. Please enter a number.")
            return

        selected_file = files[choice]
        filepath = os.path.join(self.data_path, selected_file)

        # Try to read the file contents to get the trajectory points
        try:
            with open(filepath, 'r') as f:
                yaml_data = yaml.safe_load(f)

            waypoints_json_str = yaml_data['/path_generator']['ros__parameters']['waypoints_json']
            trajectory = json.loads(waypoints_json_str)

        except Exception as e:
            print(f"Failed to read trajectory for visualization: {e}")
            trajectory = None  # Fallback in case of visualization error

        # Now Call ros2 param load
        try:
            cmd = ['ros2', 'param', 'load', '/path_generator', filepath]
            print(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            # Print command output
            print(result.stdout)
            if result.returncode != 0:
                print(f"Error loading parameters:\n{result.stderr}")
            else:
                print(f"Trajectory {selected_file} successfully sent to path_generator ✅")

        except Exception as e:
            print(f"Failed to load trajectory with ros2 param load: {e}")
            return  # If load failed, don't continue to visualization

        # Visualize the trajectory if it was parsed correctly
        if trajectory:
            markers = MarkerArray()

            # Line strip marker
            line_marker = Marker()
            line_marker.header.frame_id = 'map'
            line_marker.header.stamp = self.get_clock().now().to_msg()
            line_marker.ns = 'trajectory_line'
            line_marker.id = 0
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            line_marker.scale.x = 0.05  # Thickness
            line_marker.color.r = 1.0
            line_marker.color.g = 0.0
            line_marker.color.b = 1.0
            line_marker.color.a = 1.0
            line_marker.pose.orientation.w = 1.0

            for pose in trajectory:
                p = Point()
                p.x = pose['x']
                p.y = pose['y']
                p.z = 0.0
                line_marker.points.append(p)

            markers.markers.append(line_marker)

            # Optionally: also add small spheres at each point (optional nice visualization)
            for i, pose in enumerate(trajectory):
                sphere_marker = Marker()
                sphere_marker.header.frame_id = 'map'
                sphere_marker.header.stamp = self.get_clock().now().to_msg()
                sphere_marker.ns = 'trajectory_points'
                sphere_marker.id = i + 1
                sphere_marker.type = Marker.SPHERE
                sphere_marker.action = Marker.ADD
                sphere_marker.pose.position.x = pose['x']
                sphere_marker.pose.position.y = pose['y']
                sphere_marker.pose.position.z = 0.0
                sphere_marker.scale.x = 0.05
                sphere_marker.scale.y = 0.05
                sphere_marker.scale.z = 0.05
                sphere_marker.color.r = 0.0
                sphere_marker.color.g = 1.0
                sphere_marker.color.b = 0.0
                sphere_marker.color.a = 1.0
                sphere_marker.pose.orientation.w = 1.0

                markers.markers.append(sphere_marker)

            self.marker_pub.publish(markers)
            print("Trajectory markers published to RViz ✅")

    def toggle_line_following(self):
        print("Toggling line follower...")

        if self.line_follow_msg.data == 1:
            # Deactivate
            self.line_follow_msg.data = 0
            self.line_follow_pub.publish(self.line_follow_msg)
            print("🛑 Line follower deactivated")
        else:
            # Step 1: Switch command source to 'line'
            req = SwitchPublisher.Request()
            req.active_source = 'line'
            future = self.switch_cmd_cli.call_async(req)
            while not future.done() and rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.1)

            try:
                result = future.result()
                print("✅ Switched command source to 'line'")
            except Exception as e:
                print(f"❌ Failed to switch to 'line': {e}")
                return

            # Step 2: Activate line follower
            self.line_follow_msg.data = 1
            self.line_follow_pub.publish(self.line_follow_msg)
            print("✅ Line follower activated")

    def toggle_detection_fsm(self):
        print("Toggle Semaphore detection ...")

        # Step 2: Send a message to /fsm_action to activate line follower
        if (self.detection_fsm_msg.data == 1):
            self.detection_fsm_msg.data = 0  # deactivate
            self.detection_fsm_pub.publish(self.detection_fsm_msg)
            print("🛑 Detection FSM deactivated ")
        else:
            self.detection_fsm_msg.data = 1  # activate
            self.detection_fsm_pub.publish(self.detection_fsm_msg)
            print("✅ Detection FSM activated ")


    def toggle_cmd_source(self):
        mode = input("Enter mode (ik/teleop/line): ").strip()
        if mode not in ['ik', 'teleop', 'line']:
            print("Invalid mode.")
            return

        req = SwitchPublisher.Request()
        req.active_source = mode

        future = self.switch_cmd_cli.call_async(req)

        while not future.done() and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

        try:
            result = future.result()
            print(f"Switched control to {mode}! ✅")
        except Exception as e:
            print(f"Failed to switch mode: {e}")



def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryCommander()

    # Spin node in a separate thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # Now menu can run without blocking the subscription
    node.menu_loop()

    # After exiting menu
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()