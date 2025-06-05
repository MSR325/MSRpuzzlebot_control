#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Twist, Quaternion, PoseStamped, TransformStamped
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry, Path
import numpy as np
from .ekf_tricycle import EKFTricycleState  # your class
import time
import tf_transformations
from tf2_ros import TransformBroadcaster

class EKFNode(Node):
    def __init__(self):
        super().__init__('ekf_tricycle_node')

        # EKF filter
        self.ekf = EKFTricycleState()

        # Config: Choose calibration mode
        self.calibrate_measurement_noise = False
        self.calibrate_process_noise = False
        self.load_noise_from_files = True

        if self.load_noise_from_files:
            try:
                self.ekf.load_R("src/msr_localization/data/ekf_R_calibration.yaml")
                self.get_logger().info("✅ Loaded R matrix from file.")
            except Exception as e:
                self.get_logger().warn(f"⚠️ Could not load R: {e}")

            try:
                self.ekf.load_Q("src/msr_localization/data/ekf_Q_calibration.yaml")
                self.get_logger().info("✅ Loaded Q matrix from file.")
            except Exception as e:
                self.get_logger().warn(f"⚠️ Could not load Q: {e}")

        self.base_frame_id = self.declare_parameter("base_frame_id", "base_link").get_parameter_value().string_value
        self.odom_frame_id = self.declare_parameter("odom_frame_id", "odom").get_parameter_value().string_value

        if self.calibrate_measurement_noise and self.calibrate_process_noise:
            raise ValueError("Only one calibration mode can be active at a time!")

        # Subscriptions
        self.create_subscription(Imu, '/bno085/imu', self.imu_callback, 10)
        self.create_subscription(Odometry, '/odom', self.encoder_odom_callback, 10)  # odom topic has the linear and angular vels calculated with encoders [v_enc, omega_enc] 
        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        # EKF ODOM Publisher
        self.odom_pub = self.create_publisher(Odometry, '/ekf_odom', 10)

        # EKF state
        self.latest_imu = None
        self.latest_encoder = None
        self.u_k = np.array([0.0, 0.0])
        self.v_cmd_prev = 0.0
        self.prev_time = self.get_clock().now().nanoseconds / 1e9

        # Wheel model
        self.wheel_radius = 0.05  # m
        self.wheel_separation = 0.19  # m

        # Integrated wheel angles
        self.left_wheel_pos = 0.0
        self.right_wheel_pos = 0.0

        # Joint state publisher
        self.joint_pub_robot1 = self.create_publisher(JointState, '/robot1/joint_states', 10)
        self.joint_pub_robot2 = self.create_publisher(JointState, '/robot2/joint_states', 10)

        self.measurement_frequency = 55 # imu 

        self.last_record_time = 0.0  # for calibration downsampling
        self.calibration_freq = 50.0  # [Hz], lowest common frequency
        self.calibration_dt = 1.0 / self.calibration_freq

        self.ekf_path_pub = self.create_publisher(Path, '/ekf_path', 10)
        self.odom_path_pub = self.create_publisher(Path, '/odom_path', 10)

        self.ekf_path_msg = Path()
        self.ekf_path_msg.header.frame_id = "ekf_odom"

        self.odom_path_msg = Path()
        self.odom_path_msg.header.frame_id = self.odom_frame_id

        self.ekf_path_counter = 0
        self.odom_path_counter = 0

        self.tf_broadcaster = TransformBroadcaster(self)

        # Timer loop
        self.timer = self.create_timer(0.02, self.run_ekf_loop)  # 50Hz
        self.get_logger().info(f"EKF Node initialized")

    def cmd_vel_callback(self, msg: Twist):
        self.u_k = np.array([msg.linear.x, msg.angular.z])

    def imu_callback(self, msg: Imu):
        gyro_z = msg.angular_velocity.z
        accel_x = msg.linear_acceleration.x
        self.latest_imu = (gyro_z, accel_x)
        # self.get_logger().info(f"received imu data")

    def encoder_odom_callback(self, msg: Odometry):
        self.latest_encoder = (msg.twist.twist.linear.x, msg.twist.twist.angular.z)  # [v_enc, omega_enc]
        # self.get_logger().info(f"Received encoders data")

        if self.odom_path_counter % 3 == 0:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose = msg.pose.pose
            self.odom_path_msg.poses.append(pose)
            self.odom_path_msg.header.stamp = msg.header.stamp
            self.odom_path_pub.publish(self.odom_path_msg)
        self.odom_path_counter += 1


    def run_ekf_loop(self):
        if self.latest_encoder is None or self.latest_imu is None:
            self.get_logger().info(f"return")
            return

        now = self.get_clock().now().nanoseconds / 1e9
        dt = now - self.prev_time
        if dt <= 0.001:
            self.get_logger().info(f"return 2")
            return
        self.prev_time = now

        v_enc, omega_enc = self.latest_encoder
        gyro_z, accel_x = self.latest_imu
        z_k = np.array([v_enc, omega_enc, gyro_z, accel_x])

        # --- EKF step ---
        self.ekf.ekf_step(dt, self.u_k, self.v_cmd_prev, z_k)
        self.v_cmd_prev = self.u_k[0]

        # --- Optional calibration ---
        if self.calibrate_measurement_noise or self.calibrate_process_noise:
            if now - self.last_record_time >= self.calibration_dt:
                self.ekf.record_measurement(z_k, self.u_k, now)
                self.last_record_time = now

        if self.calibrate_measurement_noise:
            self.ekf.compute_measurement_noise(duration=10.0, frequency=self.measurement_frequency)
            if self.ekf.calibrated_R:
                self.get_logger().info("✅ Finished calibrating measurement noise.")
                self.calibrate_measurement_noise = False  # 🔕 Stop computing
                self.ekf.save_R("src/msr_localization/data/ekf_R_calibration.yaml")

            else:
                self.get_logger().info("🔊 Computing measurement noise...")
        elif self.calibrate_process_noise:
            self.ekf.compute_process_noise(duration=5.0, frequency=self.measurement_frequency)
            if self.ekf.calibrated_Q:
                self.get_logger().info("✅ Finished calibrating process noise.")
                self.calibrate_process_noise = False
                self.ekf.save_Q("src/msr_localization/data/ekf_Q_calibration.yaml")
            else:
                self.get_logger().info("🔊 Computing process noise...")

        # Log estimated state
        x, y, theta = self.ekf.get_pose()
        v = self.ekf[3]
        omega = self.ekf[4]
        # self.get_logger().info(f"[EKF] Pose: x={x:.2f}, y={y:.2f}, θ={theta:.2f}")
        stamp_msg = rclpy.time.Time(seconds=now).to_msg()
        
        self.publish_odometry(x, y, theta, v, omega, now)
        if self.ekf_path_counter % 3 == 0:
            self.update_ekf_path(x, y, theta, now)
        self.ekf_path_counter += 1
        self.publish_ekf_tf(x, y, theta, now)


        # Estimate wheel velocities from EKF output
        v = self.ekf[3]
        omega = self.ekf[4]

        v_l = v - (self.wheel_separation / 2.0) * omega
        v_r = v + (self.wheel_separation / 2.0) * omega

        # Integrate wheel positions
        self.left_wheel_pos += (v_l / self.wheel_radius) * dt
        self.right_wheel_pos += (v_r / self.wheel_radius) * dt

        joint_state = JointState()
        joint_state.header.stamp = stamp_msg
        joint_state.name = ['left_wheel_joint', 'right_wheel_joint']
        joint_state.position = [self.left_wheel_pos, self.right_wheel_pos]

        self.joint_pub_robot1.publish(joint_state)
        self.joint_pub_robot2.publish(joint_state)




    def publish_odometry(self, x, y, theta, v, omega, stamp):
        odom = Odometry()
        odom.header.stamp = rclpy.time.Time(seconds=stamp).to_msg()
        odom.header.frame_id = self.odom_frame_id
        odom.child_frame_id = self.base_frame_id

        # Position
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = 0.0

        # Orientation from theta
        quat = tf_transformations.quaternion_from_euler(0, 0, theta)
        odom.pose.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

        # Velocity
        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = omega

        self.odom_pub.publish(odom)


    def update_ekf_path(self, x, y, theta, stamp):
        pose = PoseStamped()
        pose.header.stamp = rclpy.time.Time(seconds=stamp).to_msg()
        pose.header.frame_id = "map"  # 
        self.ekf_path_msg.header.frame_id = "map"
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        quat = tf_transformations.quaternion_from_euler(0, 0, theta)
        pose.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
        self.ekf_path_msg.poses.append(pose)
        self.ekf_path_msg.header.stamp = pose.header.stamp
        self.ekf_path_pub.publish(self.ekf_path_msg)


    def publish_ekf_tf(self, x, y, theta, stamp):
        t = TransformStamped()
        t.header.stamp = rclpy.time.Time(seconds=stamp).to_msg()
        t.header.frame_id = "map"           # or "map", depending on your world frame
        t.child_frame_id = self.base_frame_id       # the robot's body

        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0

        q = tf_transformations.quaternion_from_euler(0, 0, theta)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)



def main(args=None):
    rclpy.init(args=args)
    node = EKFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
