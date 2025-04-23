"""
Node that drives the robot in a square path of a side length of 2 meters
Implements strategies for open loop control robustness
The controller is auto-tuned: The user selects the speed or time to finish the path
and the controller must estimate the velocities, acceleration or time required
Perturbation, non-linearities, and noise must be taken into consideration
"""

# Imports
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist

def get_user_input(square_path_ctrl):
    while True:
        try:
            decision = int(input(
                "To operate an auto-tunned open loop control system for the Puzzlebot, \n"
                "select the speed or time to finish the square path:\n"
                "Type '0' for speed | Type '1' for time: "))
                    
            if decision == 0:

                while True:
                    try:

                        path_linear_speed = float(input("\nEnter desired linear speed (0.01 to 0.2 m/s): "))
                        if (0.01 <= path_linear_speed <= 0.2):
                            break
                        print("\nError: Number must be between 0.01 to 0.2 m/s. Please try again.")
                    except ValueError:
                        print("Error: Please enter a valid number.")
                    
                while True:
                    try:
                        
                        path_angular_speed = float(input("\nEnter desired angular speed (0.01 to 0.5 rad/s): "))
                        if (0.01 < path_angular_speed <= 0.5):
                            break
                        print("\nError: Number must be between 0.01 to 0.5 rad/s. Please try again.")
                    except ValueError:
                        print("Error: Please enter a valid number.")
                
                square_path_ctrl.linear_speed = path_linear_speed
                square_path_ctrl.angular_speed = path_angular_speed
                square_path_ctrl.forward_time = square_path_ctrl.side_length / path_linear_speed
                square_path_ctrl.rotate_time = square_path_ctrl.half_pi / path_angular_speed
                break

            elif decision == 1:

                while True:
                    try:

                        total_time = float(input(
                            "\nEnter desired total path time (must be > 0 seconds): "
                        ))
                        if total_time > 0:
                            break
                        print("Error: Time must be positive")
                    except ValueError:
                        print("Error: Please enter a valid number.")
                
                # The square path consists of 4 forward segments and 4 rotations
                # A ratio between linear and angular speeds is assumed: 0.4
                speed_ratio = 0.4
                square_path_ctrl.linear_speed = (4*square_path_ctrl.side_length + 4*square_path_ctrl.half_pi/speed_ratio) / total_time
                square_path_ctrl.angular_speed = square_path_ctrl.linear_speed * speed_ratio
                square_path_ctrl.forward_time = square_path_ctrl.side_length / square_path_ctrl.linear_speed
                square_path_ctrl.rotate_time = square_path_ctrl.half_pi / square_path_ctrl.angular_speed
                square_path_ctrl.finish_path_time = total_time
                break

            else:
                print("Error: Please enter either 0 (for speed) or 1 (for time).")
            
            print(f"\nCalculated parameters:")
            print(f"Linear speed: {square_path_ctrl.linear_speed:.3f} m/s")
            print(f"Angular speed: {square_path_ctrl.angular_speed:.3f} rad/s")
            print(f"Estimated forward time per segment: {square_path_ctrl.forward_time:.2f} s")
            print(f"Estimated rotation time per turn: {square_path_ctrl.rotate_time:.2f} s")

        except ValueError:
            print("Error: Please enter either 0 (for speed) or 1 (for time).")  

#Class definition
class SquarePathCtrl(Node):
    def __init__(self):
        super().__init__('square_path_ctrl')

        # self.wait_for_ros_time() Uncomment this function if using Gazebo for simulating the robot

        # Publisher to /cmd_vel
        self.cmd_vel_pub = self.create_publisher(Twist, 'pseudo_cmd_vel', 10)

        # Time-based control variables
        self.state = 0 # 0: forward, 1: rotate, 2: stop
        self.state_start_time = self.get_clock().now()

        # Define square side length
        self.side_length = 2.0 # 2 m
        # Define 180 deg
        self.pi = 3.1416
        # Define 90 deg
        self.half_pi = self.pi / 2

        # Define forward movement counter
        self.forward_counter = 0

        # Timer to update state machine
        self.timer_period = 0.2 # 10 Hz control loop
        self.timer = self.create_timer(self.timer_period, self.control_loop)

        # Motion parameters (modified by user input)
        # Define speeds
        self.linear_speed = None # m/s (will be set by user)
        self.angular_speed = None # rad/s (will be set by user)
        # Define finish time for square path
        self.finish_path_time = None # s (will be set by user)
        # Define durations (seconds) / also time since the input was applied
        self.forward_time = None # Time to move 2 m (will be set by user)
        self.rotate_time = None # Time to rotate 180 deg (will be set by user)

        self.get_logger().info("Open loop controller for square path initialized!")

    def control_loop(self):
        now = self.get_clock().now()
        elapsed_time = (now - self.state_start_time).nanoseconds * 1e-9
        cmd = Twist()

        if self.state == 0:
            # Move forward
            cmd.linear.x = self.linear_speed
            self.get_logger().info("Moving forward...")
            if elapsed_time >= self.forward_time:
                self.forward_counter += 1
                if self.forward_counter == 4:
                    self.state_start_time = now
                    self.state = 2
                    self.forward_counter = 0
                    self.get_logger().info("Finished moving forward. Stopping...")
                else:
                    self.state_start_time = now
                    self.state = 1
                    self.get_logger().info("Finished moving forward. Starting rotation...")

        elif self.state == 1:
            # Rotate 180 degrees
            cmd.angular.z = self.angular_speed
            self.get_logger().info("Rotating 90 degrees...")
            if elapsed_time >= self.rotate_time:
                self.state = 0
                self.state_start_time = now
                self.get_logger().info("Finished rotation. Moving forward...")

        elif self.state == 2:
            # Stop
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.get_logger().info("Stopped.")
            # Optionally: cancel the timer after stopping
            self.timer.cancel()

        # Publish velocity command
        self.cmd_vel_pub.publish(cmd)

    def pose_sub_callback(self, msg):
        self.path_point = msg.data
    
    """
    When connecting Gazebo with ROS, the user Nodes must wait until Gazebo
    publishes the topic "/clock" to synchronize the Gazebo simulation time with ROS time.
    Otherwise, the nodes might start without Gazebo and could cause instability.
    For calculations of time durations when using simulation time, clients should always wait until
    the first non-zero time value has been received before starting.
    """
    def wait_for_ros_time(self):
        self.get_logger().info("Waiting for ROS time to become active...")
        while rclpy.ok():
            now = self.get_clock().now()
            if now.nanoseconds > 0:
                break
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info(f"ROS time is active!")

# Main
def main(args=None):
    rclpy.init(args=args)
    
    # Create the node
    node = SquarePathCtrl()
    
    # Get user input once
    get_user_input(node)
    
    # Print a clear message indicating execution is starting
    node.get_logger().info("Starting square path execution with the provided parameters...")
    node.get_logger().info(f"Linear speed: {node.linear_speed:.3f} m/s")
    node.get_logger().info(f"Angular speed: {node.angular_speed:.3f} rad/s")
    
    # Execute the node's control loop
    try:
        rclpy.spin(node)  # This will run the timer callback (control_loop)
    except KeyboardInterrupt:
        node.get_logger().info("Execution interrupted by user")
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        node.destroy_node()

# Execute Node
if __name__ == '__main__':
    main()