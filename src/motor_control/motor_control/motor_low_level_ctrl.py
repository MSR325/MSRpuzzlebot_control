import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray
from rcl_interfaces.msg import SetParametersResult
from math import fabs

class MotorPIDController(Node):
    def __init__(self):
        super().__init__('motor_pid_controller')

        # --- PID state ---
        self.Kp = 0.02
        self.Ki = 0.2
        self.Kd = 0.00005
        self.integral = 0.0
        self.last_error = 0.0

        # --- Control loop ---
        self.setpoint = 0.0
        self.measured_speed = 0.0
        self.filtered_speed = 0.0
        self.last_time = self.get_clock().now()

        self.declare_parameter('Kp', 0.02)
        self.declare_parameter('Ki', 0.2)
        self.declare_parameter('Kd', 0.00005)

        # Load initial values
        self.Kp = self.get_parameter('Kp').value
        self.Ki = self.get_parameter('Ki').value
        self.Kd = self.get_parameter('Kd').value

        self.create_subscription(Float32, '/VelocityEncL', self.encoder_callback, 10)
        self.create_subscription(Float32, '/ik_cmd_vel', self.setpoint_callback, 10)

        self.pwm_pub = self.create_publisher(Float32, '/ControlL', 10)

        self.timer = self.create_timer(0.02, self.control_loop)  # 50 Hz
        self.add_on_set_parameters_callback(self.param_update_callback)

        self.get_logger().info("🚀 Motor PID Controller started")

    def setpoint_callback(self, msg):
        self.setpoint = msg.data
        self.get_logger().info(f"📨 New setpoint: {self.setpoint:.3f} rad/s")

    def encoder_callback(self, msg):
        self.measured_speed = msg.data

    def compute_pid(self, setpoint, measured, dt):
        error = setpoint - measured

        if fabs(setpoint) < 1e-6:
            self.integral = 0.0
        else:
            self.integral += error * dt

        derivative = 0.0 if dt < 1e-6 else (error - self.last_error) / dt
        self.last_error = error

        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        return max(min(output, 1.0), -1.0)  # Clamp between -1 and 1

    def low_pass_filter(self, new_val, old_val, dt, tau=0.05):
        alpha = dt / (tau + dt)
        return alpha * new_val + (1 - alpha) * old_val

    def param_update_callback(self, params):
        for param in params:
            if param.name == "Kp":
                self.Kp = param.value
            elif param.name == "Ki":
                self.Ki = param.value
            elif param.name == "Kd":
                self.Kd = param.value

        self.get_logger().info(f"🔧 PID gains updated: Kp={self.Kp}, Ki={self.Ki}, Kd={self.Kd}")
        return SetParametersResult(successful=True)

    def control_loop(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        # Apply low-pass filter
        self.filtered_speed = self.low_pass_filter(
            self.measured_speed, self.filtered_speed, dt
        )

        # Compute PID
        control = self.compute_pid(self.setpoint, self.filtered_speed, dt)

        # Map to PWM range 0.0–0.35
        pwm_out = fabs(control) * 0.35

        pwm_msg = Float32()
        pwm_msg.data = pwm_out

        self.pwm_pub.publish(pwm_msg)

        # Optional debug output
        self.get_logger().info(
            f"stp {self.setpoint:.2f} | rad/s {self.filtered_speed:.2f} | err err={self.setpoint - self.filtered_speed:.2f} | ctrl_out ctrl={control:.3f} → PWM PWM={pwm_out:.3f}"
        )

def main(args=None):
    rclpy.init(args=args)
    node = MotorPIDController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
