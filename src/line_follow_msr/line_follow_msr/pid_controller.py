import rclpy
from rclpy.node import Node
import time
import math

class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=.0, setpoint=0.0, output_limits=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits

        # State variables
        self._prev_error = 0.0
        self._integral = 0.0
        self._last_time = None

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self._last_time = None

    def compute(self, input_value):
        current_time = time.time()

        # Calculate error
        error = self.setpoint - input_value

        # Initialize time if first call
        if self._last_time is None:
            self._last_time = current_time
            self._last_error = error
            return 0.0

        # Calculate time delta
        dt = current_time - self._last_time
        if dt <= 0.0:
            dt = 1e-6 # Avoid division by zero

        # Proportional term
        proportional = self.kp * error

        # Integral term
        self._integral += error * dt
        integral = self.ki * self._integral

        # Derivative term
        derivative = self.kd * (error - self._last_error) / dt

        # Output
        output = proportional + integral + derivative
    
        # Apply output limits
        if self.output_limits is not None:
            min_limit, max_limit = self.output_limits
            if output > max_limit:
                output = max_limit
                # Anti-windup: Limit integral if saturated
                self._integral -= error * dt
            elif output < min_limit:
                output = min_limit
                self._integral -= error * dt
            
        # Save values for next iteration
        self._last_error = error
        self.last_time = current_time

        return output