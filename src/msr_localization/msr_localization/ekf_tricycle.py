import numpy as np
import yaml

class EKFTricycleState:
    def __init__(self, x=0.0, y=0.0, theta=0.0, v=0.0, omega=0.0, bias_imu=0.0, bias_gyro=0.0):
        self.x = np.array([
            x,          # x_position
            y,          # y_position
            theta,      # heading
            v,          # linear velocity
            omega,      # angular velocity
            bias_imu,   # accelerometer bias
            bias_gyro   # gyroscope bias
        ], dtype=float)
        self.P = np.eye(7) * 0.1  # Initial state uncertainty (can be tuned)

        self.Q = np.diag([
            0.01,   # x noise
            0.01,   # y noise
            0.001,  # theta noise
            0.05,   # linear velocity noise
            0.02,   # angular velocity noise
            0.01,   # accelerometer bias drift
            0.005   # gyro bias drift
        ])

        self.R = np.diag([
            0.02,   # encoder v noise
            0.01,   # encoder omega noise
            0.005,  # gyro noise
            0.05    # IMU forward acceleration noise
        ])

        self.measurement_history = []
        self.encoder_history = []
        self.t_start = None
        
        self.calibrated_R = False
        self.calibrated_Q = False



    def __getitem__(self, index):
        return self.x[index]

    def __setitem__(self, index, value):
        self.x[index] = value

    def as_vector(self):
        return self.x.copy()

    def predict_state(self, dt, u):
        """
        Applies the velocity-based motion model with control input:
        u = [v_cmd, omega_cmd]
        """
        x, y, theta, _, _, bias_imu, bias_gyro = self.x
        v_cmd, omega_cmd = u

        # Predict new pose
        x_new = x + v_cmd * np.cos(theta) * dt
        y_new = y + v_cmd * np.sin(theta) * dt
        theta_new = theta + omega_cmd * dt

        # Replace velocities with commanded ones
        self.x = np.array([
            x_new,
            y_new,
            theta_new,
            v_cmd,
            omega_cmd,
            bias_imu,
            bias_gyro
        ])

    def jacobian_A(self, dt):
        """Jacobian of the motion model ∂f/∂x, using current commanded velocities."""
        _, _, theta, _, _, _, _ = self.x
        v_cmd = self.x[3]

        A = np.eye(7)
        A[0, 2] = -v_cmd * np.sin(theta) * dt
        A[0, 3] =  np.cos(theta) * dt
        A[1, 2] =  v_cmd * np.cos(theta) * dt
        A[1, 3] =  np.sin(theta) * dt
        A[2, 4] =  dt

        return A

    def h(self, dv_approx):
        """
        Measurement function h(x), using externally computed forward acceleration.
        dv_approx = (v_cmd[k] - v_cmd[k-1]) / dt
        """
        _, _, _, v, omega, bias_imu, bias_gyro = self.x
        return np.array([
            v,                            # encoder velocity
            omega,                        # encoder yaw rate
            omega + bias_gyro,            # IMU gyroscope
            dv_approx + bias_imu          # IMU forward acceleration (supposedly, since dv is calculaded with v_cmd)
        ])

    def jacobian_H(self, dt):
        """
        Jacobian H = ∂h/∂x for the measurement model.
        Depends on dt for ∂a/∂v = 1/dt
        """
        H = np.zeros((4, 7))

        # ∂v/∂v
        H[0, 3] = 1.0

        # ∂omega/∂omega
        H[1, 4] = 1.0

        # ∂(omega + bias_gyro)/∂omega and ∂.../∂bias_gyro
        H[2, 4] = 1.0
        H[2, 6] = 1.0

        # ∂(dv + bias_imu)/∂v and ∂.../∂bias_imu
        H[3, 3] = 1.0 / dt  # ∂(dv) ≈ ∂(Δv)/∂v
        H[3, 5] = 1.0

        return H
    
    def ekf_step(self, dt, u_k, v_cmd_prev, z_k):
        """
        Executes one full EKF step: prediction + correction.

        Parameters:
        - dt: time step
        - u_k: current control input [v_cmd, omega_cmd]
        - v_cmd_prev: previous velocity command (for dv_approx)
        - z_k: current measurement vector [v_enc, omega_enc, gyro, accel]
        """

        # ---- PREDICTION STEP ----
        A = self.jacobian_A(dt)
        self.predict_state(dt, u_k)  # updates self.x
        self.P = A @ self.P @ A.T + self.Q

        # ---- MEASUREMENT UPDATE ----
        dv_approx = (u_k[0] - v_cmd_prev) / dt
        H = self.jacobian_H(dt)
        z_pred = self.h(dv_approx)
        y = z_k - z_pred                       # innovation
        S = H @ self.P @ H.T + self.R          # innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)    # Kalman gain

        # Update state estimate and covariance
        self.x += K @ y
        I = np.eye(len(self.x))
        self.P = (I - K @ H) @ self.P


    def get_pose(self):
        return self.x[0], self.x[1], self.x[2]

    def reset(self, x0=None, P0=None):
        self.x = np.zeros_like(self.x) if x0 is None else np.array(x0, dtype=float)
        self.P = np.eye(7) * 0.1 if P0 is None else np.array(P0, dtype=float)


    def record_measurement(self, z_k, u_k, timestamp):
        if self.t_start is None:
            self.t_start = timestamp
        self.measurement_history.append((timestamp, z_k.copy()))
        self.encoder_history.append((timestamp, u_k[0]))  # only store v_cmd

    def compute_measurement_noise(self, duration=5.0, frequency=50.0):
        """
        Compute R from measurements during static period.
        duration: time window in seconds
        """
        if self.calibrated_R:
            return

        samples_needed = int(duration * frequency)
        if len(self.measurement_history) < samples_needed:
            return

        z_data = np.array([z for (_, z) in self.measurement_history[-samples_needed:]])

        variances = np.var(z_data, axis=0)
        self.R = np.diag(variances)
        self.calibrated_R = True
        self.measurement_history.clear()
        print("✅ Updated R:")
        for i, label in enumerate(["v_enc", "omega_enc", "gyro_z", "accel_x"]):
            print(f"{label}: variance = {variances[i]}")

    def compute_process_noise(self, duration=5.0, frequency=55.0):
        """
        Estimate Q by comparing encoder velocity vs. commanded velocity over time.
        Assumes encoder velocity is part of measurement[0].
        """
        if not self.encoder_history or not self.measurement_history:
            return

        if self.calibrated_Q:
            return

        samples_needed = int(duration * frequency)

        if len(self.measurement_history) < samples_needed or len(self.encoder_history) < samples_needed:
            return

        # Take the most recent samples
        recent_measurements = self.measurement_history[-samples_needed:]
        recent_commands = self.encoder_history[-samples_needed:]

        error_v = [
            z[0] - v_cmd  # v_enc - v_cmd
            for (_, z), (_, v_cmd) in zip(recent_measurements, recent_commands)
        ]

        if len(error_v) < 2:
            print("Not enough data to estimate Q.")
            return

        # Estimate variance of velocity prediction error
        q_v = np.var(error_v)

        # Update Q matrix (only for v; others stay the same or can be done similarly)
        self.Q[3, 3] = q_v
        self.calibrated_Q = True
        self.measurement_history.clear()
        print("✅ Updated Q (v component):\n", self.Q)

    def save_R(self, filename="ekf_R.yaml"):
        data = {'R': self.R.tolist()}
        with open(filename, 'w') as file:
            yaml.dump(data, file)
        print(f"✅ Saved R matrix to {filename}")

    def save_Q(self, filename="ekf_Q.yaml"):
        data = {'Q': self.Q.tolist()}
        with open(filename, 'w') as file:
            yaml.dump(data, file)
        print(f"✅ Saved Q matrix to {filename}")

    def load_R(self, filename="ekf_R.yaml"):
        with open(filename, 'r') as file:
            data = yaml.safe_load(file)
            self.R = np.array(data['R'])
        self.calibrated_R = True
        print(f"✅ Loaded R matrix from {filename}")

    def load_Q(self, filename="ekf_Q.yaml"):
        with open(filename, 'r') as file:
            data = yaml.safe_load(file)
            self.Q = np.array(data['Q'])
        self.calibrated_Q = True
        print(f"✅ Loaded Q matrix from {filename}")

    def save_noise_matrices(self, filename="ekf_noise_calibration.yaml"):
        data = {'R': self.R.tolist(), 'Q': self.Q.tolist()}
        with open(filename, 'w') as file:
            yaml.dump(data, file)
        print(f"✅ Saved R and Q matrices to {filename}")
