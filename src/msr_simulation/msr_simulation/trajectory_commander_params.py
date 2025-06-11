#!/usr/bin/env python3
"""
trajectory_commander.py
Central “command & plumbing” node for MSR puzzle-bot
  • manages goal poses and trajectory YAML files
  • toggles between teleop / line / IK sources
  • owns feature-flag parameters for line-follower & cross-road detector
  • relays /cmd_source_request → /switch_cmd_source service
June 2025
"""

# ──────────────────────────────  IMPORTS  ─────────────────────────────────
import rclpy, json, yaml, os, time, subprocess, threading
from   rclpy.node               import Node
from   rclpy.parameter          import Parameter as RclpyParameter
from   geometry_msgs.msg        import PoseStamped, Point
from   visualization_msgs.msg   import MarkerArray, Marker
from   std_msgs.msg             import Int16, String
from   rcl_interfaces.srv       import SetParameters
from   custom_interfaces.srv    import SwitchPublisher

# ──────────────────────────────  NODE CLASS  ─────────────────────────────
class TrajectoryCommander(Node):
    # ──────────────────────────────  INIT  ───────────────────────────────
    def __init__(self):
        super().__init__('trajectory_commander')

        # -------- directories & state ------------------------------------
        self.data_path        = 'src/msr_simulation/data/trajectories'
        os.makedirs(self.data_path, exist_ok=True)
        self.current_goal     = None
        self.saved_trajectory = []
        self.pose_saver_active= False
        self.detection_fsm_msg= Int16(data=0)

        # -------- pubs / subs --------------------------------------------
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_cb, 10)
        self.create_subscription(String, '/cmd_source_request',
                                 self.cmd_req_cb, 10)

        self.marker_pub          = self.create_publisher(MarkerArray, '/trajectory_markers',      10)
        self.line_follow_pub     = self.create_publisher(Int16,        '/line_follow_enable',     10)
        self.detection_fsm_pub   = self.create_publisher(Int16,        '/detection_fsm_enable',   10)
        self.crossroad_detect_pub= self.create_publisher(Int16,        '/crossroad_detect_enable',10)

        # -------- service clients ----------------------------------------
        self.set_param_cli  = self.create_client(SetParameters,  '/inverse_kinematics/set_parameters')
        self.switch_cmd_cli = self.create_client(SwitchPublisher,'/switch_cmd_source')
        self.set_param_cli.wait_for_service(5.0)
        self.switch_cmd_cli.wait_for_service(5.0)

        # -------- dynamic feature flags ----------------------------------
        self.declare_parameter('line_follow_enabled',     False)
        self.declare_parameter('crossroad_detect_enabled',False)
        self._line_follow = self.get_parameter('line_follow_enabled').value
        self._crossroad   = self.get_parameter('crossroad_detect_enabled').value
        self.add_on_set_parameters_callback(self.on_param_update)

        # push initial flag values
        self.line_follow_pub.publish(    Int16(data=int(self._line_follow)))
        self.crossroad_detect_pub.publish(Int16(data=int(self._crossroad)))

        self.get_logger().info('🚀 TrajectoryCommander ready')

    # ────────────────────────────────────────────────────────────────────
    #  CALLBACKS & PARAM-SYNC
    # ────────────────────────────────────────────────────────────────────
    def goal_cb(self, msg: PoseStamped):
        yaw = self.quat_to_yaw(msg.pose.orientation)
        self.current_goal = dict(x=msg.pose.position.x,
                                 y=msg.pose.position.y,
                                 yaw=yaw)
        self.get_logger().info(f'🎯 Goal captured  x={msg.pose.position.x:.3f} '
                               f'y={msg.pose.position.y:.3f} yaw={yaw:.3f}')

    def cmd_req_cb(self, msg: String):
        req = SwitchPublisher.Request(active_source=msg.data.strip())
        future = self.switch_cmd_cli.call_async(req)
        future.add_done_callback(
            lambda f: self.get_logger().info(
                f"Mode switch to '{req.active_source}': "
                f"{'OK' if not f.exception() else f.exception()}"))

    def on_param_update(self, params):
        for p in params:
            if p.name == 'line_follow_enabled':
                self._line_follow = bool(p.value)
                self.line_follow_pub.publish(Int16(data=int(self._line_follow)))
            elif p.name == 'crossroad_detect_enabled':
                self._crossroad = bool(p.value)
                self.crossroad_detect_pub.publish(Int16(data=int(self._crossroad)))
        from rcl_interfaces.msg import SetParametersResult
        return SetParametersResult(successful=True)

    # ────────────────────────────────────────────────────────────────────
    #  MENU LOOP
    # ────────────────────────────────────────────────────────────────────
    def menu_loop(self):
        while rclpy.ok():
            print(f"""
Trajectory Commander   (LF={'ON' if self._line_follow else 'OFF'} |
                        CR={'ON' if self._crossroad   else 'OFF'})
1. 🎯 Send next goal pose
2. 💾 Save trajectory
3. 🔄 Send specific trajectory
4. 🎮 Toggle IK / Teleop / Line
5. 🛣️  Toggle Line Following Mode
6. 🚦 Toggle Semaphore Detection FSM
7. 👣 Toggle pose_saver
8. 🧭 Toggle Crossroad Detection
-1. 🔚 Exit
""")
            choice = input("Select: ").strip()
            if   choice == '1': self.send_next_goal()
            elif choice == '2': self.save_trajectory()
            elif choice == '3': self.send_specific_trajectory()
            elif choice == '4': self.toggle_cmd_source()
            elif choice == '5': self.flip_param('line_follow_enabled', self._line_follow)
            elif choice == '6': self.toggle_detection_fsm()
            elif choice == '7': self.toggle_pose_saver()
            elif choice == '8': self.flip_param('crossroad_detect_enabled', self._crossroad)
            elif choice == '-1': break
            else: print("❌ Invalid option.")

    def flip_param(self, name, current):
        self.set_parameters([RclpyParameter(name,
                                            RclpyParameter.Type.BOOL,
                                            not current)])

    # ────────────────────────────────────────────────────────────────────
    #  GOAL → IK PARAMS
    # ────────────────────────────────────────────────────────────────────
    def send_next_goal(self):
        print('Waiting up to 10 s for /goal_pose …')
        self.current_goal = None
        start = time.time()
        while self.current_goal is None and time.time() - start < 10 and rclpy.ok():
            time.sleep(0.05)               # let the background spin thread do the work
        if self.current_goal is None:
            print('❌ Timeout.'); return

        req = SetParameters.Request(parameters=[
            RclpyParameter('desired_x',   RclpyParameter.Type.DOUBLE, self.current_goal['x']).to_parameter_msg(),
            RclpyParameter('desired_y',   RclpyParameter.Type.DOUBLE, self.current_goal['y']).to_parameter_msg(),
            RclpyParameter('desired_yaw', RclpyParameter.Type.DOUBLE, self.current_goal['yaw']).to_parameter_msg()
        ])
        fut = self.set_param_cli.call_async(req)
        fut.add_done_callback(
            lambda f: print('✅ Goal sent.' if (not f.exception() and
                                               all(r.successful for r in f.result().results))
                            else f'❌ {f.exception() or f.result()}'))
        print('⏳ Sending goal (watch console for confirmation)…')


    # ────────────────────────────────────────────────────────────────────
    #  TRAJECTORY CAPTURE / STORAGE
    # ────────────────────────────────────────────────────────────────────
    def save_trajectory(self):
        try: n_pts = int(input('How many points to save? '))
        except ValueError: print('Invalid.'); return

        self.saved_trajectory.clear()
        print(f'Publish {n_pts} /goal_pose messages…')
        collected, timeout = 0, 30.0
        while collected < n_pts and rclpy.ok():
            prev = self.current_goal
            start = time.time()
            while self.current_goal == prev and time.time() - start < timeout and rclpy.ok():
                time.sleep(0.05)            # no spin_once here
            if self.current_goal != prev:
                self.saved_trajectory.append(self.current_goal.copy())
                collected += 1
                print(f'  ✅ {collected}/{n_pts}')
            else:
                print('⚠️  Timeout waiting for a new goal.'); break


        fname = input('Filename (without .yaml): ').strip()
        if not fname: print('Empty filename.'); return
        path = os.path.join(self.data_path, f'{fname}.yaml')

        yaml_data = {
            '/path_generator': {
                'ros__parameters': {
                    'sample_time': "0.018",                       # ←  quotes → string
                    'waypoints_json': json.dumps(self.saved_trajectory, indent=4)
                }
            }
        }
        with open(path, 'w') as f: yaml.dump(yaml_data, f)
        print(f'📄 Saved to {path}')

    # ────────────────────────────────────────────────────────────────────
    #  LOAD TRAJECTORY & RViz VISUALISATION
    # ────────────────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────────
    #  LOAD TRAJECTORY & push to /path_generator with type-fallback
    # ────────────────────────────────────────────────────────────────────
    def send_specific_trajectory(self):
        import json, yaml, os, subprocess, time
        from geometry_msgs.msg import Point
        from visualization_msgs.msg import Marker, MarkerArray
        from rcl_interfaces.srv import SetParameters

        # ----- pick YAML file -------------------------------------------------
        files = [f for f in os.listdir(self.data_path) if f.endswith('.yaml')]
        if not files:
            print('No YAML trajectory files found.'); return
        for i, f in enumerate(files): print(f'[{i}] {f}')
        try:
            idx = int(input('Index: ')); assert 0 <= idx < len(files)
        except: print('❌ Invalid index'); return
        path = os.path.join(self.data_path, files[idx])

        # ----- parse points for RViz -----------------------------------------
        try:
            with open(path) as f: y = yaml.safe_load(f)
            traj = json.loads(y['/path_generator']['ros__parameters']['waypoints_json'])
        except Exception as e:
            print(f'Could not read waypoints_json from YAML: {e}')
            traj = None   # we can still push the YAML even if RViz fails

        # ---------------------------------------------------------------------
        # 1) TRY THE FAST WAY  → ros2 param load (works if types already match)
        # ---------------------------------------------------------------------
        cmd = ['ros2', 'param', 'load', '/path_generator', path]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print(res.stdout or '✅ Parameters loaded via ros2 param load')
        else:
            print('First attempt failed; retrying with SetParameters …')
            client = self.create_client(SetParameters, '/path_generator/set_parameters')
            if not client.wait_for_service(timeout_sec=5.0):
                print('❌ /path_generator/set_parameters not available.'); return

            # build both variants once
            attempts = [
                ('double', RclpyParameter.Type.DOUBLE, 0.018),
                ('string', RclpyParameter.Type.STRING, "0.018")
            ]

            def _try(idx=0):
                if idx >= len(attempts):
                    print('❌ All type attempts were rejected.'); return
                label, typ, val = attempts[idx]
                req = SetParameters.Request(parameters=[
                    RclpyParameter('sample_time', typ, val).to_parameter_msg(),
                    RclpyParameter('waypoints_json', RclpyParameter.Type.STRING,
                                   json.dumps(traj) if traj else "").to_parameter_msg()
                ])
                fut = client.call_async(req)
                fut.add_done_callback(lambda f, i=idx, l=label: _on_reply(f, i, l))

            def _on_reply(fut, idx, label):
                if fut.exception() or not fut.result().successful:
                    print(f'↪ {label} attempt rejected; trying next …')
                    _try(idx + 1)
                else:
                    print(f'✅ Parameters accepted with sample_time as {label}')

            _try()   # kick off the first attempt


        # ----- RViz visualisation (unchanged) ---------------------------------
        if traj:
            ma   = MarkerArray()
            line = Marker(); line.header.frame_id = 'map'
            line.type = Marker.LINE_STRIP; line.action = Marker.ADD
            line.scale.x = 0.05
            line.color.r = line.color.b = line.color.a = 1.0
            line.pose.orientation.w = 1.0
            for p in traj:
                line.points.append(Point(x=p['x'], y=p['y'], z=0.0))
            ma.markers.append(line)

            for i, p in enumerate(traj):
                s = Marker(); s.header.frame_id = 'map'
                s.type = Marker.SPHERE; s.action = Marker.ADD; s.id = i + 1
                s.pose.position.x = p['x']; s.pose.position.y = p['y']
                s.scale.x = s.scale.y = s.scale.z = 0.05
                s.color.g = s.color.a = 1.0
                s.pose.orientation.w = 1.0
                ma.markers.append(s)
            self.marker_pub.publish(ma)
            print('🖼️  RViz markers published.')

    # ────────────────────────────────────────────────────────────────────
    #  SIMPLE TOGGLES
    # ────────────────────────────────────────────────────────────────────
    def toggle_detection_fsm(self):
        self.detection_fsm_msg.data ^= 1
        self.detection_fsm_pub.publish(self.detection_fsm_msg)
        print('✅ Detection FSM ON' if self.detection_fsm_msg.data else '🛑 Detection FSM OFF')

    def toggle_cmd_source(self):
        print('\n  [0] teleop  [1] line  [2] ik')
        modes = {0: 'teleop', 1: 'line', 2: 'ik'}
        try:
            mode = modes[int(input('Index: ').strip())]
        except (ValueError, KeyError):
            print('❌'); return

        req = SwitchPublisher.Request(active_source=mode)
        fut = self.switch_cmd_cli.call_async(req)

        # print result asynchronously so we don't block / clash with the executor
        fut.add_done_callback(
            lambda f: self.get_logger().info(
                f"Switch to '{mode}': "
                f"{'OK' if (not f.exception() and f.result().success) else 'FAILED'}"))


        print(f'⏳ Requested switch to {mode} (wait for confirmation in log)')

    # ---- pose saver helper identical to your previous version ----------
    # ────────────────────────────────────────────────────────────────────
    #  Pose-saver toggle (non-blocking)
    # ────────────────────────────────────────────────────────────────────
    def toggle_pose_saver(self):
        from std_msgs.msg import Int16
        if self.pose_saver_active:
            # stop
            self.pose_saver_enable_pub.publish(Int16(data=0))
            self.pose_saver_active = False
            print('🛑 Pose saver stopped.')
            return

        # start
        try:
            freq = float(input('Pose save frequency (Hz): '))
        except ValueError:
            print('Invalid number.'); return
        fname = input('Filename (without .yaml): ').strip()
        if not fname:
            print('Filename cannot be empty.'); return

        # build parameter list
        params = [
            RclpyParameter('pose_save_frequency',
                           RclpyParameter.Type.DOUBLE,  freq).to_parameter_msg(),
            RclpyParameter('trajectory_file_name',
                           RclpyParameter.Type.STRING,  fname).to_parameter_msg()
        ]

        # async service call -------------------------------------------------
        client = self.create_client(SetParameters,
                                    '/pose_saver/set_parameters')
        if not client.wait_for_service(timeout_sec=5.0):
            print('❌ /pose_saver/set_parameters service not available.'); return

        fut = client.call_async(SetParameters.Request(parameters=params))

        # result callback (runs in spin thread)
        def _on_ps_result(f):
            if f.exception():
                self.get_logger().error(f'Pose-saver param set failed: {f.exception()}')
                return
            self.get_logger().info('Pose-saver parameters accepted ✔')
            # create publisher only once
            if not hasattr(self, 'pose_saver_enable_pub'):
                self.pose_saver_enable_pub = self.create_publisher(
                    Int16, '/pose_saver_enable', 10)
            self.pose_saver_enable_pub.publish(Int16(data=1))
            self.pose_saver_active = True
            self.get_logger().info(
                f'🟢 Pose saver started ({freq} Hz → {fname}.yaml). '
                f'Choose menu [7] again to stop.')

        fut.add_done_callback(_on_ps_result)
        print('⏳ Sending parameters… (watch log for confirmation)')

    # ────────────────────────────────────────────────────────────────────
    #  UTILS
    # ────────────────────────────────────────────────────────────────────
    @staticmethod
    def quat_to_yaw(q):
        import math
        return math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))

# ──────────────────────────────  MAIN  ───────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryCommander()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    try: node.menu_loop()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
