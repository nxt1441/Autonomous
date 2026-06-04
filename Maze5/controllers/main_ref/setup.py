from CONSTANTS import TIME_STEP


def setup_robot(robot):
    # Motors — skid-steer layout: FL, FR, RL, RR
    motors = {
        'fl': robot.getDevice('fl_wheel_joint'),
        'fr': robot.getDevice('fr_wheel_joint'),
        'rl': robot.getDevice('rl_wheel_joint'),
        'rr': robot.getDevice('rr_wheel_joint'),
    }
    for m in motors.values():
        m.setPosition(float('inf'))
        m.setVelocity(0.0)

    # Wheel encoders
    sensors = {
        'fl': robot.getDevice('front left wheel motor sensor'),
        'fr': robot.getDevice('front right wheel motor sensor'),
        'rl': robot.getDevice('rear left wheel motor sensor'),
        'rr': robot.getDevice('rear right wheel motor sensor'),
    }
    for s in sensors.values():
        s.enable(TIME_STEP)

    # IMU suite
    imu = {
        'accelerometer': robot.getDevice('imu accelerometer'),
        'gyro':          robot.getDevice('imu gyro'),
        'compass':       robot.getDevice('imu compass'),
    }
    for dev in imu.values():
        dev.enable(TIME_STEP)

    # Cameras
    cam_rgb   = robot.getDevice('camera rgb')
    cam_depth = robot.getDevice('camera depth')
    cam_rgb.enable(TIME_STEP)
    cam_depth.enable(TIME_STEP)

    # LiDAR
    lidar = robot.getDevice('laser')
    lidar.enable(TIME_STEP)
    lidar.enablePointCloud()

    # IR range sensors — order: [fl, rl, fr, rr]
    dist_sensors = []
    for name in ['fl_range', 'rl_range', 'fr_range', 'rr_range']:
        dev = robot.getDevice(name)
        dev.enable(TIME_STEP)
        dist_sensors.append(dev)

    return motors, sensors, imu, cam_rgb, cam_depth, lidar, dist_sensors
