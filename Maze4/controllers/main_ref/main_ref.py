from my_robot import MyRobot


def main():
    robot = MyRobot()

    main_path = robot.explore()

    if not main_path:
        print('[main] No final path found — exploration may have missed a pillar.')
        return

    print(f'[main] Final path found ({len(main_path)} waypoints). Following...')
    robot.step(100)
    robot.follow_final_path(main_path, debug_vis=True, replan_interval=60)


if __name__ == '__main__':
    main()
