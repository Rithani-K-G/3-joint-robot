
#!/usr/bin/env python3

"""
Move the three-joint robot to an XYZ position using pymoveit2.

The user provides an XYZ target in metres.
MoveIt calculates the required joint positions using IK,
plans a collision-aware trajectory, and executes it through
the configured arm_controller.

Coordinates are relative to base_link.
"""

import sys
import math
import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from pymoveit2 import MoveIt2


# These values must match the three-joint robot's configuration.
JOINT_NAMES = ["joint1", "joint2", "joint3"]

BASE_LINK = "base_link"
END_EFFECTOR_LINK = "tool0"
PLANNING_GROUP = "arm"


def read_coordinate(label: str) -> float:
    """Read one finite coordinate from the terminal."""
    while True:
        try:
            value = float(input(f"{label} (metres): ").strip())

            if math.isfinite(value):
                return value

            print("Enter a finite number.")

        except ValueError:
            print("Invalid number. Example: 0.3864")


def main():

    print("\n=== Three-joint robot: XYZ motion using MoveIt 2 ===")
    print("Reference frame: base_link")
    print("End-effector: tool0")
    print("Units: metres\n")

    # Read the target coordinates.
    try:
        x = read_coordinate("X")
        y = read_coordinate("Y")
        z = read_coordinate("Z")

    except KeyboardInterrupt:
        print("\nInput cancelled.")
        return 1

    print("\nTarget:")
    print(f"  X = {x:.4f} m")
    print(f"  Y = {y:.4f} m")
    print(f"  Z = {z:.4f} m")

    answer = input("\nPlan this target? [y/N]: ").strip().lower()

    if answer not in ("y", "yes"):
        print("Cancelled.")
        return 0

    # Initialize ROS 2.
    rclpy.init()

    node = Node(
        "xyz_moveit_client",
        parameter_overrides=[
            Parameter("use_sim_time", value=True)
        ],
    )

    moveit = None
    executor = MultiThreadedExecutor(num_threads=2)
    spin_thread = None

    try:

        # Create the MoveIt interface.
        moveit = MoveIt2(
            node=node,
            joint_names=JOINT_NAMES,
            base_link_name=BASE_LINK,
            end_effector_name=END_EFFECTOR_LINK,
            group_name=PLANNING_GROUP,
        )

        # Run the ROS 2 executor in a separate thread.
        executor.add_node(node)

        spin_thread = threading.Thread(
            target=executor.spin,
            daemon=True,
        )

        spin_thread.start()

        node.get_logger().info(
            f"Planning XYZ target: ({x:.3f}, {y:.3f}, {z:.3f})"
        )

        # Ask MoveIt to calculate the required joint positions
        # using inverse kinematics and plan a trajectory.
        trajectory = moveit.plan(
            position=(x, y, z),
            frame_id=BASE_LINK,
            target_link=END_EFFECTOR_LINK,
            timeout_sec=30.0,
        )

        if trajectory is None:

            print(
                "\nMoveIt could not find a valid trajectory.\n"
                "Check the target coordinates, IK configuration, "
                "joint limits, planning scene and controller setup."
            )

            return 2

        print("\nMoveIt found a trajectory.")

        answer = input(
            "Execute the trajectory? [y/N]: "
        ).strip().lower()

        if answer not in ("y", "yes"):

            print("Trajectory planned but not executed.")
            return 0

        # Execute the planned trajectory.
        if not moveit.execute(trajectory):

            print("\nMoveIt rejected the trajectory.")
            return 3

        print("\nTrajectory sent. Waiting for execution result...")

        success = moveit.wait_until_executed(
            timeout_sec=60.0,
            cancel_on_timeout=True,
        )

        if success:

            print("Motion completed successfully.")
            return 0

        print("Motion did not complete successfully.")

        failure = moveit.last_failure()

        if failure:
            print(f"Details: {failure}")

        return 4

    except KeyboardInterrupt:

        print("\nInterrupted by user.")

        if moveit is not None:
            moveit.cancel_execution()

        return 130

    except Exception as exc:

        print(f"\nError: {exc}")
        return 5

    finally:

        if moveit is not None:
            moveit.destroy()

        executor.shutdown()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        if spin_thread is not None:
            spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    sys.exit(main())
