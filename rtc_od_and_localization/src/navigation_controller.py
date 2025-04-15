
"""

Purpose:

    This node is responsible for making the robot navigate to the specified goal position (e.g., (2.5, 0, 0)).

Responsibilities:

    Receive Goal Pose:

        The node subscribes to /goal_pose or a custom topic where a goal position (in the map frame) is published.

    Path Planning:

        Use Nav2 or a custom controller to plan a path to the goal.

        Handle obstacles (use RTAB-Map's map or obstacles detected by the object detection node).

    Localization:

        Track the robot’s position via RTAB-Map localization (pose from /tf).

    Movement Control:

        Use a PID controller or a robot-specific controller to move towards the goal.

    Publish Status:

        Publish robot status on /rosout, and update on navigation progress.

    Reach Goal:

        Ensure the robot arrives within the specified tolerance (30 cm) of the goal.

"""