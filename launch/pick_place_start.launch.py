"""Un solo pick and place - DRIMS 2026.

Avvia SOLO l'esecutore del behavior tree su pick_place.xml. Nessun planner:
questo albero usa pose letterali riferite a dice_tf e non dipende da
FACE_LAYOUT, quindi e' quello da usare per la messa in servizio sull'hardware.

La cella, il dado (in simulazione) o la camera piu' il detector (sul robot
vero) vanno lanciati a parte:

    ros2 launch drims_description ur5e_1_start.launch.py fake:=false
    ros2 launch drims_description robotiq_hande_urcap_adapter.launch.py robot_ip:=192.168.254.101
    ros2 run dice_detector dice_detector --ros-args -p top_face_distance:=0.585

Uso:
    ros2 launch drims_homework pick_place_start.launch.py

Nessun argomento fake: lo stesso albero vale per simulazione e robot vero,
perche' i valori del gripper degli organizzatori funzionano su entrambi.
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('drims_homework')

    config_path_cmd = DeclareLaunchArgument(
        'bt_executer_config_path',
        default_value=os.path.join(pkg_dir, 'config', 'pick_place_bt_config.yaml'),
        description='Full path to the bt executer config file')

    tree_cmd = DeclareLaunchArgument(
        'tree',
        default_value='pick_place.xml',
        description='Albero da eseguire')

    bt_executer_node = Node(
        package='easy_motion_behavior_tree',
        executable='bt_executer_node',
        # NON togliere name=: il nodo si chiama 'bt_executor_node' nel C++,
        # mentre la chiave dello yaml e' 'bt_executer_node'. E' questo rename
        # che fa combaciare le due cose; senza, i parametri non vengono letti.
        name='bt_executer_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            LaunchConfiguration('bt_executer_config_path'),
            {'bt_xml_file': LaunchConfiguration('tree')},
        ],
        on_exit=[Shutdown()],
    )

    ld = LaunchDescription()
    ld.add_action(config_path_cmd)
    ld.add_action(tree_cmd)
    ld.add_action(LogInfo(msg=['Pick and place singolo: ',
                               LaunchConfiguration('tree')]))
    ld.add_action(bt_executer_node)
    return ld
