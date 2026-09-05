"""Launcher dell'applicazione di missione - DRIMS 2026.

Avvia il planner del dado e l'esecutore del behavior tree: sono un'unita'
funzionale, non ha senso averne uno senza l'altro. La cella e il dado vanno
lanciati a parte, in due terminali separati:

    ros2 launch drims_description ur5e_1_start.launch.py fake:=true
    ros2 launch drims_dice_simulator spawn_dice.launch.py face_up:=5

Uso:
    ros2 launch drims_homework mission_start.launch.py
    ros2 launch drims_homework mission_start.launch.py target_face:=1
    ros2 launch drims_homework mission_start.launch.py tree:=benchmark.xml

NB: qui NON c'e' un argomento fake, e non e' una dimenticanza. Lo stesso
albero vale per simulazione e robot vero, perche' i valori del gripper degli
organizzatori - 0.045 aperto, 0.0 chiuso, effort al default - funzionano su
entrambi. Se un giorno qualcosa dovesse divergere davvero, il modo e' un
secondo file d'ingresso con il solo <Script> diverso, risolto come default di
'tree' da un fake:=; ma finche' non divergono, un interruttore che non commuta
niente e' solo un modo in piu' di sbagliare.

La faccia obiettivo si cambia anche a runtime, senza rilanciare nulla:
    ros2 topic pub -1 --qos-durability transient_local \
        /target_face std_msgs/msg/Int32 "{data: 4}"
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, Shutdown, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('drims_homework')

    bt_config_path_cmd = DeclareLaunchArgument(
        'bt_executer_config_path',
        default_value=os.path.join(pkg_dir, 'config', 'mission_bt_config.yaml'),
        description='Full path to the bt executer config file')

    planner_config_path_cmd = DeclareLaunchArgument(
        'planner_config_path',
        default_value=os.path.join(pkg_dir, 'config', 'dice_planner_config.yaml'),
        description='Full path to the dice planner config file')

    tree_cmd = DeclareLaunchArgument(
        'tree',
        default_value='mission.xml',
        description='Albero da eseguire: mission.xml (gara) o benchmark.xml (prove)')

    target_face_cmd = DeclareLaunchArgument(
        'target_face',
        default_value='2',
        description='Faccia obiettivo iniziale (1-6). Il topic /target_face ha la precedenza')

    planner_node = Node(
        package='drims_homework',
        executable='dice_planner',
        name='dice_planner_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            LaunchConfiguration('planner_config_path'),
            {'initial_target_face': LaunchConfiguration('target_face')},
        ],
    )

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
        # L'esecutore esce appena la radice ritorna SUCCESS o FAILURE.
        # Senza questo il launch resterebbe vivo senza piu' nulla da fare.
        on_exit=[Shutdown()],
    )

    # il planner ha bisogno di qualche istante per ricevere /joint_states,
    # risolvere dice_tf e produrre il primo piano: l'albero interroga
    # /plan_available appena parte, e senza questo ritardo lo troverebbe vuoto
    delayed_bt = TimerAction(period=6.0, actions=[bt_executer_node])

    ld = LaunchDescription()
    ld.add_action(bt_config_path_cmd)
    ld.add_action(planner_config_path_cmd)
    ld.add_action(tree_cmd)
    ld.add_action(target_face_cmd)
    ld.add_action(LogInfo(msg=['Albero: ', LaunchConfiguration('tree'),
                               ' | faccia obiettivo iniziale: ',
                               LaunchConfiguration('target_face')]))
    ld.add_action(planner_node)
    ld.add_action(delayed_bt)
    return ld
