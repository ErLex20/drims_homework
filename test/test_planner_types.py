"""Verifica i TIPI che il planner mette nelle risposte di servizio.

Esiste per una ragione precisa. rclpy fa un'asserzione sul tipo di ogni campo:
un numpy.bool_ al posto di un bool fa morire il nodo con

    AssertionError: The 'success' field must be of type 'bool'

e il sintomo non assomiglia alla causa - il behavior tree continua a girare
vedendo SERVICE_UNREACHABLE, che sembra un problema di rete o di logica.
Misurato il 2026-09-05: una missione ha girato a vuoto fino a esaurire i
tentativi perche' at_rest() restituiva un numpy.bool_.

E' lo stesso genere di errore che c'e' in robotiq_2f_adapter_node.py:381,
dove un effort=0 intero al posto di 0.0 fa abortire l'action del gripper.
Sono errori che sfuggono a qualunque lettura del codice e che un controllo di
tipo di tre righe cattura sempre.

Uso:  python3 test/test_planner_types.py
"""
import sys
import types
from pathlib import Path

import numpy as np

# il modulo importa ROS: si carica con degli stub, qui interessa solo la
# geometria e i tipi, non la comunicazione
for name in ('rclpy', 'rclpy.callback_groups', 'rclpy.executors', 'rclpy.node',
             'rclpy.qos', 'easy_motion_msgs', 'easy_motion_msgs.srv',
             'geometry_msgs', 'geometry_msgs.msg', 'moveit_msgs',
             'moveit_msgs.srv', 'sensor_msgs', 'sensor_msgs.msg', 'std_msgs',
             'std_msgs.msg', 'std_srvs', 'std_srvs.srv', 'tf2_ros'):
    module = types.ModuleType(name)
    module.__dict__.update({k: type(k, (object,), {}) for k in (
        'Node', 'ReentrantCallbackGroup', 'MultiThreadedExecutor', 'QoSProfile',
        'DurabilityPolicy', 'ReliabilityPolicy', 'DiceIdentification',
        'TransformStamped', 'GetPositionIK', 'JointState', 'Int32', 'Trigger',
        'Buffer', 'TransformBroadcaster', 'TransformListener')})
    sys.modules[name] = module

import importlib.util
here = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    'dp', here / 'drims_homework' / 'dice_planner.py')
dp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dp)

errors = []


def check(label, value, expected):
    """Il tipo dev'essere ESATTAMENTE quello, non un sottotipo: numpy.bool_
    supera isinstance(x, bool)? No - ma bool(np.True_) si', ed e' proprio la
    confusione che fa passare il bug. Si confronta type() direttamente."""
    if type(value) is not expected:
        errors.append(f"{label}: {type(value).__name__} invece di "
                      f"{expected.__name__}  (valore {value!r})")


class Log:
    def warn(self, m): pass
    def info(self, m): pass


# --- at_rest, il caso che ha morso ------------------------------------------
state = dp.DieState(Log())
matrix = np.eye(4)
matrix[:3, 3] = [0.86, 0.34, 0.017]
check('at_rest senza riferimento', state.at_rest(matrix), bool)

state.observe(matrix, 6)
check('at_rest a dado appoggiato', state.at_rest(matrix), bool)

alto = matrix.copy()
alto[2, 3] = 0.20
check('at_rest a dado sollevato', state.at_rest(alto), bool)

# --- le altre uscite del tracciatore ----------------------------------------
axis, sure = state.axis_of(6)
check('axis_of: certezza', sure, bool)
check('_fits', state._fits((0, 0, 1), 6), bool)
for a, (face, confirmed) in state.layout().items():
    check(f'layout[{a}]: faccia', face, int)
    check(f'layout[{a}]: confermata', confirmed, bool)

# --- la geometria: niente numpy nei float che finiscono nelle TF ------------
print('=== tipi delle uscite del planner')
if errors:
    print(f"FALLITO: {len(errors)} problemi")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
print(f"  at_rest, axis_of, _fits, layout: tutti tipi Python nativi")
print("PASSATO: nessun problema")
