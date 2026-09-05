"""Planner del dado - DRIMS 2026.

Decide COME afferrare e ruotare il dado; l'albero di comportamento si limita a
sequenziare. Tutta la geometria vive qui, l'albero non ne contiene.


PERCHE' SERVE UN PLANNER, E NON UNA TABELLA

Il problema vero non e' muovere il robot: e' che da vista zenitale si vede UNA
faccia. Le quattro laterali e quella sotto sono nascoste, e i pattern dei pip
sono simmetrici per rotazione, quindi nemmeno l'immagine dice quale numero sta
su quale lato. Con questa sola informazione non esiste una scelta corretta
dell'asse di presa: qualunque cosa si faccia al primo ciclo e' un tentativo.

Una versione precedente teneva una tabella MISURATA a mano sul simulatore. E'
diventata falsa due volte in un giorno - il fix dell'orientamento del dado nel
simulatore, e la rotazione della cella di -90 gradi - continuando a produrre
piani plausibili e sbagliati senza mai un errore. Sul robot vero non sarebbe
nemmeno ricostruibile: il dado della gara non e' quello del modello.

Una versione ancora precedente rinunciava del tutto e provava le quattro prese
a rotazione. Misurato in simulazione: obiettivo 3 raggiunto in 5 cicli, ma
obiettivo 1 MAI in 12, perche' la posa di rilascio conserva l'imbardata e la
mappa (faccia, presa) -> faccia e' deterministica; la sequenza cade in
un'orbita chiusa - osservata 3,6,3,5,6,3,5,6,4,6,4,5 - che non tocca
l'obiettivo. Una passeggiata deterministica su un grafo puo' ciclare.

Quello che segue e' la terza risposta, ed e' quella giusta: l'informazione che
manca al primo ciclo NON manca al secondo. Ogni flip e' anche una misura.


COME

Si mantiene una stima dell'orientamento del dado, R_die, e una mappa
LAYOUT: asse del corpo -> numero di faccia, che si RIEMPIE da sola.

  osservazione   la faccia in alto dice quale numero sta sull'asse del corpo
                 che punta in su; la faccia opposta e' 7 meno quella, quindi
                 ogni osservazione lega DUE assi
  predizione     il flip e' una rotazione COMANDATA, quindi nota: dopo di esso
                 R_die <- Rot(n, flip) * R_die, e il rilascio conserva
                 l'orientamento
  riaggancio     alla nuova osservazione si sceglie, fra le 24 rotazioni del
                 cubo, quella che rende la stima piu' vicina alla predizione.
                 E' quel passo che risolve l'ambiguita' di imbardata modulo 90
                 gradi che l'immagine da sola non puo' risolvere

Con due assi noti su tre il resto si deduce: un dado standard occidentale
soddisfa n1 x n2 = n3, cioe' con l'1 in alto e il 2 verso di se' il 3 sta a
destra. Quattro facce note fissano l'orientamento del dado canonico in modo
unico, quindi le ultime due si leggono per costruzione. Sono IPOTESI, marcate
come tali: se il dado non e' standard la prossima osservazione le corregge e
non si perde nulla di confermato.

Costo: due cicli di esplorazione nel caso peggiore, poi uno solo per
raggiungere qualunque faccia. Contro i 5-12 misurati sulla versione a
tentativi.


LE SINGOLARITA'

Il flip fa passare l'utensile da verticale a orizzontale con il dado in mano.
E' esattamente il tipo di moto che porta dentro una singolarita' del polso, e
li' una rotazione cartesiana piccola richiede giri di giunto grandi: misurato
il 2026-09-02, wrist_1 da -2.036 a +5.258, 418 gradi percorsi in 19 secondi di
cui 2*pi di puro srotolamento.

Tre meccanismi, tutti necessari e nessuno sufficiente da solo:

1. wrap_near riporta ogni soluzione IK al giro piu' vicino allo stato di
   partenza. Toglie lo srotolamento LETTERALE, quello da +5.258 a -1.025.

2. wrist_flip genera l'altra soluzione del polso sferico, (t4+pi, -t5, t6+pi).
   Il solver LMA, pur seminato sullo stato corrente, salta spesso a quella
   lontana. Generandola per via analitica e scegliendo la piu' vicina si
   smette di dipendere dalla fortuna del solver.

3. Il percorso si CAMPIONA e si risolve a catena, ogni punto seminato sulla
   soluzione del precedente, e si scarta la variante in cui un giunto
   percorre piu' del limite fra due campioni vicini. Questo e' il punto:
   vicino a una singolarita' un passo cartesiano piccolo COSTA molto in
   giunti, quindi il test misura la condizione della catena lungo il percorso
   invece di indovinarla da una formula. Vale per la singolarita' di polso,
   di gomito e di spalla senza doverle distinguere. Valutare solo le quattro
   pose estreme, come faceva la versione precedente, permette di scavalcare
   una singolarita' senza accorgersene.

In piu' c'e' una liberta' che si puo' spendere: la presa e' simmetrica.
Afferrare con l'asse delle dita lungo +n e ruotare di +90 gradi da' lo STESSO
risultato di afferrare lungo -n e ruotare di -90 - le dita si scambiano, il
dado no. Sono due configurazioni di polso molto diverse a parita' di esito, e
si sceglie quella che costa meno. E' gratis, ed e' spesso la differenza fra un
flip pulito e un mezzo giro di polso.
"""

import threading
import time

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation, Slerp

from easy_motion_msgs.srv import DiceIdentification
from geometry_msgs.msg import Pose, TransformStamped
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetCartesianPath, GetPositionIK, GetStateValidity
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformBroadcaster, TransformListener

TWO_PI = 2.0 * np.pi
UP = np.array([0.0, 0.0, 1.0])

# I sei assi del corpo del dado, come terne intere.
AXES = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

# Dado standard occidentale: 1 in alto, 2 verso di se', 3 a destra. In termini
# di normali uscenti questo e' n1 x n2 = n3, e fissa la chiralita'. Serve solo
# per DEDURRE la terza coppia quando due sono note; ogni valore dedotto resta
# un'ipotesi finche' non lo si osserva.
CANONICAL = {(1, 0, 0): 3, (-1, 0, 0): 4,
             (0, 1, 0): 5, (0, -1, 0): 2,
             (0, 0, 1): 1, (0, 0, -1): 6}


def cube_rotations():
    """Le 24 rotazioni proprie del cubo, come matrici a coefficienti interi."""
    out = []
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        for signs in np.ndindex(2, 2, 2):
            matrix = np.zeros((3, 3), dtype=int)
            for row, column in enumerate(perm):
                matrix[row, column] = 1 - 2 * signs[row]
            if round(np.linalg.det(matrix)) == 1:
                out.append(matrix)
    return out


ROTATIONS = cube_rotations()


def to_matrix(translation, quaternion):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    matrix[:3, 3] = translation
    return matrix


def from_matrix(matrix):
    return matrix[:3, 3].copy(), Rotation.from_matrix(matrix[:3, :3]).as_quat()


def snap_axis(vector):
    """L'asse intero del corpo piu' vicino a un vettore."""
    index = int(np.argmax(np.abs(vector)))
    axis = [0, 0, 0]
    axis[index] = 1 if vector[index] > 0 else -1
    return tuple(axis)


def wrap_near(value, reference, limit=TWO_PI):
    """Riporta value al giro piu' vicino a reference, rispettando i limiti.

    E' la correzione dello srotolamento: una soluzione IK a +5.258 rad e' la
    stessa posa di una a -1.025, ma percorrerla costa 2*pi in piu'.
    """
    candidate = reference + np.arctan2(
        np.sin(value - reference), np.cos(value - reference))
    if abs(candidate) <= limit:
        return candidate
    for shift in (-TWO_PI, TWO_PI):
        if abs(candidate + shift) <= limit:
            return candidate + shift
    return None


def wrist_flip(solution):
    """L'altra soluzione del polso sferico: (t4+pi, -t5, t6+pi)."""
    out = np.array(solution, dtype=float)
    out[3] += np.pi
    out[4] = -out[4]
    out[5] += np.pi
    return out


def best_variant(raw, seed, limit):
    """Fra la soluzione grezza e l'alternativa di polso, entrambe riportate al
    giro piu' vicino al seme, quella che costa meno."""
    best = None
    for candidate in (np.array(raw, dtype=float), wrist_flip(raw)):
        wrapped = np.empty_like(candidate)
        ok = True
        for i, (value, reference) in enumerate(zip(candidate, seed)):
            w = wrap_near(value, reference, limit)
            if w is None:
                ok = False
                break
            wrapped[i] = w
        if ok:
            cost = float(np.max(np.abs(wrapped - seed)))
            if best is None or cost < best[0]:
                best = (cost, wrapped)
    return best


def tool_frame(finger_axis):
    """Orientamento di presa: utensile a puntare in basso, dita lungo
    finger_axis. E' la costruzione diretta della matrice invece di un angolo
    theta attorno all'asse di avvicinamento, perche' cio' che il piano deve
    fissare e' proprio l'asse delle dita: e' quello che sceglie su quale
    coppia di facce si stringe, e quindi attorno a cosa ruota il dado."""
    x = np.asarray(finger_axis, dtype=float)
    x = x / np.linalg.norm(x)
    z = -UP
    y = np.cross(z, x)
    return np.column_stack((x, y, z))


class DieState:
    """Stima dell'orientamento del dado e mappa asse -> faccia.

    Non e' un filtro: la rotazione comandata e' nota esattamente e il
    riaggancio e' una scelta discreta fra 24 possibilita'. L'unica incertezza
    e' se il dado rimbalza al rilascio, e in quel caso il riaggancio sceglie
    semplicemente un'altra rotazione, coerente con cio' che si vede.
    """

    def __init__(self, logger):
        self.log = logger
        self.rotation = None            # corpo -> mondo
        self.confirmed = {}             # asse del corpo -> faccia osservata
        self.resting_z = None
        self.pending = None             # rotazione comandata non ancora vista
        self.last_face = None

    # ------------------------------------------------------------- ingresso

    def expect(self, rotation):
        """Registra la rotazione che il ciclo in corso applichera' al dado.

        E' l'unico ingresso di informazione che NON viene dai sensori, ed e'
        anche il piu' affidabile: il flip e' comandato, non stimato.
        """
        self.pending = rotation

    def observe(self, matrix, face):
        """matrix: posa osservata del dado nel frame base. face: faccia in alto.

        Il riferimento del riaggancio e' la PREDIZIONE, non la stima vecchia.
        Senza questo passo il riaggancio riporterebbe sempre la stima
        sull'orientamento precedente - le due sono legate da una simmetria del
        cubo, quindi geometricamente indistinguibili - e la mappa si
        contraddirebbe a ogni ciclo. Misurato in simulazione il 2026-09-04:
        'asse (0,0,1): atteso 5, visto 6. Mappa azzerata.' a ogni flip.

        A dire QUANDO applicare la predizione e' il numero di faccia, non la
        geometria: un flip di 90 gradi cambia sempre la faccia in alto, quindi
        faccia diversa significa flip avvenuto, faccia uguale significa che il
        ciclo non ha ancora spostato il dado.
        """
        observed = matrix[:3, :3]
        z = matrix[2, 3]
        self.resting_z = z if self.resting_z is None else min(self.resting_z, z)

        if self.rotation is None:
            # primo aggancio: il frame del corpo si DEFINISCE come quello
            # osservato adesso. Non serve che sia in accordo con niente, serve
            # solo che da qui in poi resti lo stesso.
            self.rotation = observed
        else:
            reference = self.rotation
            if self.pending is not None and face != self.last_face:
                reference = self.pending @ self.rotation
                self.pending = None
            # Fra le 24 rotazioni del cubo si sceglie prima la COERENZA con
            # cio' che si e' gia' visto, poi la vicinanza alla predizione. Le
            # due quasi sempre coincidono; quando divergono e' perche' il dado
            # ha rimbalzato al rilascio, e li' la mappa e' il dato piu' solido
            # dei due.
            best = None
            for M in ROTATIONS:
                candidate = observed @ M.T
                axis = snap_axis(candidate.T @ UP)
                angle = np.linalg.norm(
                    Rotation.from_matrix(candidate @ reference.T).as_rotvec())
                key = (0 if self._fits(axis, face) else 1, angle)
                if best is None or key < best[0]:
                    best = (key, candidate)
            (consistent, angle), self.rotation = best
            if angle > np.deg2rad(50.0):
                self.log.warn(
                    f"riaggancio a {np.rad2deg(angle):.0f} gradi dalla "
                    f"predizione: il dado ha probabilmente rimbalzato")

        self.last_face = face
        axis = snap_axis(self.rotation.T @ UP)
        opposite = tuple(-v for v in axis)
        if not self._fits(axis, face):
            # nessun riaggancio riesce a conciliare l'osservazione con la
            # mappa: e' andata fuori sincrono (dado sostituito, o un rimbalzo
            # interpretato male). Si tiene cio' che si vede ADESSO e si butta
            # il resto, invece di trascinare una contraddizione.
            self.log.warn(f"osservazione incompatibile con la mappa "
                          f"(asse {axis}, faccia {face}): mappa azzerata.")
            self.confirmed.clear()
        self.confirmed[axis] = face
        self.confirmed[opposite] = 7 - face

    def _fits(self, axis, face):
        """L'osservazione (asse in su, faccia) e' compatibile con la mappa?

        Due condizioni, e la SECONDA mancava. Misurato sul robot vero il
        2026-09-04: la mappa e' finita con +x=6 e +z=6 insieme, cioe' la
        faccia 6 su due assi non opposti, che e' geometricamente impossibile.

        Ci si arriva cosi': il dado non si muove, ma l'imbardata osservata
        salta di ~90 gradi (instabilita' del rilevatore, misurata a 73 gradi di
        escursione). Il riaggancio la legge come una rotazione vera, l'asse in
        su diventa +x, e si scrive +x=6 senza che nulla cancelli +z=6, perche'
        il controllo guardava solo se lo STESSO asse riceveva una faccia
        DIVERSA. Guardare anche il contrario - la stessa faccia su un altro
        asse - non serve solo a tenere la mappa pulita: entra nel punteggio
        del riaggancio, e quindi SCARTA a priori la rotazione spuria di 90
        gradi invece di accettarla e poi convivere con la contraddizione.
        """
        opposite = tuple(-v for v in axis)
        known = self.confirmed.get(axis)
        if known is not None and known != face:
            return False
        for other, value in self.confirmed.items():
            if other in (axis, opposite):
                continue
            if value in (face, 7 - face):
                return False
        return True

    def at_rest(self, matrix):
        """Il dado e' appoggiato? Se e' in mano al robot l'osservazione non va
        usata: aggiornerebbe la mappa con una posa che il rilascio cambiera'."""
        if self.resting_z is None:
            return True
        # bool() esplicito: il confronto fra numpy float da' un numpy.bool_, e
        # rclpy lo RIFIUTA con "The 'success' field must be of type 'bool'"
        # quando finisce dentro una risposta di servizio. Misurato il
        # 2026-09-05: il planner moriva a meta' missione e il BT continuava a
        # girare vedendo SERVICE_UNREACHABLE, cioe' un fallimento che sembrava
        # del comportamento e invece era un tipo sbagliato.
        return bool(matrix[2, 3] <= self.resting_z + 0.02)

    # -------------------------------------------------------------- lettura

    def layout(self):
        """Mappa completa asse -> (faccia, confermata). Le voci non osservate
        si deducono dal dado standard SOLO se la deduzione e' unica.

        La condizione di unicita' non e' un dettaglio. Con una sola coppia
        osservata - cioe' dopo la prima occhiata - restano QUATTRO rotazioni
        del cubo compatibili, le quattro attorno alla verticale, e sceglierne
        una vale un tiro di dado a un quarto. Misurato in simulazione il
        2026-09-04: con {+z:5, -z:2} noti la deduzione dava +y=6, mentre il
        dado aveva +y=3. Un'ipotesi giusta al 25 per cento presentata come
        candidato prioritario fa sprecare il ciclo che invece, esplorando, si
        sarebbe ripagato con una misura certa.

        Con due coppie osservate - quattro facce - la rotazione compatibile e'
        una sola, e a quel punto la deduzione vale.
        """
        full = {axis: (face, True) for axis, face in self.confirmed.items()}
        if len(full) >= 6:
            return full
        consistent = []
        for M in ROTATIONS:
            mapped = {snap_axis(M @ np.array(axis)): face
                      for axis, face in CANONICAL.items()}
            if all(mapped[axis] == face for axis, face in self.confirmed.items()):
                consistent.append(mapped)
        if len(consistent) == 1:
            for axis, face in consistent[0].items():
                full.setdefault(axis, (face, False))
        return full

    def axis_of(self, face):
        """L'asse del corpo che porta una faccia, e se e' un dato certo."""
        for axis, (value, sure) in self.layout().items():
            if value == face:
                return axis, sure
        return None, False

    def unknown_axes(self):
        return [axis for axis in AXES if axis not in self.confirmed]


class DicePlanner(Node):

    def __init__(self):
        super().__init__('dice_planner_node')

        self.declare_parameter('base_frame', 'table_top')
        self.declare_parameter('dice_frame', 'dice_tf')
        self.declare_parameter('ee_frame', 'tip')
        self.declare_parameter('move_group', 'manipulator')
        self.declare_parameter('joint_names', [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'])

        self.declare_parameter('dice_size', 0.030)
        # Quanto sotto il CENTRO del dado va portato il TCP alla presa. Non
        # zero: il frame tip sta all'estremita' delle dita, quindi centrarlo
        # sul dado lascerebbe i polpastrelli sopra lo spigolo superiore.
        self.declare_parameter('grasp_depth', 0.0135)
        self.declare_parameter('pregrasp_height', 0.10)
        # Quota di sollevamento prima di ruotare.
        #
        # NOTA NEGATIVA, misurata e da non ripetere: avevo provato a rendere
        # questa quota un grado di liberta' del piano - tre valori, scelti
        # dall'IK - sperando di recuperare le prese risolutive che il filtro
        # di singolarita' scarta. Su 36 combinazioni il piano scelto e' stato
        # 0.10 in TUTTI i 71 casi, e i cicli persi sono rimasti 12. Il
        # cattivo condizionamento nasce dalla ROTAZIONE del polso, non dalla
        # quota a cui la si esegue: alzare il punto del flip trasla il braccio
        # ma il polso deve spazzare gli stessi 90 gradi. Costo triplicato in
        # chiamate IK, beneficio nullo.
        self.declare_parameter('lift_height', 0.10)
        self.declare_parameter('place_clearance', 0.030)

        self.declare_parameter('joint_limit', TWO_PI)
        # Deve corrispondere a cartesian_max_step del motion server
        # (drims_description/config/ur5e/motion_server_config.yaml): e' il
        # passo con cui computeCartesianPath campionera' davvero il percorso.
        self.declare_parameter('cartesian_max_step', 0.01)
        # Anche questa deve corrispondere al motion server: e' la frazione di
        # percorso sotto la quale RIFIUTA il moto. Serve al planner per porsi
        # esattamente la stessa domanda prima di proporre un piano.
        self.declare_parameter('cartesian_fraction_threshold', 0.99)
        # Rilevatore di singolarita', espresso come rad di giunto per METRO di
        # percorso cartesiano - NON per campione.
        #
        # Era una soglia per campione (0.6 rad), e finche' il campionamento era
        # una costante andava bene. Ora i campioni si ricavano da
        # cartesian_max_step, quindi una soglia per campione cambierebbe
        # significato al variare del passo: misurato il 2026-09-04, passando da
        # 25 mm a 10 mm di passo i valori osservati sono scesi da 0.34-0.40 a
        # 0.16-0.17 - lo stesso fattore 2.5 - e con la soglia ferma a 0.6 il
        # filtro sarebbe diventato 2.5 volte piu' permissivo senza che nessuno
        # lo avesse deciso.
        #
        # 24 rad/m e' esattamente la vecchia soglia riportata al metro:
        # 0.6 rad / 0.025 m. Cosi' il criterio resta lo stesso di quando e'
        # stato validato, e smette di dipendere dalla densita' di campionamento.
        self.declare_parameter('joint_rate_limit', 24.0)
        # Il pregrasp e' un moto LIBERO in spazio giunti da una posa lontana:
        # li' un riallineamento di polso di mezzo giro e' legittimo e non
        # dice niente sul condizionamento.
        self.declare_parameter('max_step_free', 3.3)
        # Sotto questa soglia wrist_2 allinea gli assi 4 e 6: il polso perde
        # un grado di liberta' e l'IK diventa mal condizionata.
        self.declare_parameter('wrist_singularity_margin', 0.10)

        self.declare_parameter('ik_timeout', 0.05)
        # false di proposito: qui si valuta la CINEMATICA, non le collisioni.
        # A tempo di pianificazione lo stato del gripper puo' essere quello
        # del ciclo precedente, e avere le collisioni attive darebbe scarti
        # spuri. Le collisioni le verifica il motion server quando pianifica.
        self.declare_parameter('ik_avoid_collisions', False)

        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('poll_rate', 4.0)
        self.declare_parameter('plan_rate', 2.0)
        # Oltre questo tempo dalla presa in carico di un piano, se la faccia
        # non e' cambiata il ciclo e' fallito e si ripianifica. Un ciclo sano
        # dura ~20 s in simulazione.
        self.declare_parameter('busy_timeout', 45.0)
        self.declare_parameter('initial_target_face', 0)

        get = lambda name: self.get_parameter(name).value
        self.base_frame = get('base_frame')
        self.dice_frame = get('dice_frame')
        self.ee_frame = get('ee_frame')
        self.move_group = get('move_group')
        self.joint_names = list(get('joint_names'))
        self.dice_size = get('dice_size')
        self.grasp_depth = get('grasp_depth')
        self.pregrasp_height = get('pregrasp_height')
        self.lift_height = get('lift_height')
        self.place_clearance = get('place_clearance')
        self.joint_limit = get('joint_limit')
        self.joint_rate_limit = get('joint_rate_limit')
        # soglia per campione, derivata: e' cio' che il confronto usa
        self.max_step_cartesian = self.joint_rate_limit * get('cartesian_max_step')
        self.max_step_free = get('max_step_free')
        self.wrist_margin = get('wrist_singularity_margin')
        self.cartesian_max_step = get('cartesian_max_step')
        self.cartesian_fraction = get('cartesian_fraction_threshold')
        self.ik_timeout = get('ik_timeout')
        self.ik_avoid_collisions = get('ik_avoid_collisions')
        self.busy_timeout = get('busy_timeout')

        initial = int(get('initial_target_face'))
        self.lock = threading.Lock()
        self.state = DieState(self.get_logger())
        self.face = None
        self.at_rest = True     # finche' non si osserva, si assume appoggiato
        self.target = initial if 1 <= initial <= 6 else None
        self.joints = None
        self.candidate = None      # miglior piano per l'osservazione corrente
        self.latched = None        # cio' che si pubblica, congelato per il ciclo
        self.reject = 'avvio'
        # Istante in cui l'albero ha preso in carico un piano. Finche' il
        # ciclo e' in corso non c'e' NIENTE da ripianificare: il dado non si
        # muove piu' rispetto a quanto gia' deciso, e continuare a valutare
        # varianti significa solo sottrarre chiamate IK a move_group proprio
        # mentre sta pianificando i moti veri. Una valutazione completa costa
        # fino a ~150 chiamate a /compute_ik, e a 2 Hz sarebbero ~300 al
        # secondo in sottofondo.
        self.busy_since = None

        group = ReentrantCallbackGroup()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        latched_qos = QoSProfile(depth=1,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Int32, '/target_face', self._on_target,
                                 latched_qos, callback_group=group)
        self.create_subscription(JointState, '/joint_states', self._on_joints,
                                 10, callback_group=group)

        self.dice_client = self.create_client(
            DiceIdentification, '/dice_identification', callback_group=group)
        self.ik_client = self.create_client(
            GetPositionIK, '/compute_ik', callback_group=group)
        # Gli stessi due servizi che usera' il motion server per decidere se
        # il moto si puo' fare. Vedi _feasible.
        self.cartesian_client = self.create_client(
            GetCartesianPath, '/compute_cartesian_path', callback_group=group)
        self.validity_client = self.create_client(
            GetStateValidity, '/check_state_validity', callback_group=group)

        self.create_service(Trigger, '/goal_reached', self._on_goal_reached,
                            callback_group=group)
        self.create_service(Trigger, '/plan_available', self._on_plan_available,
                            callback_group=group)

        self.create_timer(1.0 / get('publish_rate'), self._broadcast,
                          callback_group=group)

        self._poll_period = 1.0 / get('poll_rate')
        self._plan_period = 1.0 / get('plan_rate')
        threading.Thread(target=self._poll_loop, daemon=True).start()
        threading.Thread(target=self._plan_loop, daemon=True).start()

        self.get_logger().info(
            f"Dice planner avviato. base={self.base_frame} dado={self.dice_frame} "
            f"obiettivo iniziale={self.target}")

    # ------------------------------------------------------------- ingressi

    def _on_target(self, msg):
        with self.lock:
            if self.target != msg.data:
                self.get_logger().info(f"Faccia obiettivo: {msg.data}")
            self.target = int(msg.data)

    def _on_joints(self, msg):
        table = dict(zip(msg.name, msg.position))
        if all(name in table for name in self.joint_names):
            with self.lock:
                self.joints = np.array([table[n] for n in self.joint_names])

    def _call(self, client, request, timeout=5.0):
        """call_async piu' attesa passiva. Niente spin annidati: l'esecutore
        multithread serve il future, questo thread lo sonda soltanto. E' il
        modo per non incorrere nel deadlock classico di rclpy quando si chiama
        un servizio dall'interno di un altro."""
        if not client.service_is_ready():
            return None
        future = client.call_async(request)
        deadline = time.time() + timeout
        while not future.done() and time.time() < deadline:
            time.sleep(0.002)
        return future.result() if future.done() else None

    def _dice_matrix(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame, self.dice_frame, rclpy.time.Time())
        except Exception:
            return None
        t = transform.transform.translation
        r = transform.transform.rotation
        return to_matrix([t.x, t.y, t.z], [r.x, r.y, r.z, r.w])

    def _poll_loop(self):
        """Identifica il dado e, SOLO quando e' appoggiato, aggiorna la stima.

        Ignorare le osservazioni a dado sollevato non e' prudenza generica:
        il rilascio cambiera' comunque quella posa, quindi imparare da essa
        significherebbe imparare qualcosa di gia' scaduto.
        """
        while rclpy.ok():
            time.sleep(self._poll_period)
            try:
                response = self._call(self.dice_client,
                                      DiceIdentification.Request())
            except Exception as exc:
                self.get_logger().warn(f"identificazione: {exc}")
                continue
            if response is None or not response.success:
                continue
            matrix = self._dice_matrix()
            if matrix is None:
                continue
            with self.lock:
                # La faccia si aggiorna SEMPRE; e' se ci si puo' CREDERE che
                # dipende dal fatto che il dado sia appoggiato, ed e' un'altra
                # cosa - la distinzione la fa /goal_reached, non questo poll.
                #
                # Prima qui c'era un 'continue' quando il dado non era a
                # riposo, con l'intento giusto (a mezz'aria la faccia esibita
                # non dice come il dado si posera') ma con una conseguenza che
                # non avevo previsto: se un ciclo fallisce CON IL DADO IN MANO,
                # il dado resta in aria, at_rest non torna mai vero, e
                # self.face resta congelata per sempre. Misurato il
                # 2026-09-05: il servizio rispondeva "Face number: 3", cioe'
                # l'obiettivo raggiunto, mentre il planner diceva ancora
                # "current=4" e la missione girava a vuoto fino a esaurire i
                # tentativi.
                self.at_rest = self.state.at_rest(matrix)
                changed = self.face != int(response.face_number)
                self.face = int(response.face_number)
                if not self.at_rest:
                    # niente mappa e niente sblocco del ciclo: quelli si
                    # aggiornano solo su un'osservazione a dado appoggiato
                    continue
                if changed and self.busy_since is not None:
                    # la faccia e' cambiata a dado appoggiato: il ciclo ha
                    # prodotto il suo effetto, si puo' tornare a pianificare
                    self.busy_since = None
                known_before = len(self.state.confirmed)
                self.state.observe(matrix, self.face)
                if len(self.state.confirmed) > known_before:
                    self.get_logger().info(
                        f"Mappa del dado: {self._layout_text()}")

    def _layout_text(self):
        names = {(1, 0, 0): '+x', (-1, 0, 0): '-x', (0, 1, 0): '+y',
                 (0, -1, 0): '-y', (0, 0, 1): '+z', (0, 0, -1): '-z'}
        layout = self.state.layout()
        return ' '.join(f"{names[a]}={layout[a][0]}{'' if layout[a][1] else '?'}"
                        for a in AXES if a in layout)

    # -------------------------------------------------------------- servizi

    def _on_goal_reached(self, request, response):
        """L'obiettivo e' raggiunto solo con il dado APPOGGIATO.

        Le due condizioni sono separate apposta: la faccia e' sempre
        l'osservazione piu' fresca, ma vale come risposta solo quando il dado
        e' fermo sul piano. A mezz'aria fra le dita la faccia esibita non dice
        come il dado si posera'.
        """
        with self.lock:
            face, target, at_rest = self.face, self.target, self.at_rest
        response.message = (f"current={face} target={target}"
                            + ("" if at_rest else " (dado non appoggiato)"))
        response.success = bool(face is not None and target is not None
                                and face == target and at_rest)
        return response

    def _on_plan_available(self, request, response):
        """Congela il piano corrente per la durata del ciclo.

        Il congelamento e' voluto: l'albero chiama questo servizio una volta
        per ciclo, subito prima di muoversi, e da li' in poi le TF non devono
        piu' cambiare sotto i piedi di chi le sta inseguendo.
        """
        with self.lock:
            candidate, reject = self.candidate, self.reject
            if self.busy_since is not None:
                # L'albero richiede un piano mentre un ciclo risulta ancora in
                # corso: significa che quel ciclo e' fallito senza spostare il
                # dado (presa non riuscita, moto abortito, Recover). Il piano
                # in mano e' quello che ha appena fallito, quindi si riapre
                # subito il cancello: questo tentativo riusa il piano vecchio,
                # ma il successivo ne trova gia' uno nuovo. Senza questa
                # riapertura l'albero ritenterebbe lo STESSO piano fino allo
                # scadere del tempo di guardia, bruciando tentativi.
                self.busy_since = None
                self.get_logger().warn(
                    'nuovo piano richiesto a ciclo non concluso: il ciclo '
                    'precedente non ha spostato il dado')
            if candidate is not None:
                self.latched = candidate
                self.busy_since = time.time()
                # da qui in poi il tracciatore sa cosa aspettarsi: e' quello
                # che gli permette di riagganciare la stima dopo il flip
                self.state.expect(candidate['rotation'])
        if candidate is None:
            response.success = False
            response.message = reject
            return response
        response.success = True
        response.message = candidate['why']
        self.get_logger().info(response.message)
        return response

    # ------------------------------------------------------------- geometria

    def _chain(self, dice, finger_axis, flip_deg, lift_height):
        """Le pose del ciclo, nel frame base.

        Tutto discende dalla posa dell'utensile alla presa. Da li' in poi il
        dado e' RIGIDAMENTE attaccato all'utensile, quindi la sua posa e'
        sempre T_tool * dice_in_tool, e la posa di rilascio si ottiene
        invertendo: si decide dove deve stare il DADO e si ricava dove deve
        stare l'utensile. E' l'unico modo di non sbagliare la quota di
        rilascio, perche' l'offset di presa RUOTA con l'utensile: col flip di
        90 gradi diventa orizzontale e smette di contribuire all'altezza.
        """
        half = self.dice_size / 2.0

        # dice_tf sta sulla FACCIA SUPERIORE del dado, NON sul centro.
        # Verificato sul simulatore il 2026-09-04 con la cella 1:
        #     table_top -> dice_tf       z = 0.050
        #     table_top -> dice_com_tf   z = 0.035   <- il centro
        # quindici millimetri esatti, cioe' mezzo spigolo. Il rilevatore vero
        # usa la stessa convenzione: costruisce il frame dal piano della faccia
        # superiore, a top_face_distance dalla telecamera.
        #
        # Fin qui il codice trattava dice_tf come il centro. Con grasp_depth a
        # 0.0135 l'errore si compensava per caso - il TCP finiva 1.5 mm sopra
        # il centro, cioe' una presa buona, ed e' il motivo per cui le 36 corse
        # passavano - ma surface_z veniva calcolata 15 mm troppo in alto, e il
        # rilascio avveniva 15 mm piu' su di quanto place_clearance dichiarasse.
        # Convertendo qui una volta sola, tutto il resto torna a dire il vero.
        #
        # Si usa l'asse Z PROPRIO del frame del dado, non la verticale del
        # mondo: coincidono a dado appoggiato, ma cosi' la conversione resta
        # esatta anche se il dado e' inclinato o se lo si guarda mentre e' in
        # mano al robot.
        centre = dice[:3, 3] - half * (dice[:3, :3] @ np.array([0.0, 0.0, 1.0]))

        rotation = tool_frame(finger_axis)

        grasp = np.eye(4)
        grasp[:3, :3] = rotation
        grasp[:3, 3] = centre - self.grasp_depth * UP

        pregrasp = grasp.copy()
        pregrasp[:3, 3] = centre + self.pregrasp_height * UP

        flip = grasp.copy()
        flip[:3, :3] = rotation @ Rotation.from_euler(
            'x', flip_deg, degrees=True).as_matrix()
        flip[:3, 3] = grasp[:3, 3] + lift_height * UP

        dice_in_tool = np.linalg.inv(grasp) @ dice
        held = flip @ dice_in_tool

        # la quota del piano si legge dal dado OSSERVATO, non da un piano
        # assunto a zero: misurato in simulazione il 2026-09-03, un dado a
        # riposo su table_top ha il centro a z=0.090, non a 0.015
        surface_z = centre[2] - half
        target_dice = held.copy()
        target_dice[2, 3] = surface_z + half + self.place_clearance
        place = target_dice @ np.linalg.inv(dice_in_tool)

        return {'pregrasp': pregrasp, 'grasp': grasp,
                'flip': flip, 'place': place}

    def _path(self, chain):
        """Il percorso campionato, come lista di (etichetta, posa, limite).

        Il campionamento riproduce quello che fara' computeCartesianPath:
        posizione interpolata linearmente, orientamento per slerp. Serve a
        misurare il condizionamento LUNGO il percorso invece che nei soli
        estremi, che e' il modo per non scavalcare una singolarita' senza
        accorgersene.
        """
        path = [('pregrasp', chain['pregrasp'], self.max_step_free)]
        for label, start, end in (
                ('discesa', chain['pregrasp'], chain['grasp']),
                ('flip', chain['grasp'], chain['flip']),
                ('posa', chain['flip'], chain['place'])):
            # Il numero di campioni NON e' un parametro nostro: si ricava da
            # cartesian_max_step del motion server, cosi' il planner verifica
            # gli STESSI punti che computeCartesianPath chiedera'.
            #
            # Prima erano tre costanti (4, 7, 4) scelte a occhio. Misurato il
            # 2026-09-04: sulla discesa il planner metteva 4 punti dove MoveIt
            # ne chiede ~10, cioe' passi di 25 mm contro 10 mm, un punto ogni
            # 2.5. Verificare meno punti di quanti ne verranno richiesti
            # significa poter scavalcare una configurazione difficile senza
            # vederla, dichiarare il percorso sano, e poi vederselo rifiutare
            # in 60 ms - che e' esattamente il guasto osservato.
            #
            # Si conta anche l'ARCO della rotazione, non solo la traslazione:
            # nel flip la traslazione e' piccola ma il polso spazza 90 gradi,
            # ed e' li' che sta il condizionamento peggiore.
            span = float(np.linalg.norm(end[:3, 3] - start[:3, 3]))
            angle = float(np.linalg.norm(Rotation.from_matrix(
                end[:3, :3] @ start[:3, :3].T).as_rotvec()))
            # l'arco di un punto a distanza ~mezzo utensile dall'asse
            span = max(span, angle * 0.10)
            count = max(2, int(np.ceil(span / self.cartesian_max_step)))
            slerp = Slerp([0.0, 1.0], Rotation.from_matrix(
                np.stack((start[:3, :3], end[:3, :3]))))
            for u in np.linspace(0.0, 1.0, count + 1)[1:]:
                pose = np.eye(4)
                pose[:3, :3] = slerp([u]).as_matrix()[0]
                pose[:3, 3] = (1.0 - u) * start[:3, 3] + u * end[:3, 3]
                path.append((label, pose, self.max_step_cartesian))
        return path

    def _compute_ik(self, pose, seed):
        request = GetPositionIK.Request()
        request.ik_request.group_name = self.move_group
        request.ik_request.ik_link_name = self.ee_frame
        request.ik_request.avoid_collisions = self.ik_avoid_collisions
        request.ik_request.timeout.sec = int(self.ik_timeout)
        request.ik_request.timeout.nanosec = int((self.ik_timeout % 1.0) * 1e9)
        request.ik_request.pose_stamped.header.frame_id = self.base_frame
        request.ik_request.pose_stamped.header.stamp = \
            self.get_clock().now().to_msg()
        translation, quaternion = from_matrix(pose)
        p = request.ik_request.pose_stamped.pose
        p.position.x, p.position.y, p.position.z = [float(v) for v in translation]
        (p.orientation.x, p.orientation.y,
         p.orientation.z, p.orientation.w) = [float(v) for v in quaternion]
        # is_diff: i giunti elencati sovrascrivono lo stato corrente, il resto
        # (dita del gripper incluse) resta quello vero
        request.ik_request.robot_state.is_diff = True
        request.ik_request.robot_state.joint_state.name = self.joint_names
        request.ik_request.robot_state.joint_state.position = \
            [float(v) for v in seed]

        response = self._call(self.ik_client, request)
        if response is None or response.error_code.val != 1:
            return None
        js = response.solution.joint_state
        table = dict(zip(js.name, js.position))
        if not all(name in table for name in self.joint_names):
            return None
        raw = np.array([table[name] for name in self.joint_names])
        chosen = best_variant(raw, seed, self.joint_limit)
        return None if chosen is None else chosen[1]

    def _pose_msg(self, matrix):
        translation, quaternion = from_matrix(matrix)
        pose = Pose()
        (pose.position.x, pose.position.y,
         pose.position.z) = [float(v) for v in translation]
        (pose.orientation.x, pose.orientation.y,
         pose.orientation.z, pose.orientation.w) = [float(v) for v in quaternion]
        return pose

    def _robot_state(self, joints):
        state = RobotState()
        state.joint_state.name = list(self.joint_names)
        state.joint_state.position = [float(v) for v in joints]
        # is_diff: gli altri giunti (le dita) restano quelli veri
        state.is_diff = True
        return state

    def _contact(self, matrix, seed):
        """Chi tocca chi in una posa, per poterlo scrivere nel log.

        Costa una IK piu' una verifica e si paga SOLO quando un candidato e'
        gia' stato scartato: serve a dire *perche'*, non a decidere.
        """
        solution = self._compute_ik(matrix, seed)
        if solution is None:
            return 'IK non risolta'
        request = GetStateValidity.Request()
        request.group_name = self.move_group
        request.robot_state = self._robot_state(solution)
        response = self._call(self.validity_client, request)
        if response is None:
            return 'validita non verificabile'
        if response.valid:
            return 'nessun contatto nella posa finale'
        pairs = {f'{c.contact_body_1}/{c.contact_body_2}'
                 for c in response.contacts}
        return ', '.join(sorted(pairs)) if pairs else 'in collisione'

    def _feasible(self, chain, pregrasp_joints):
        """None se MoveIt sa eseguire i tre tratti cartesiani, altrimenti il
        motivo.

        Perche' non basta la IK. _score verifica il CONDIZIONAMENTO del
        percorso - che nessun giunto scatti fra due campioni vicini - e lo fa
        con /compute_ik, che di collisioni non sa nulla (ik_avoid_collisions
        e' false apposta: chiedere all'IK di evitarle le farebbe restituire un
        ramo diverso, rompendo proprio la continuita' che stiamo misurando).
        Il motion server invece chiede a computeCartesianPath con
        avoid_collisions:true e rifiuta sotto 0.99 di frazione.

        Misurato in cella 1 il 2026-09-05: il portale della telecamera e' una
        cornice di travi APPOGGIATA sul piano (camera_frame a z=0.01), e a
        fine flip la mano e' orizzontale e sporge ~160 mm oltre il dado.
        Meta' delle rotazioni possibili ci finisce dentro:
            place fraction 0.500  ->  wrist_2_link/camera_frame_right
            place fraction 0.500  ->  camera_frame_front/robotiq_hande_link
        Il planner le proponeva lo stesso, l'albero le eseguiva per 25
        secondi, il motion server le rifiutava in 8 ms e il ciclo finiva in
        Recover con il dado lasciato cadere da 10 cm. Due cicli su tre della
        missione del 2026-09-05 sono andati cosi'.

        Si verifica lo STESSO percorso che verra' eseguito, incatenando i
        tratti: lo stato finale di un tratto e' lo stato iniziale del
        successivo, come fa il motion server.
        """
        if not self.cartesian_client.service_is_ready():
            return None            # senza oracolo si prosegue come prima
        joints = np.asarray(pregrasp_joints, dtype=float)
        for label in ('grasp', 'flip', 'place'):
            request = GetCartesianPath.Request()
            request.header.frame_id = self.base_frame
            request.header.stamp = self.get_clock().now().to_msg()
            request.start_state = self._robot_state(joints)
            request.group_name = self.move_group
            request.link_name = self.ee_frame
            request.waypoints = [self._pose_msg(chain[label])]
            request.max_step = self.cartesian_max_step
            request.jump_threshold = 0.0
            request.avoid_collisions = True
            response = self._call(self.cartesian_client, request, timeout=8.0)
            if response is None:
                return None        # move_group non risponde: non si accusa
            if response.fraction < self.cartesian_fraction:
                return (f'{label}: MoveIt percorre solo '
                        f'{response.fraction:.2f} del tratto cartesiano '
                        f'(serve {self.cartesian_fraction:.2f}); '
                        f'{self._contact(chain[label], joints)}')
            points = response.solution.joint_trajectory.points
            if points:
                table = dict(zip(response.solution.joint_trajectory.joint_names,
                                 points[-1].positions))
                if all(name in table for name in self.joint_names):
                    joints = np.array([table[n] for n in self.joint_names])
        return None

    def _score(self, chain, seed):
        """Costo del percorso, oppure il motivo per cui va scartato.

        IK a catena: ogni campione e' seminato sulla soluzione del precedente,
        cosi' il punteggio riflette il percorso che il robot fara' davvero, e
        non una sequenza di soluzioni scelte indipendentemente.
        """
        current = np.asarray(seed, dtype=float)
        total = 0.0
        pregrasp_joints = None
        worst = 0.0     # solo sui tratti CARTESIANI: e' li' che un passo
                        # grande fra campioni vicini segnala una singolarita'.
                        # Includervi il pregrasp, che e' un moto libero da una
                        # posa lontana, renderebbe il numero illeggibile.
        for label, pose, limit in self._path(chain):
            solution = self._compute_ik(pose, current)
            if solution is None:
                return None, f"IK non risolta su {label}"
            step = float(np.max(np.abs(solution - current)))
            if step > limit:
                return None, (f"{label}: un giunto percorrerebbe {step:.2f} rad "
                              f"fra due campioni vicini (limite {limit}). "
                              f"E' il sintomo di una singolarita' sul percorso")
            if abs(solution[4]) < self.wrist_margin:
                return None, (f"{label}: wrist_2 a {solution[4]:.3f} rad, "
                              f"dentro la singolarita' di polso")
            total += step
            if limit == self.max_step_cartesian:
                worst = max(worst, step)
            if label == 'pregrasp':
                # la configurazione con cui la catena e' stata VERIFICATA:
                # serve a poterla confrontare con quella davvero raggiunta
                pregrasp_joints = solution.copy()
            current = solution
        return (total, worst, pregrasp_joints), None

    # -------------------------------------------------------- ciclo di piano

    def _candidates(self, target):
        """Gli assi del corpo che vale la pena portare in alto, in ordine di
        preferenza. Ogni elemento e' (priorita', asse del corpo, spiegazione).

          0  porta su l'obiettivo, e lo sappiamo per averlo visto
          1  porta su l'obiettivo secondo l'ipotesi del dado standard
          2  porta su una faccia MAI vista: il ciclo non raggiunge
             l'obiettivo ma insegna qualcosa, e al giro dopo si conclude
          3  ne' l'uno ne' l'altro, ma bisogna pur muoversi

        La priorita' 2 e' il punto di tutto: quando l'informazione manca, si
        sceglie il movimento che la produce, invece di sorteggiare.
        """
        layout = self.state.layout()
        up = snap_axis(self.state.rotation.T @ UP)
        down = tuple(-v for v in up)
        out = []
        for axis in AXES:
            if axis in (up, down):
                continue                       # un flip di 90 non li tocca
            entry = layout.get(axis)
            if entry is not None and entry[0] == target:
                priority = 0 if entry[1] else 1
                why = ('obiettivo, faccia gia osservata' if entry[1]
                       else 'obiettivo, dedotto dal dado standard')
            elif axis not in self.state.confirmed:
                priority = 2
                why = 'faccia mai vista: il ciclo serve a misurarla'
            else:
                priority = 3
                why = 'nessuna informazione da guadagnare'
            out.append((priority, axis, why))
        out.sort(key=lambda item: item[0])
        return out

    def _plan_once(self):
        with self.lock:
            face, target, seed = self.face, self.target, self.joints
            has_state = self.state.rotation is not None
        if face is None or target is None:
            return None, 'faccia o obiettivo non ancora noti'
        if seed is None:
            return None, 'nessuno stato dei giunti'
        if not has_state:
            return None, 'stima del dado non ancora agganciata'
        if face == target:
            return None, 'obiettivo gia raggiunto'
        dice = self._dice_matrix()
        if dice is None:
            return None, f'nessuna trasformazione verso {self.dice_frame}'
        if not self.state.at_rest(dice):
            return None, 'dado non appoggiato'

        best = None
        rejects = []
        for priority, axis, why in self._candidates(target):
            world = self.state.rotation @ np.array(axis, dtype=float)
            normal = np.cross(world, UP)
            if np.linalg.norm(normal) < 1e-6:
                continue                       # asse verticale, gia' escluso
            normal = normal / np.linalg.norm(normal)
            # Le due varianti equivalenti: dita lungo +n con flip +90, oppure
            # lungo -n con flip -90. Stesso esito sul dado, configurazioni di
            # polso molto diverse. Si sceglie con l'IK, non a priori.
            for sign in (1.0, -1.0):
                chain = self._chain(dice, sign * normal, sign * 90.0,
                                    self.lift_height)
                score, reason = self._score(chain, seed)
                if score is None:
                    rejects.append(
                        f"p{priority} dita={'+n' if sign > 0 else '-n'}: {reason}")
                    continue
                # Ultima parola a MoveIt: qui si scartano le varianti che il
                # motion server rifiuterebbe. Si paga solo per le varianti che
                # hanno gia' superato il filtro di singolarita'.
                reason = self._feasible(chain, score[2])
                if reason is not None:
                    rejects.append(
                        f"p{priority} dita={'+n' if sign > 0 else '-n'}: {reason}")
                    continue
                key = (priority, score[0])
                if best is None or key < best[0]:
                    best = (key, chain, sign, priority, why, score)
            if best is not None and best[3] <= 1:
                break                          # gia' risolto l'obiettivo

        # Le esclusioni del gate spiegano piu' di quelle di singolarita': la
        # singolarita' e' una scelta nostra, la collisione e' un fatto della
        # cella. Si mostrano per prime, che lo spazio nel messaggio e' poco.
        rejects.sort(key=lambda text: 0 if 'MoveIt' in text else 1)

        if best is None:
            return None, ('nessuna variante praticabile. ' + ' | '.join(rejects[:3]))

        _, chain, sign, priority, why, score = best
        frames = {
            'pregrasp_tf': from_matrix(chain['pregrasp']),
            'grasp_tf': from_matrix(chain['grasp']),
            'flip_target_tf': from_matrix(chain['flip']),
            'place_tf': from_matrix(chain['place']),
        }
        # La rotazione che il dado subira': l'utensile ruota attorno al
        # proprio asse X, che nel mondo e' l'asse delle dita, e il dado e'
        # rigido con l'utensile. Il rilascio conserva l'orientamento.
        finger = chain['grasp'][:3, :3] @ np.array([1.0, 0.0, 0.0])
        frames['rotation'] = Rotation.from_rotvec(
            finger * np.deg2rad(sign * 90.0)).as_matrix()
        frames['pregrasp_joints'] = score[2]
        frames['why'] = (
            f"{face} -> {target}: {why}; dita={'+n' if sign > 0 else '-n'}, "
            f"costo {score[0]:.2f} rad, passo cartesiano peggiore "
            f"{score[1]:.2f}; pregrasp_atteso="
            f"[{', '.join(f'{v:.3f}' for v in score[2])}]; "
            f"mappa [{self._layout_text()}]")
        if priority >= 2 and rejects:
            frames['why'] += f"; scartate: {' | '.join(rejects[:2])}"
        return frames, None

    def _plan_loop(self):
        while rclpy.ok():
            time.sleep(self._plan_period)
            with self.lock:
                busy = self.busy_since
            if busy is not None:
                if time.time() - busy < self.busy_timeout:
                    continue
                # tempo di guardia scaduto: il ciclo non ha cambiato la faccia
                # entro il tempo in cui avrebbe dovuto, quindi e' andato
                # storto e serve un piano nuovo, non quello vecchio
                with self.lock:
                    self.busy_since = None
                self.get_logger().warn(
                    'ciclo senza effetto entro il tempo di guardia: ripianifico')
            try:
                frames, reason = self._plan_once()
            except Exception as exc:
                frames, reason = None, f'eccezione nel planner: {exc}'
                self.get_logger().warn(reason)
            with self.lock:
                self.candidate = frames
                self.reject = reason if frames is None else ''

    def _broadcast(self):
        with self.lock:
            plan = self.latched
        if plan is None:
            return
        stamp = self.get_clock().now().to_msg()
        transforms = []
        for name, value in plan.items():
            if name in ('why', 'rotation', 'pregrasp_joints'):
                continue
            translation, quaternion = value
            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = self.base_frame
            transform.child_frame_id = name
            transform.transform.translation.x = float(translation[0])
            transform.transform.translation.y = float(translation[1])
            transform.transform.translation.z = float(translation[2])
            transform.transform.rotation.x = float(quaternion[0])
            transform.transform.rotation.y = float(quaternion[1])
            transform.transform.rotation.z = float(quaternion[2])
            transform.transform.rotation.w = float(quaternion[3])
            transforms.append(transform)
        self.tf_broadcaster.sendTransform(transforms)


def main(args=None):
    rclpy.init(args=args)
    node = DicePlanner()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # il gestore di SIGINT del launch ha gia' chiuso il contesto: senza
        # questa guardia la seconda chiamata solleva RCLError e il launch
        # riporta 'process has died, exit code 1' su una chiusura pulita,
        # cioe' proprio il messaggio che serve per accorgersi dei guasti veri
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
