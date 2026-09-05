"""Verifica statica degli alberi di comportamento - DRIMS 2026.

Non sostituisce una prova a runtime, ma cattura la classe di guasti che
questa struttura introduce, e che a runtime si manifestano male: un SubTree
non risolto e' un errore chiaro al caricamento, ma una COSTANTE che non
arriva no. Una porta di blackboard vuota diventa zero, e position=0 sul
gripper significa "chiuso" in simulazione: l'albero girerebbe chiudendo la
pinza dove doveva aprirla.

Controlla:
  1. XML bene formato;
  2. per ogni punto d'ingresso, che ogni SubTree riferito sia definito nel
     grafo degli include, e che nessun ID sia definito due volte;
  3. che ogni chiamata a SubTree porti _autoremap="true": senza, il subtree
     riceve una blackboard NUOVA e le costanti non arrivano;
  4. che ogni chiave letta come {chiave} sia prodotta da qualcuno - uno
     Script del grafo, o una porta di uscita di una foglia;
  5. che i due ingressi di ciascuna coppia in PAIRS siano identici a meno del
     blocco Script. PAIRS e' vuota oggi (un solo albero serve entrambi i
     bersagli); il controllo resta per il giorno in cui servisse di nuovo.

Uso:  python3 test_trees.py <cartella trees>
"""
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# attributi che SCRIVONO sulla blackboard invece di leggerla
OUTPUT_ATTRS = {'result_code', 'face_number', 'pose', 'success', 'message'}
# Coppie di ingressi che devono restare identiche a meno dello Script.
#
# Vuota, e per una buona ragione: c'era stato un momento in cui si credeva che
# la polarita' di position del gripper fosse invertita fra simulazione e robot
# vero, e servivano due file d'ingresso. Non e' cosi': con i valori degli
# organizzatori (0.045 aperto, 0.0 chiuso, effort al default) lo stesso albero
# vale per entrambi, e i due ingressi sono stati ricondotti a uno.
# Se in futuro qualcosa dovesse divergere davvero, si aggiunge qui la coppia e
# il controllo torna a garantire che divergano SOLO nelle costanti.
PAIRS = []

errors, notes = [], []


def parse(path):
    try:
        return ET.parse(path).getroot()
    except ET.ParseError as exc:
        errors.append(f"{path.name}: XML malformato: {exc}")
        return None


def include_graph(path, seen=None):
    """I file raggiunti dagli include, in ordine, senza ricicli."""
    seen = seen if seen is not None else []
    if path.name in [p.name for p in seen]:
        return seen
    seen.append(path)
    root = parse(path)
    if root is None:
        return seen
    for inc in root.findall('include'):
        child = path.parent / inc.get('path')
        if not child.exists():
            errors.append(f"{path.name}: include inesistente '{inc.get('path')}'")
            continue
        include_graph(child, seen)
    return seen


def check_entry(path):
    root = parse(path)
    if root is None or root.get('main_tree_to_execute') is None:
        return
    main = root.get('main_tree_to_execute')
    graph = include_graph(path)

    defined, origin = set(), {}
    for f in graph:
        r = parse(f)
        for bt in (r.findall('BehaviorTree') if r is not None else []):
            bid = bt.get('ID')
            if bid in defined:
                errors.append(f"{path.name}: ID '{bid}' definito due volte "
                              f"({origin[bid]} e {f.name})")
            defined.add(bid)
            origin[bid] = f.name

    if main not in defined:
        errors.append(f"{path.name}: main_tree_to_execute='{main}' non definito")

    refs, provided, read = set(), set(), {}
    for f in graph:
        r = parse(f)
        if r is None:
            continue
        for node in r.iter():
            if node.tag == 'SubTree':
                refs.add(node.get('ID'))
                if node.get('_autoremap') != 'true':
                    errors.append(f"{f.name}: SubTree '{node.get('ID')}' senza "
                                  f"_autoremap=\"true\": le costanti non arriverebbero")
            if node.tag == 'Script':
                for key in re.findall(r'(\w+)\s*:=', node.get('code') or ''):
                    provided.add(key)
            for attr, value in node.attrib.items():
                for key in re.findall(r'\{([^}]+)\}', value):
                    (provided if attr in OUTPUT_ATTRS else read).__setitem__(key, f.name) \
                        if attr not in OUTPUT_ATTRS else provided.add(key)

    for bad in sorted(refs - defined):
        errors.append(f"{path.name}: SubTree '{bad}' riferito ma non definito")
    for key, where in sorted(read.items()):
        if key not in provided:
            errors.append(f"{path.name}: chiave '{{{key}}}' letta in {where} "
                          f"ma nessuno la produce")

    notes.append(f"  {path.name:<22} main={main:<14} "
                 f"include={'+'.join(f.name for f in graph[1:]) or '-':<38} "
                 f"subtree ok={len(refs)}  chiavi lette={len(read)}")


def normalise(text):
    """Toglie commenti, Script e spaziatura: resta la STRUTTURA."""
    text = re.sub(r'<!--.*?-->', '', text, flags=re.S)
    text = re.sub(r'<Script\b.*?/>', '<Script/>', text, flags=re.S)
    return re.sub(r'\s+', ' ', text).strip()


def check_pair(a, b):
    if not (a.exists() and b.exists()):
        return
    if normalise(a.read_text()) == normalise(b.read_text()):
        notes.append(f"  {a.name} == {b.name} a meno dello Script  OK")
    else:
        errors.append(f"{a.name} e {b.name} divergono OLTRE lo Script: "
                      f"la logica va tenuta nel corpo condiviso")

    def values(p):
        m = re.search(r'<Script code="(.*?)"\s*/>', p.read_text(), flags=re.S)
        return dict(re.findall(r'(\w+)\s*:=\s*([^;"]+)', m.group(1))) if m else {}

    va, vb = values(a), values(b)
    if set(va) != set(vb):
        errors.append(f"{a.name} e {b.name} definiscono costanti diverse: "
                      f"{set(va) ^ set(vb)}")
    diff = {k: (va[k].strip(), vb[k].strip()) for k in va if k in vb
            and va[k].strip() != vb[k].strip()}
    notes.append(f"    costanti che differiscono: "
                 f"{', '.join(f'{k} {x}->{y}' for k, (x, y) in sorted(diff.items()))}")


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
    print(f"=== verifica statica in {d}\n")
    for f in sorted(d.glob('*.xml')):
        root = parse(f)
        if root is not None and root.get('main_tree_to_execute'):
            check_entry(f)
    print("PUNTI D'INGRESSO")
    print('\n'.join(notes[:99]))
    notes.clear()
    print("\nCOPPIE SIMULAZIONE / ROBOT VERO")
    for a, b in PAIRS:
        check_pair(d / a, d / b)
    print('\n'.join(notes))
    print()
    if errors:
        print(f"FALLITO: {len(errors)} problemi")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("PASSATO: nessun problema")
    return 0


sys.exit(main())
