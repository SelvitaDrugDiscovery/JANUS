import re

import pandas as pd
from rdkit import Chem


def _canon_ring(s: str):
    """
    Transforms cyclical list of atoms to canonical view. Puts the letter with max char value in the
    beginning of the ring string. If two letters identical, compares next to them and so on,
    until differences occur.
    Examples:
        'cncc' -> 'nccc'
        'cnocc' -> 'occcn'
        'cncnnc' -> 'nnccnc'

    :param s: cyclical list of ring strings, ex 'ccnccc' for pyridine.
    :return: canonical view of ring string, ex 'nccccc' for pyridine.
    """

    n = len(s)
    max_idx = 0

    # main loop
    for i in range(1, n):
        j = 0
        # find max letter over whole cyclic s
        while j < n:
            # cyclical index
            idx = (i + j) % n
            m_idx = (max_idx + j) % n

            if s[idx] > s[m_idx]:
                max_idx = i
                break
            # compare next letters if multiple maximums
            elif s[idx] == s[m_idx]:
                j += 1
            else:
                break

    return s[max_idx:] + s[:max_idx]


def canon_ring(s: str):
    """
    Fully canonical ring view, takes into account possible inversion of cycle
    Selects the one with greater value, ex 'ocnccc' > 'occcnc'
    :param s:
    :return: canonicalized view of ring string
    """

    res = _canon_ring(s)
    inv_res = _canon_ring(s[::-1])
    return max(res, inv_res)


def get_symbol(atom):
    if atom.GetIsAromatic():
        return atom.GetSymbol().lower()
    else:
        return atom.GetSymbol()


def get_rings(smi: str):
    mol = Chem.MolFromSmiles(smi)
    ri = mol.GetRingInfo()

    res = []
    for rids in ri.AtomRings():
        ring_atoms = [get_symbol(mol.GetAtomWithIdx(i)) for i in rids]
        ring = ''.join(ring_atoms)
        res.append(canon_ring(ring))
    return res


def count_rings(smi_ser):
    ring_counts = {}
    for smi in smi_ser:
        rings = get_rings(smi)
        for r in rings:
            if r in ring_counts:
                ring_counts[r] += 1
            else:
                ring_counts[r] = 1
    return pd.Series(ring_counts).sort_values(ascending=False)


def filter_rings(smi, chembl_rings, count_threshold=100):
    print('aaaaa')
    rings = get_rings(smi)
    print('rings', rings)
    for r in rings:
        is_arom = re.match(r'^[a-z]*$', r)
        print('is_arom', bool(is_arom))
        print('ring_count', chembl_rings.get(r, 0))
        if is_arom and chembl_rings.get(r, 0) < count_threshold:
            return False
    return True


def filter_rings_ser(smi_ser, chembl_rings, threshold=100):
    return smi_ser.apply(filter_rings, args=(chembl_rings, threshold))
