"""Pauli compiler based on arXiv:2408.03294.
``compiler`` takes a generator set and a target Pauli string and outputs a
:math:`\\mathcal{O}(N)` length sequence of Pauli strings that generates the target
Pauli string via nested commutators.
"""
import itertools
from collections import deque
from collections.abc import Generator
from dataclasses import dataclass
from itertools import permutations
from typing import Iterable

from paulie.common.pauli_string_bitarray import PauliString
from paulie.common.pauli_string_collection import PauliStringCollection
from paulie.common.pauli_string_factory import get_identity, get_pauli_string, get_single


def _evaluate_sequence(sequence: list[PauliString]) -> PauliString | None:
    """Evaluate a sequence stored as ``[base, A1, ..., Am]``."""
    return PauliStringCollection(sequence).evaluate_commutator_sequence()


def _evaluate_paulie_orientation(sequence: list[PauliString]) -> PauliString | None:
    """Evaluate a sequence stored as ``[Am, ..., A1, base]``."""
    if not sequence:
        return None
    return PauliStringCollection(sequence[:-1]).nested_adjoint(sequence[-1])


def _sequence_to_paulie_orientation(sequence: list[PauliString]) -> list[PauliString]:
    """Convert ``[base, A1, ..., Am]`` to PauLie's ``nested_adjoint`` orientation."""
    if not sequence:
        return []
    return list(reversed(sequence[1:])) + [sequence[0]]


def left_a_minimal(k: int) -> list[PauliString]:
    r"""Return the minimal left universal set ``{X_i, Z_i}_i \cup {Z^{\otimes k}}``."""
    a_ops: list[PauliString] = []
    for index in range(k):
        a_ops.append(get_single(k, index, "X"))
        a_ops.append(get_single(k, index, "Z"))
    a_ops.append(get_pauli_string("Z" * k))
    return a_ops


def choose_u_for_b(k: int) -> PauliString:
    """Choose the fixed left tag used when coupling to right-side generators."""
    return get_single(k, 0, "X")


def _all_left_paulis(k: int) -> list[PauliString]:
    """Enumerate all non-identity Pauli strings on ``k`` qubits."""
    return [p for p in get_identity(k).gen_all_pauli_strings() if not p.is_identity()]


@dataclass
class SubsystemCompilerConfig:
    """Configuration of the subsystem compiler."""

    k_left: int
    n_total: int


class SubsystemCompiler:
    """Compiler for the right subsystem contribution."""

    def __init__(self, cfg: SubsystemCompilerConfig):
        if cfg.k_left < 2:
            raise ValueError("k_left must be >= 2 for the Pauli Compiler algorithm")
        self.k = cfg.k_left
        self.n_total = cfg.n_total
        self.n_right = self.n_total - self.k
        self.u_tag = choose_u_for_b(self.k)
        self.left_pool = _all_left_paulis(self.k)

    def extend_left(self, a_left: PauliString) -> PauliString:
        """Extend a left Pauli string by identities on the right."""
        return a_left + get_identity(self.n_right)

    def extend_pair(self, u_left: PauliString, b_right: PauliString) -> PauliString:
        """Combine left and right parts into a full-length Pauli string."""
        return u_left + b_right

    def factor_w_orders(self, w_right: PauliString) -> list[list[tuple[PauliString, PauliString]]]:
        """Factor ``w_right`` into ordered right-local pieces.

        Each element of the output is a list ``[(U_i, B_i)]`` describing one valid
        order of factors. Only the Y case has two possible local orderings.
        """
        assert len(w_right) == self.n_right
        per_site_options: list[list[list[PauliString]]] = []

        for site, label in enumerate(str(w_right)):
            if label == "Y":
                per_site_options.append(
                    [
                        [get_single(self.n_right, site, "X"), get_single(self.n_right, site, "Z")],
                        [get_single(self.n_right, site, "Z"), get_single(self.n_right, site, "X")],
                    ]
                )
            elif label == "X":
                per_site_options.append([[get_single(self.n_right, site, "X")]])
            elif label == "Z":
                per_site_options.append([[get_single(self.n_right, site, "Z")]])
            else:
                per_site_options.append([[]])

        sequences: list[list[PauliString]] = []

        def rec(index: int, acc: list[PauliString]) -> None:
            if index == len(per_site_options):
                sequences.append(list(acc))
                return
            for segment in per_site_options[index]:
                acc.extend(segment)
                rec(index + 1, acc)
                if segment:
                    del acc[-len(segment) :]

        rec(0, [])
        return [[(self.u_tag, b) for b in flat] for flat in sequences]

    def _choose_a1_a2(self, u_op: PauliString) -> tuple[PauliString, PauliString]:
        """Choose helpers ``A_1, A_2`` with the required commutation pattern."""
        anti_with_u = u_op.get_anti_commutants(self.left_pool)
        for a1 in anti_with_u:
            for a2 in anti_with_u:
                if a2 == a1:
                    continue
                if a1 | a2:
                    return a1, a2
        raise RuntimeError("Failed to find A1,A2 in iP_k^*.")

    def _choose_aprime(self, u_i: PauliString, p_left: PauliString) -> PauliString:
        """Choose a helper ``A'`` that anticommutes with ``u_i`` and commutes with ``p_left``."""
        for helper in u_i.get_anti_commutants(self.left_pool):
            if helper | p_left:
                return helper
        raise RuntimeError("Failed to find A' in iP_k^*.")

    def _rest_full_after(
        self,
        ui_bi: list[tuple[PauliString, PauliString]],
        index: int,
        helpers: list[PauliString],
    ) -> tuple[PauliString, PauliString]:
        """Return the accumulated left and right factors after ``index``."""
        p_left = get_identity(self.k)
        for j in range(index + 1, len(ui_bi)): #Product in Algorithm 3, line 6
            p_left = p_left @ ui_bi[j][0] #Product of ui for index j>i
        for helper in helpers:
            p_left = p_left @ helper #Product of a in h

        p_right = get_identity(self.n_right)
        for j in range(index + 1, len(ui_bi)):
            p_right = p_right @ ui_bi[j][1] #Product of bi for index j>i
        return p_left, p_right

    def _product_uj_a(
        self,
        index: int,
        ui_bi: list[tuple[PauliString, PauliString]],
        helpers: list[PauliString],
    ) -> PauliString:
        """
        Return product of Uj right multiply with product of A for index j to r-1.
        Args:
            index (int): Index of Uj.
            ui_bi (list[tuple(PauliString, PauliString)]): List of (Uj, Bj) pairs.
            helpers (list[PauliString]): List of A
        Returns:
            prod_uj_a (PauliString): Product of Uj right multiply with product of A
        """
        prod_uj = get_identity(self.k)
        for j in range(index, len(ui_bi)): #Product in Algorithm 3, line 6
            prod_uj = prod_uj @ ui_bi[j][0] #Product of Uj

        prod_a = get_identity(self.k)
        for a in helpers:
            prod_a = prod_a @ a #Product of A

        prod_uj_a = prod_uj @ prod_a #(Product of Uj)(Product of A)

        return prod_uj_a

    def _product_uj_bj(
        self,
        index: int,
        ui_bi: list[tuple[PauliString, PauliString]],
    ) -> PauliString:
        """
        Return product of Uj ⊗ Bj for index j to r-1.
        Args:
            index (int): Index of Uj.
            ui_bi (list[tuple(PauliString, PauliString)]): List of (Uj, Bj) pairs.
        Returns:
            prod_uj_bj (PauliString): Product of Uj ⊗ Bj
        """
        prod_uj_bj = get_identity(self.n_total)
        for j in range(index, len(ui_bi)): #Product in Algorithm 3, line 6
            prod_uj_bj = prod_uj_bj @ ui_bi[j][0].tensor(ui_bi[j][1]) #Product of uj ⊗ bj

        return prod_uj_bj

    def _product_a_i(
        self,
        helpers: list[PauliString],
    ) -> PauliString:
        """
        Return product of A ⊗ I in helpers.
        Args:
            helpers (list[PauliString]): List of A
        Returns:
            prod_a_i (PauliString): Product of A ⊗ I
        """
        prod_a_i = get_identity(self.n_total)
        for a in helpers:
            prod_a_i = prod_a_i @ self.extend_left(a) #Product of A

        return prod_a_i

    def subsystem_compiler(self, w_right: PauliString) -> list[PauliString]:
        """Compile a target supported only on the right subsystem."""
        assert len(w_right) == self.n_right

        for ui_bi in self.factor_w_orders(w_right): #Algorithm 3, line 1: Choose [u1 ⊗ b1,...,ur ⊗ br] s.t. b1⋅⋅⋅br=w_right
            if not ui_bi:
                return []

            index = len(ui_bi) - 1 #Algorithm 3, line 2
            sequence: list[PauliString] = [self.extend_pair(ui_bi[-1][0], ui_bi[-1][1])] #Algorithm 3, line 3: Last pair ur ⊗ br
            helpers: list[PauliString] = [] #Algorithm 3, line 4
            helper_uses: dict[int, int] = {}

            while index >= 1: #Algorithm 3, line 5: Loop from index r-1 to 1
                u_i, b_i = ui_bi[index]
                p_left, p_right = self._rest_full_after(ui_bi, index, helpers)

                if (p_left @ u_i).is_identity(): #Algorithm 3, line 6
                    count = helper_uses.get(index, 0)
                    if count >= 1:
                        sequence.append(self.extend_pair(u_i, b_i))
                        index -= 1
                        continue

                    a1, a2 = self._choose_a1_a2(u_i)
                    helpers = [a1, a2]
                    helper_uses[index] = count + 1
                    sequence.append(self.extend_left(a1))
                    sequence.append(self.extend_left(a2))
                    continue

                current = self.extend_pair(u_i, b_i)
                rest_full = p_left + p_right
                if current | rest_full:
                    count = helper_uses.get(index, 0)
                    if count >= 1:
                        sequence.append(current)
                        index -= 1
                        continue

                    a_prime = self._choose_aprime(u_i, p_left)
                    helpers = [a_prime]
                    helper_uses[index] = count + 1
                    sequence.append(self.extend_left(a_prime))
                    continue

                sequence.append(current)
                index -= 1

            return sequence

        return []


def left_map_over_a(
    v_from: PauliString,
    v_to: PauliString,
    generators: list[PauliString],
) -> list[PauliString]:
    """
    Find a left-only adjoint path from ``v_from`` to ``v_to`` using BFS.
    Args:
        v_from (PauliString): Starting Pauli string.
        v_to (PauliString): Target Pauli string.
        generators (list[PauliString]): Universal set.
    Returns:
        sequence (list[PauliString]): List of [A1,...,As-1] mapping from v_from to v_to.
    """
    if v_from == v_to:
        return []

    queue: deque[PauliString] = deque([v_from])
    parent: dict[PauliString, tuple[PauliString, PauliString]] = {}
    seen: set[PauliString] = {v_from}

    while queue:
        current = queue.popleft()
        if current == v_to:
            sequence: list[PauliString] = []
            cursor = current
            while cursor != v_from:
                previous, used = parent[cursor]
                sequence.append(used)
                cursor = previous
            return sequence

        for helper in current.get_anti_commutants(generators):
            nxt = helper @ current
            if nxt in seen:
                continue
            seen.add(nxt)
            parent[nxt] = (current, helper)
            queue.append(nxt)

    raise RuntimeError("Left map BFS failed.")


@dataclass
class PauliCompilerConfig:
    """Configuration of the optimal compiler."""

    k_left: int
    n_total: int
    fallback_depth: int = 8
    fallback_nodes: int = 200000


class OptimalPauliCompiler:
    """Compiler implementing the construction from arXiv:2408.03294.

    Compiles a target Pauli string into an O(N) length sequence of generators
    that produces the target via nested commutators.

    Args:
        cfg: Compiler configuration specifying the left-right partition
            and fallback search limits.
    """

    def __init__(self, cfg: PauliCompilerConfig):
        if cfg.k_left < 2:
            raise ValueError("k_left must be >= 2 for the Pauli Compiler algorithm")
        self.k = cfg.k_left
        self.n_total = cfg.n_total
        self.n_right = self.n_total - self.k
        self.a_left = left_a_minimal(self.k)
        self.sub = SubsystemCompiler(SubsystemCompilerConfig(k_left=self.k, n_total=self.n_total))
        self.fallback_depth = cfg.fallback_depth
        self.fallback_nodes = cfg.fallback_nodes

    def extend_left(self, a_left: PauliString) -> PauliString:
        """Extend a left Pauli string to the full system."""
        return a_left + get_identity(self.n_right)

    def _left_factor_from_sequence(self, ops: list[PauliString]) -> PauliString:
        """Extract the left factor of a compiled right-subsystem sequence."""
        result = _evaluate_sequence(ops)
        if result is None:
            return get_single(self.k, 0, "X")
        return result.get_substring(0, self.k)

    def _candidate_decompositions(
            self, w_right: PauliString
    ) -> list[tuple[PauliString, PauliString]]:
        """Return candidate decompositions ``W = W1 @ W2``
        with ``W1`` anti-commuting with ``W2``."""
        candidates: list[tuple[PauliString, PauliString]] = []
        seen: set[tuple[PauliString, PauliString]] = set()

        for w1 in w_right.get_anti_commutants():
            w2 = w1 @ w_right

            w1, w2 = sorted((w1, w2))
            pair = (w1, w2)

            if pair not in seen:
                seen.add(pair)
                candidates.append(pair)

        return candidates

    def _all_interleavings_preserving(
        self,
        a_block: list[PauliString],
        b_block: list[PauliString],
        c_block: list[PauliString],
        cap: int = 60000,
    ) -> Iterable[list[PauliString]]:
        """Yield capped interleavings preserving the order inside each block."""
        count = 0
        len_a, len_b, len_c = len(a_block), len(b_block), len(c_block)

        def rec(
            i: int, j: int, k: int, prefix: list[PauliString]
        ) -> Generator[list[PauliString], None, None]:
            nonlocal count
            if count >= cap:
                return
            if i == len_a and j == len_b and k == len_c:
                count += 1
                yield list(prefix)
                return
            if i < len_a:
                prefix.append(a_block[i])
                yield from rec(i + 1, j, k, prefix)
                prefix.pop()
                if count >= cap:
                    return
            if j < len_b:
                prefix.append(b_block[j])
                yield from rec(i, j + 1, k, prefix)
                prefix.pop()
                if count >= cap:
                    return
            if k < len_c:
                prefix.append(c_block[k])
                yield from rec(i, j, k + 1, prefix)
                prefix.pop()

        return rec(0, 0, 0, [])

    def _all_interleavings_preserving4(
        self,
        a_block: list[PauliString],
        b_block: list[PauliString],
        c_block: list[PauliString],
        d_block: list[PauliString],
        cap: int = 120_000,
    ) -> Iterable[list[PauliString]]:
        """Yield capped interleavings preserving the order inside four blocks."""
        count = 0
        len_a, len_b, len_c, len_d = len(a_block), len(b_block), len(c_block), len(d_block)

        def rec(
            i: int, j: int, k: int, l_idx: int, prefix: list[PauliString]
        ) -> Generator[list[PauliString], None, None]:
            nonlocal count
            if count >= cap:
                return
            if i == len_a and j == len_b and k == len_c and l_idx == len_d:
                count += 1
                yield list(prefix)
                return
            if i < len_a:
                prefix.append(a_block[i])
                yield from rec(i + 1, j, k, l_idx, prefix)
                prefix.pop()
                if count >= cap:
                    return
            if j < len_b:
                prefix.append(b_block[j])
                yield from rec(i, j + 1, k, l_idx, prefix)
                prefix.pop()
                if count >= cap:
                    return
            if k < len_c:
                prefix.append(c_block[k])
                yield from rec(i, j, k + 1, l_idx, prefix)
                prefix.pop()
                if count >= cap:
                    return
            if l_idx < len_d:
                prefix.append(d_block[l_idx])
                yield from rec(i, j, k, l_idx + 1, prefix)
                prefix.pop()

        return rec(0, 0, 0, 0, [])

    def _permutation_adj(
        self,
        g: list[PauliString]
    ) -> list[PauliString] | None:
        """
        Search for permutation of g that adj map is non-zero.
        Args:
            g (list[PauliString]): List of Pauli string before permutation.
        Returns:
            sequence (list[PauliString]): List of permuted Pauli string that adj map is non-zero.
        """

        for perm in itertools.permutations(g):

            if _evaluate_sequence(perm) is not None:
                return perm

    def _bfs_case3(
        self, w_right: PauliString, depth_cap: int, node_cap: int
    ) -> list[PauliString] | None:
        """Fallback bounded BFS for the case ``V = I`` and ``W != I``."""
        universal_set = construct_universal_set(self.n_total, self.k)
        target_left = get_identity(self.k)
        nodes = 0
        frontier: list[tuple[PauliString | None, list[int]]] = [(None, [])]
        visited: set[tuple[int, PauliString]] = set()

        for depth in range(1, depth_cap + 1):
            new_frontier: list[tuple[PauliString | None, list[int]]] = []
            for result, seq_idx in frontier:
                for op_index, operator in enumerate(universal_set):
                    nodes += 1
                    if nodes > node_cap:
                        return None
                    new_result = operator if result is None else (operator ^ result)
                    if new_result is None:
                        continue
                    state_key = (depth, new_result)
                    if state_key in visited:
                        continue
                    visited.add(state_key)
                    new_sequence = seq_idx + [op_index]
                    if depth >= 2:
                        if (
                            new_result.get_substring(0, self.k) == target_left
                            and new_result.get_substring(self.k, self.n_right) == w_right
                        ):
                            return [universal_set[idx] for idx in new_sequence]
                    new_frontier.append((new_result, new_sequence))
            frontier = new_frontier
        return None

    def compile(self, v_left: PauliString, w_right: PauliString) -> list[PauliString]:
        """Compile a target specified by its left and right factors.

        Args:
            v_left: Left factor of the target (length ``k_left``).
            w_right: Right factor of the target (length ``n_total - k_left``).

        Returns:
            Sequence in PauLie's ``nested_adjoint`` orientation: ``[Am, ..., A1, base]``.

        Raises:
            ValueError: If the lengths of ``v_left`` and ``w_right``
                do not match the configured partition.
            RuntimeError: If no valid sequence is found.
        """

        if len(v_left) != self.k or len(w_right) != self.n_right:
            raise ValueError(
                f"Expected v_left of length {self.k} and w_right of length {self.n_right}, "
                f"got {len(v_left)} and {len(w_right)}."
            )

        if w_right.is_identity(): #Algorithm 2, line 2
            for a_s in self.a_left: #Algorithm 2, line 3: loop through As in default universal set TODO: change to user define universal set
                try: #Algorithm 2, line 3: Choose [A1,...,As]
                    seq_a = left_map_over_a(a_s, v_left, self.a_left) #Find [A1,...,As-1]
                    seq_a.append(a_s)
                except RuntimeError:
                    continue
                sequence: list[PauliString] | None = [self.extend_left(a) for a in seq_a] #Algorithm 2, line 4: Extend to full system [A1 ⊗ I,...,As ⊗ I]
                return sequence #Return the first found map
                ### No need to check
                #result = None
                #if sequence is not None:
                #    result = _evaluate_sequence(sequence) # Calculate adjoint map of [A1 ⊗ I,...,As ⊗ I]
                #    if (
                #        result.get_substring(0, self.k) == v_left #Compare left system of result to v_left
                #        and result.get_substring(self.k, self.n_right) == w_right #Compare right system of result to w_right
                #    ):
                #        return _sequence_to_paulie_orientation(sequence) #Return the first found map
            raise RuntimeError("Left-only mapping failed.")

        elif v_left.is_identity(): #Algorithm 2, line 5
            for w1, w2 in self._candidate_decompositions(w_right): #Algorithm 2, line 6: Choose w1,w2 s.t. iw_right=[w1,w2]
                g1 = self.sub.subsystem_compiler(w1) #Algorithm 2, line 7.1
                v1_prime = self._left_factor_from_sequence(g1) #Algorithm 2, line 7.2
                g2 = self.sub.subsystem_compiler(w2) #Algorithm 2, line 8.1
                v2_prime = self._left_factor_from_sequence(g2) #Algorithm 2, line 8.2
                seq_a = left_map_over_a(v2_prime, v1_prime, self.a_left) #Algorithm 2, line 9: Choose [A1,...,As]
                ext_a = [self.extend_left(a) for a in seq_a] #Algorithm 2, line 10: Extend to full system [A1 ⊗ I,...,As ⊗ I]
                g = [*g1, *g2, *ext_a]  #Algorithm 2, line 10: Concatenation of sequence
                sequence = self._permutation_adj(g) #Algorithm 2, line 11: Choose permutation of g s.t. sequence != 0
                if sequence is not None:
                    return sequence #Algorithm 2, line 12: Return permuted sequence that has non-zero adj map

            seq_fb = self._bfs_case3(w_right, self.fallback_depth, self.fallback_nodes)
            if seq_fb is not None:
                return _sequence_to_paulie_orientation(seq_fb)

            w_str = str(w_right)
            site = next(index for index, label in enumerate(w_str) if label != "I")
            label = "X" if w_str[site] == "Z" else ("Z" if w_str[site] == "X" else "X")
            w1 = get_single(self.n_right, site, label)
            w2 = w1 @ w_right
            g1 = self.sub.subsystem_compiler(w1)
            g2 = self.sub.subsystem_compiler(w2)
            v1_prime = self._left_factor_from_sequence(g1)
            v2_prime = self._left_factor_from_sequence(g2)
            seq_a = left_map_over_a(v2_prime, v1_prime, self.a_left)
            ext_a = [self.extend_left(a) for a in seq_a]
            return _sequence_to_paulie_orientation(
                list(reversed(g1)) + ext_a + list(reversed(g2))
            )

        else:
            g_right = self.sub.subsystem_compiler(w_right)
            v_prime = self._left_factor_from_sequence(g_right)
            seq = left_map_over_a(v_prime, v_left, self.a_left)
            candidates = [
                list(g_right) + [self.extend_left(a) for a in seq],
                [self.extend_left(a) for a in seq] + list(g_right),
                list(reversed(g_right)) + [self.extend_left(a) for a in seq],
            ]
            for sequence in candidates:
                result = _evaluate_sequence(sequence)
                if (
                    result is not None
                    and result.get_substring(0, self.k) == v_left
                    and result.get_substring(self.k, self.n_right) == w_right
                ):
                    return _sequence_to_paulie_orientation(sequence)
            return _sequence_to_paulie_orientation(
                list(g_right) + [self.extend_left(a) for a in seq]
            )

def construct_universal_set(n_total: int, k: int) -> list[PauliString]:
    """Construct the universal generator set used by the compiler.

    The set consists of left-local generators extended with identities
    and right-local generators tagged with a fixed left Pauli string.

    Args:
        n_total: Total number of qubits.
        k: Number of qubits in the left partition.

    Returns:
        List of ``2 * n_total + 1`` Pauli strings forming the universal set.

    Raises:
        ValueError: If ``k`` is out of range.
    """
    if not 1 <= k < n_total:
        raise ValueError("Require 1 <= k < N")

    a_k = left_a_minimal(k)
    n_right = n_total - k
    u_tag = choose_u_for_b(k)
    right_b = [get_single(n_right, index, "X") for index in range(n_right)] + [
        get_single(n_right, index, "Z") for index in range(n_right)
    ]
    a_prime = [a + get_identity(n_right) for a in a_k]
    b_prime = [u_tag + b for b in right_b]
    return a_prime + b_prime


def compile_target(target: PauliString, k_left: int) -> list[PauliString]:
    """Compile a full target Pauli string into a generator sequence.

    Args:
        target: The target Pauli string to compile.
        k_left: Number of qubits in the left partition (must be >= 2).

    Returns:
        Sequence in PauLie's ``nested_adjoint`` orientation such that
        ``PauliStringCollection(seq[:-1]).nested_adjoint(seq[-1]) == target``.

    Raises:
        ValueError: If ``k_left`` is out of range.
    """
    n_total = len(target)
    if not 2 <= k_left < n_total:
        raise ValueError("Require 2 <= k_left < len(target)")

    v_left = target.get_substring(0, k_left)
    w_right = target.get_substring(k_left, n_total - k_left)
    compiler = OptimalPauliCompiler(PauliCompilerConfig(k_left=k_left, n_total=n_total))
    return compiler.compile(v_left, w_right)
