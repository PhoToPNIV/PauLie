"""
Test pauli compiler
"""
from itertools import product
import pytest
from paulie import PauliString, PauliStringCollection
from paulie import compile_target

gates = ["I", "X", "Y", "Z"]
pauli_string_length = 5
test_cases = [
    "".join(p)
    for n in range(1, pauli_string_length + 1)
    for p in product(gates, repeat=n)
]

@pytest.mark.parametrize("input_arg", test_cases)
def test_compile_target(input_arg) -> None:
    """
    Test pauli compiler for all possible combination
    """
    target = PauliString(pauli_str=input_arg)

    for k in range(1, len(target)):

        if k < 2 or k == len(target):
            with pytest.raises(ValueError):
                compile_target(target, k_left=k)
        else:
            sequence = compile_target(target, k_left=k)
            result = PauliStringCollection(sequence).evaluate_commutator_sequence()
            assert target == result
