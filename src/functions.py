from random import randint

def Random_basis_gen(length):
    return [randint(0, 1) for _ in range(length)]  # Must be 0 or 1 only

def Compare_basis(alice_bases, bob_bases, alice_results):
    matched_results = []
    for i in range(len(alice_bases)):
        if alice_bases[i] == bob_bases[i]:
            matched_results.append(alice_results[i])
    return matched_results
