N = 1000000
dna = 'A'*N
def count_v6(dna, base):
    m = []   # matches for base in dna: m[i]=True if dna[i]==base
    for c in dna:
        m.append(True if c == base else False)
    return sum(m)

print(count_v6('abababababsdb','b'))
