# Write a function count_pairs(dna, pair) that returns the number of occurrences of a pair of characters (pair) in a DNA string (dna). 
# For example, calling the function with dna as 'ACTGCTATCCATT' and pair as 'AT' will return 2. Filename: count_pairs.py.

def count_pairs(dna, pair):
	counter=0
	n=len(dna)
	m=len(pair)
	for i in range(n):
		if dna[i]==pair[0] :
			flag = True
			for j in range(m):
				if dna[i+j] != pair[j]:
					flag = False
			if flag:
				counter+=1;
				
	return counter
	
print(count_pairs("ACTGCTATCCATT", "CAT"))