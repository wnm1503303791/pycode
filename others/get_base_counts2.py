# def get_base_counts(dna):
	# counts = {'A': 0, 'T': 0, 'G': 0, 'C': 0, 'others': 0}
	# for base in dna:
		# if base=='A' or base=='T' or base=='G' or base=='C':
			# counts[base] += 1
		# else:
			# counts['others'] += 1
		# return counts
def get_base_counts(dna):
	counts = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
	for base in dna:
		if base=='A' or base=='T' or base=='G' or base=='C':
			counts[base] += 1
		else:
			continue
		return counts
	
print(get_base_counts("ACGGAGATTTCGGTATGCAT"))
print(get_base_counts("ACGGAGGGTATGCAT"))
print(get_base_counts("ADLSTTLLD"))