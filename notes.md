



---


Task:
	+ Predict 2D chemical structures of small molecules detected in complex biological extracts
	+ 2D chemical structure is a SMILES string.("O=C=O "for CO2)
	+ 



SMILE string:
	tells us the chemical structure of a molecule.
	
|                                |       |                     |
| ------------------------------ | ----- | ------------------- |
| **Propanone** (acetone)        | C₃H₆O | **`unset CC(=O)C`** |
| **Propanal** (propionaldehyde) | C₃H₆O | **`unset CCC=O`**   |
	
![[Pasted image 20261011120632.png]]


- [i] Idea:
	- make python to predict the smiles string of the chemical.
		- it should be rule based.


	+ instrument_type, collision_energy_orig_units can be omitted.
		+ same value through out (check on code)