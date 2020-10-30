
add_arrays_linear:
	@ R0 contains the pointer to array x
	@ R1 contains the pointer to array y
	@ R2 contains the pointer to array z
	@ R3 contains the size
	PUSH {LR}
	PUSH {R4-R6}      @ Save content of these registers
	MOV R4, #0        @ Loop index variable
loop_head:
	CMP R4, R3
	BGE loop_exit
	LDR R5, [R0], #1  @ Load one element of x array
	LDR R6, [R1], #1  @ Load one element of y array
	ADD R5, R5, R6
	STR R5, [R2], #1  @ Store sum into z array
	ADD R4, R4, #1
	BAL loop_head
loop_exit:
	POP {R4-R6}       @ Restore content of these registers
	POP {SP}          @ Return