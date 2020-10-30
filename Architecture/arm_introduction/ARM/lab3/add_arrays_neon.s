
add_arrays_neon:
	@ R0 contains the pointer to array x
	@ R1 contains the pointer to array y
	@ R2 contains the pointer to array z
	@ R3 contains the size
	PUSH {LR}
	PUSH {R4,D0-D2}   @ Save content of these registers
	MOV R4, #0        @ Loop index variable
loop_head:
	CMP R4, R3
	BGE loop_exit
loop_body:
	@  Load 8 elements of array x into 64 bit register
	VLD1.64 {D0}, [R0]!
	@  Load 8 elements of array y into 64 bit register
	VLD1.64 {D1}, [R1]!	
	@  Perform an 8x8 SIMD add on these two registers
	VADD.U8 D2, D0, D1
	@  Store the result into array z
	VSTR1.64 {D2}, [R2]!
	
	ADD R4, R4, #8
	BAL loop_head
loop_exit:
	POP {R4,D0-D2}    @ Restore content of these registers
	POP {SP}          @ Return