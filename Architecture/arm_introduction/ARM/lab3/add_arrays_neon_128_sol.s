        .global add_arrays_neon
	.text
	
add_arrays_neon:
	@ R0 contains the pointer to array x
	@ R1 contains the pointer to array y
	@ R2 contains the pointer to array z
	@ R3 contains the size
	PUSH {LR}
	PUSH {R4}   @ Save content of these registers
	MOV R4, #0        @ Loop index variable
loop_head:
	CMP R4, R3
	BGE loop_exit
loop_body:
	@  Load 8 elements of array x into 64 bit register
	VLD1.8 {d0,d1}, [R0]!
	@  Load 8 elements of array y into 64 bit register
	VLD1.8 {d2,d3}, [R1]!	
	@  Perform an 8x8 SIMD add on these two registers
	VADD.U8 q2, q0, q1
	@  Store the result into array z
	VST1.8 {d4,d5}, [R2]!
	
	ADD R4, R4, #16
	BAL loop_head
loop_exit:
	POP {R4}    @ Restore content of these registers
	POP {PC}          @ Return
