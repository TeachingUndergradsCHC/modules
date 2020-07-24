	.global main
	.func main
main:
@@@@---Do not touch this---@@@@
	
	ldr r0, addr_value1				@ get address of value1
	vldr.32 s0, [r0]				@ load value1 into s0 since it is a floating point number
	bl sinf						@ call the sinf function which takes s0 as an argument
	vcvt.f64.f32 d5, s0				@ convert the result of sinf into a double precision number
	vmov r2, r3, d5					@ move the double precision number into r2 and r3 so printf can use it
	push {r2, r3}

@@@@@@---Your code here---@@@@@@
	

	
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@ DO NOT CHANGE CODE PAST THIS
	
_exit:
	mov r7, #1
	swi 0
	
	
addr_value1:	.word value1
addr_sinout:	.word sinout

	.data
sinout:	.asciz "The sine is %f\nThe cosine is %f\nThe tangent is %f\n "
value1:	 	.float 2.54
