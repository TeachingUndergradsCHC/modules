	@ ARM assembly program to compute factorial
	@ Put the required header information here

	.extern printf
	.extern scanf
	.global main
	.func main
main:		PUSH	{LR}
	LDR	R0, =prompt	@ Get address of prompt
	BL	printf		@ Print prompt to screen
	LDR	R0, =in		@ Get addr of input format string
	LDR	R1, =n		@ Get addr of n
	BL	scanf		@ Put value in n using scanf
	LDR	R0, =n		@ Addr of n into R0
	LDR	R0, [R0]	@ n is now in R0
	@-------------
	@ Your code goes here.  Put n! in R2
	@-------------
	@ Start by putting 1 into R2
	MOV R2, #1

	@ create a label for the beginning of your loop
loop:	
	@ If R0 is 1 or 0 the code should branch to the label finish
        CMP R0, #1
	BEQ finish
	CMP R0, #0
	BEQ finish
	
	@ Otherwise write a loop which multiples R0 by R2 each iteration
	@ Decrementing R0 until R0 reaches 1
	@ (Alternatively one could write a loop with an index register
	@  that counts up from 1 to n.)
	MUL R2, R0, R2
	SUB R0, R0, #1
	BAL loop
	
	@-------------
	@ Do NOT alter code past this line.
	@-------------
finish:	MOV	R1, R2
	LDR	R0, =out
	BL 	printf
	POP	{PC}

	.data
n:		.word 0
prompt:		.asciz "Enter n: "
in:		.asciz "%d"
out:		.asciz "factorial is: %d\n"
