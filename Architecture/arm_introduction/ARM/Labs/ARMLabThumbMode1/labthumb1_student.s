@---------------------------------------------
@	ToUCH / moduleC1 / Thumb lab 1
@	Author: Gabriel Hernandez
@	Date: May.9.2019
@	Description: Students will create two summation functions 
@	based on loops using standard function requirements in 
@	Thumb and Arm mode.
@	
@---------------------------------------------
	.extern printf
	.extern scanf
	.global main
	.func main
	.arm

main:
@------------------------------------------------------------
	PUSH	{LR}
	LDR	R0, =prompt	@get address of prompt
	BL	printf		@prints Prompt
	LDR	R0, =in		@Gets input string of prompt
	LDR	R1, =xvalue	@Gets entered value of x
	BL 	scanf		@Put value in x using scanf
	LDR	R0, =xvalue	@loads value of x into R0
	LDR	R0, [R0]	@x is now in R0


@---------------------------------------------------------------------
@create code to call a summation function
@Calculate 2 times the summation store it into R4

@Your code goes here


@--------------------------------------------------------------------

	MOV 	R1, R0      @stores summation in R1 to return summation
	LDR	R0, =out    @loads string out into R0
	BL	printf      @prints string out with R1 summation
	MOV	R1, R4      @stores R4 into R1 to return 2*summation
	LDR 	R0, =out2   @Loads string out2 into R0
	BL  	printf      @prints string out2 with 2*summation
	POP 	{PC}        @ends program

@-----------------------------------------------------------------------------------------------------
@create an ARM mode summation fuction using standard function requirements
@function should be based on loop, return result in R0

@Your code goes here

@-----------------------------------------------------------------------------------------------------
@create a Thumb mode summation function using standard function requirements
@function should be based on loop, return result in R0

@Your code goes here



@----------------------------------------------------------------------
.data
xvalue:	.word 0
prompt:	.asciz "Enter x: "
in:	    .asciz "%d"
out:	.asciz "Summation is: %d\n"
out2:	.asciz "Two times the summation is: %d\n"
