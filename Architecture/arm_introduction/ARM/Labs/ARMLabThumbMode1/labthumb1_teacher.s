@------------------
@	ToUCH /	moduleC1 / Thumb lab 1
@	Author: Gabriel Hernandez
@	Date: May.9.2019
@	Description: This code will run a loop based summation function 
@		using standard function requirements in Thumb and Arm mode.
@		This is one way that students can complete this lab.
@	Register Use: R0 will hold the value x and hold the summation of the
@		value x. R1 will be used to add the values of 0 to x. R2 will
@		be used to store the summation of x as it goes through a loop.
@		R3 is used to assist R1 when in thumb mode.
@	
@------------------
	.extern printf
	.extern scanf
	.global main
	.func main
	.arm

main:
@------------------------------------------------------------Given
	PUSH	{LR}
	LDR	R0, =prompt	@get address of prompt
	BL	printf		@prints Prompt
	LDR	R0, =in		@Gets input string of prompt
	LDR	R1, =xvalue	@Gets entered value of x
	BL 	scanf		@Put value in x using scanf
	LDR	R0, =xvalue	@loads value of x into R0
	LDR	R0, [R0]	@x is now in R0

				@BLX is used to branch between thumb and arm mode

@---------------------------------------------------------------------
                            @create code to call sum function
                                @Calculate 2 times the summation store it into R4
	BLX 	thumb_sum	@Branch with link exchange to thumb_sum 
	@BL 	 _sum		@Branch with link to _sum
	LSL 	R4, R0, #1	@Multiply R0 by 2, Store result in R4


@-------------------------------------------------------------------- Given

	MOV 	R1, R0      @stores summation in R1 to return summation
	LDR	R0, =out    @loads string out into R0
	BL	printf      @prints string out with R1 summation
	MOV	R1, R4      @stores R4 into R1 to return 2*summation
	LDR 	R0, =out2   @Loads string out2 into R0
	BL  	printf      @prints string out2 with 2*summation
	POP 	{PC}        @ends program

@-------------------------------------------------------------------------------------------
                            @create a sum fuction using standard function requirements
                            @function should be based on loop return result in R0
_sum:
	PUSH    {LR}		@pushes link register to stack
	MOV	R1, #0		@stores 0 in R1
	MOV	R2, #0		@stores 0 in R2

    _loop:			@create loop
	CMP	R0, R1		@Compare R0 and R1
	ADD	R2, R1, R2	@Add R1 and R2, Store result in R2
	ADD	R1, R1, #1	@Add 1 to R1, Store result in R1
	BGT	_loop		@Branch back to loop if R0>R1
	MOV     R0, R2		@Store summation result from R2 into R0

    POP     {PC}                @ends function call

@----------------------------------------------------------------------------------------
                            @create thumb mode sum function
                            @based on loop return result in R0

.thumb
thumb_sum:
	MOV     R1, #0		@store 0 in R1
	MOV     R2, #0		@store 0 in R2
	MOV	R3, #1		@store 1 in R3

    _loopthumb:			@create loop

	ADD     R2, R1		@add R1 to R2, Store result in R2
	ADD     R1, R3		@add 1 to R1, Store result in R1
	CMP     R0, R1		@compare R0 and R1
	BGE     _loopthumb	@branch back to loop if R0>=R1
	MOV     R0, R2		@store summation result from R2 into R0
	BX      LR		@exit from function


@----------------------------------------------------------------------
.data
xvalue:	.word 0
prompt:	.asciz "Enter x: "
in:	    .asciz "%d"
out:	.asciz "Summation is: %d\n"
out2:	.asciz "Two times the summation is: %d\n"
