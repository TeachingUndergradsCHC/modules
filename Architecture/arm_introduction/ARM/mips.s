.text
.globl main

main: li $5, 0
	li $2, 1
	li $3, 10
loop:  bgt $2, $3, exit
       andi $4, $2, 1
       bnez $4, skip
       add $5, $5, $2
skip:       addi $2, $2, 1
       j loop
exit: nop
