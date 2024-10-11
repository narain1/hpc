	.file	"fma_test.c"
	.text
	.globl	main
	.type	main, @function
main:
.LFB0:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movss	.LC0(%rip), %xmm0
	movss	%xmm0, -4(%rbp)
	movss	.LC1(%rip), %xmm0
	movss	%xmm0, -8(%rbp)
	movss	.LC2(%rip), %xmm0
	movss	%xmm0, -12(%rbp)
	movss	-12(%rbp), %xmm1
	movss	-8(%rbp), %xmm0
	movl	-4(%rbp), %eax
	movaps	%xmm1, %xmm2
	movaps	%xmm0, %xmm1
	movd	%eax, %xmm0
	call	fmaf
	movd	%xmm0, %eax
	movl	%eax, -16(%rbp)
	movl	$0, %eax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	main, .-main
	.section	.rodata
	.align 4
.LC0:
	.long	1069547520
	.align 4
.LC1:
	.long	1075838976
	.align 4
.LC2:
	.long	1080033280
	.ident	"GCC: (Spack GCC) 12.1.0"
	.section	.note.GNU-stack,"",@progbits
