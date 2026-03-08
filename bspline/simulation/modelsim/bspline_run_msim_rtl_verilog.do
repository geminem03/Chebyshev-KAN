transcript on
if {[file exists rtl_work]} {
	vdel -lib rtl_work -all
}
vlib rtl_work
vmap work rtl_work

vlog -vlog01compat -work work +incdir+C:/Skule_Hardrive/Chebyshev-KAN/bspline {C:/Skule_Hardrive/Chebyshev-KAN/bspline/layer.v}
vlog -vlog01compat -work work +incdir+C:/Skule_Hardrive/Chebyshev-KAN/bspline {C:/Skule_Hardrive/Chebyshev-KAN/bspline/bspline_weights.v}
vlog -vlog01compat -work work +incdir+C:/Skule_Hardrive/Chebyshev-KAN/bspline {C:/Skule_Hardrive/Chebyshev-KAN/bspline/bspline_basis.v}
vlog -sv -work work +incdir+C:/Skule_Hardrive/Chebyshev-KAN/bspline {C:/Skule_Hardrive/Chebyshev-KAN/bspline/pe.sv}
vlog -sv -work work +incdir+C:/Skule_Hardrive/Chebyshev-KAN/bspline {C:/Skule_Hardrive/Chebyshev-KAN/bspline/pe_column.sv}

