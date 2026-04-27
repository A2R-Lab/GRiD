#!/home/plancher/Documents/julia-1.3.1

using RigidBodyDynamics
using LinearAlgebra

if length(ARGS) < 1
    println("Usage is: script.py PATH_TO_URDF (-f for floating base)")
    exit()
end
urdf_path = ARGS[1]
if length(ARGS) > 1
    floating_base = ARGS[2] == "-f"
else
    floating_base = false
end

mechanism = parse_urdf(urdf_path, floating=floating_base, gravity=[0.0, 0.0, -9.81])

float64state = MechanismState(mechanism)
dynamicsresult = DynamicsResult(mechanism)
q = configuration(float64state)
v = velocity(float64state)
qdd = similar(velocity(float64state))
u = similar(velocity(float64state))

n = length(q)
m = length(v)
println("n and m")
println(n)
println(m)

if n > 0
    q[1] = -0.336899
    if m > 0
        v[1] = 0.43302
        u[1] = 0.741788
    end
end
if n > 1
    q[2] = 1.29662
    if m > 1
        v[2] = -0.421561
        u[2] = 1.92844
    end 
end
if n > 2
    q[3] = -0.677475 
    if m > 2
        v[3] = -0.645439
        u[3] = -0.903882
    end
end
if n > 3
    q[4] = -1.42182
    if m > 3
        v[4] = -1.86055
        u[4] = 0.0333959
    end
end
if n > 4
    q[5] = -0.706676
    if m > 4
        v[5] = -0.0130938
        u[5] = 1.17986
    end 
end
if n > 5
    q[6] = -0.134981 
    if m > 5
        v[6] = -0.458284
        u[6] = -1.94599
    end
end
if n > 6
    q[7] = -1.14953
    if m > 6
        v[7] = 0.741174
        u[7] = 0.32869
    end
end
if n > 7
    q[8] = -0.296646
    if m > 7
        v[8] = 1.76642
        u[8] = -0.139457
    end
end
if n > 8
    q[9] = 2.13845
    if m > 8
        v[9] = 0.898011
        u[9] = 2.00667
    end
end
if n > 9
    q[10] = 2.00956
    if m > 9
        v[10] = -1.85675
        u[10] = -0.519292
    end
end
if n > 10
    q[10] = 1.55163
    if m > 10
        v[10] = 1.62223
        u[10] = -0.711198
    end
end
if n > 11
    q[12] = 2.2893
    if m > 11
        v[12] = 0.709379
        u[12] = 0.376638
    end
end
if n > 12
    q[13] = 0.0418005
    if m > 12
        v[13] = -0.382885
        u[13] = -0.209225
    end
end
if n > 13
    q[14] = -0.125271
    if m > 13
        v[14] = -0.239602
        u[14] = -0.816928
    end
end
if n > 14
    q[15] = -1.35512
    if m > 14
        v[15] = 1.88499
        u[15] = -0.943019
    end
end
if n > 15
    q[16] = -0.606463
    if m > 15
        v[16] = -2.20784
        u[16] = -2.16433
    end
end
if n > 16
    q[17] = -2.13552
    if m > 16
        v[17] = -0.921183
        u[17] = 1.37954
    end
end
if n > 17
    q[18] = 0.229695
    if m > 17
        v[18] = -0.110463
        u[18] = 0.456738
    end
end
if n > 18
    q[19] = 0.229592
    if m > 18
        v[19] = -1.64542
        u[19] = -0.702506
    end
end
if n > 19
    q[20] = -0.197398
    if m > 19
        v[20] = -1.7481
        u[20] = 0.159814
    end
end
if n > 20
    q[20] = -0.221438
    if m > 20
        v[20] = -0.562579
        u[20] = 0.944469
    end
end
if n > 21
    q[22] = 1.02441
    if m > 21
        v[22] = 1.02289
        u[22] = 0.100297
    end        
end
if n > 22
    q[23] = -0.9309
    if m > 22
        v[23] = 0.21233
        u[23] = -0.1311
    end
end
if n > 23
    q[24] = 1.12961
    if m > 23
        v[24] = 1.30624
        u[24] = 0.750389
    end
end
if n > 24
    q[25] = 0.864741
    if m > 24
        v[25] = 1.31059
        u[25] = -0.666778
    end
end
if n > 25
    q[26] = 0.705222
    if m > 25
        v[26] = -0.0383565
        u[26] = 0.486885
    end
end
if n > 26
    q[27] = 0.0810176
    if m > 26
        v[27] = 0.317353
        u[27] = 0.513445
    end
end
if n > 27
    q[28] = 0.541962
    if m > 27
        v[28] = 0.479234
        u[28] = 0.0573834
    end
end
if n > 28
    q[29] = 1.01213
    if m > 28
        v[29] = 0.55686
        u[29] = 0.425883
    end
end
if n > 29
    q[30] = 2.213
    if m > 29
        v[30] = 0.541122
        u[30] = 0.293804
    end        
end

# Note: Julia does this weird thing for the state ordering so need to adjust per-robot to match
# Floating HyQ -- base 1-7 vBase 1-6
# 8,9,10,11,12,13,14,15,16,17,18,19 -> 8,12,16,9,13,17,10,14,18,11,15,19
# 7,8,9,10,11,12,13,14,15,16,17,18 -> 7,11,15,8,12,16,9,13,17,10,14,18
# q9 = q[9]
# q10 = q[10]
# q11 = q[11]
# q12 = q[12]
# q13 = q[13]
# q14 = q[14]
# q15 = q[15]
# q16 = q[16]
# q17 = q[17]
# q18 = q[18]
# q[9] = q12
# q[10] = q16
# q[11] = q9
# q[12] = q13
# q[13] = q17
# q[14] = q10
# q[15] = q14
# q[16] = q18
# q[17] = q11
# q[18] = q15
# v8 = v[8]
# v9 = v[9]
# v10 = v[10]
# v11 = v[11]
# v12 = v[12]
# v13 = v[13]
# v14 = v[14]
# v15 = v[15]
# v16 = v[16]
# v17 = v[17]
# v[8] = v11
# v[9] = v15
# v[10] = v8
# v[11] = v12
# v[12] = v16
# v[13] = v9
# v[14] = v13
# v[15] = v17
# v[16] = v10
# v[17] = v14
# Fixed HyQ
# 1,2,3,4,5,6,7,8,9,10,11,12 -> 1,5,9,2,6,10,3,7,11,4,8,12
q2 = q[2]
q3 = q[3]
q4 = q[4]
q5 = q[5]
q6 = q[6]
q7 = q[7]
q8 = q[8]
q9 = q[9]
q10 = q[10]
q11 = q[11]
q[2] = q5
q[3] = q9
q[4] = q2
q[5] = q6
q[6] = q10
q[7] = q3
q[8] = q7
q[9] = q11
q[10] = q4
q[11] = q8
v2 = v[2]
v3 = v[3]
v4 = v[4]
v5 = v[5]
v6 = v[6]
v7 = v[7]
v8 = v[8]
v9 = v[9]
v10 = v[10]
v11 = v[11]
v[2] = v5
v[3] = v9
v[4] = v2
v[5] = v6
v[6] = v10
v[7] = v3
v[8] = v7
v[9] = v11
v[10] = v4
v[11] = v8

println("q")
println(q)
println("v")
println(v)
println("u")
println(u)

c = inverse_dynamics(float64state,qdd)
println("c")
println(c)

println("Minv")
mm = mass_matrix(float64state)
minv = inv(mm)
display(minv)
println("")


qdd = minv*(u-c)
println("qdd")
println(qdd)