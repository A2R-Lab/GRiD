import numpy as np
import copy
np.set_printoptions(precision=4, suppress=True, linewidth = 100)

def test_rnea_fpass(self, q, qd, qdd = None, GRAVITY = -9.81):
    # allocate memory
    n = len(qd)
    v = np.zeros((6,n))
    a = np.zeros((6,n))
    f = np.zeros((6,n))

    # get constants
    gravity_vec = np.zeros((6))
    gravity_vec[5] = -GRAVITY # a_base is gravity vec
    n_bfs_levels = self.robot.get_max_bfs_level() + 1 # starts at 0
    parent_array = self.robot.get_parent_id_array()
    Imats = self.robot.get_Imats_ordered_by_id()[1:] # ignore base inertia

    # compute Xs
    Xmat_Funcs = self.robot.get_Xmat_Funcs_ordered_by_id()
    Xmats = [Xmat_Func(qi) for (Xmat_Func,qi) in zip(Xmat_Funcs,q)]
    
    # forward pass
    for bfs_level in range(n_bfs_levels):
        if bfs_level == 0: # all things with parent of base so v_base is 0 so just qd term
            # do in parallel
            inds = self.robot.get_ids_by_bfs_level(bfs_level)
            for ind in inds:
                v[:,ind] += self.robot.get_S_by_id(ind)*qd[ind] # codegen the S (slash do each type of S in parallel? or diverge if needed)
                a[:,ind] = np.matmul(Xmats[ind],gravity_vec)
                if qdd is not None:
                    a[:,ind] += self.robot.get_S_by_id(ind)*qdd[ind]
                if self.DEBUG_MODE:
                    print("v[" + str(ind) + "] = 0")
                    print(v[:,ind])
                    print("a[" + str(ind) + "] = gravity + S*qdd(if applicable)")
                    print(a[:,ind])
        else:
            # do in parallel Xmat then add qd/qdd
            inds = self.robot.get_ids_by_bfs_level(bfs_level)
            prev_inds = self.robot.get_ids_by_bfs_level(bfs_level-1)
            for ind in inds:
                v[:,ind] = np.matmul(Xmats[ind],v[:,parent_array[ind]]) # parent can't be base
                a[:,ind] = np.matmul(Xmats[ind],a[:,parent_array[ind]])
            # do in parallel the add qd/qdd
            # codegen the S (slash do each type of S in parallel? or diverge if needed)
            for ind in inds:
                v[:,ind] += self.robot.get_S_by_id(ind)*qd[ind]
                if qdd is not None:
                    a[:,ind] += self.robot.get_S_by_id(ind)*qdd[ind]
                if self.DEBUG_MODE:
                    print("v[" + str(ind) + "] = Xv_parent + S*qd")
                    print(v[:,ind])
                    print("a[" + str(ind) + "] = Xa_parent + S*qdd(if applicable)")
                    print(a[:,ind])
            # do in parallel finish the as with the Mx
            # codegen the S (slash do each type of S in parallel? or diverge if needed)
            for ind in inds:
                a[:,ind] += self.mxS(self.robot.get_S_by_id(ind),v[:,ind],qd[ind])
                if self.DEBUG_MODE:
                    print("a[" + str(ind) + "] += MxS(v)")
                    print(a[:,ind])
    # do all f in parallel
    for ind in range(n):
        Iv = np.matmul(Imats[ind],v[:,ind])
        Ia = np.matmul(Imats[ind],a[:,ind])
        f[:,ind] = Ia + self.fxv(v[:,ind],Iv)
        if self.DEBUG_MODE:
            print("Ia: " + str(ind))
            print(Ia)
            print("Iv: " + str(ind))
            print(Iv)
            print("f = Ia + fxv(v,Iv): " + str(ind))
            print(f[:,ind])

    return (v,a,f)

def test_rnea_bpass(self, q, qd, f):
    # allocate memory
    n = len(q) # assuming len(q) = len(qd)
    c = np.zeros(n)

    # get constants
    max_bfs_levels = self.robot.get_max_bfs_level()
    parent_array = self.robot.get_parent_id_array()
    
    # backward pass
    for bfs_level in range(max_bfs_levels,-1,-1):
        # do in parallel
        inds = self.robot.get_ids_by_bfs_level(bfs_level)
        for ind in inds:
            # compute c and codegen the S with each type in parallel or diverge if needed
            c[ind] = np.matmul(np.transpose(self.robot.get_S_by_id(ind)),f[:,ind])
            # update f if applicable (at bfs level 0 the parent is the root that we won't use)
            if bfs_level != 0:
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
                temp = np.matmul(np.transpose(Xmat),f[:,ind])
                f[:,parent_array[ind]] += temp.flatten()
                if self.DEBUG_MODE:
                    print("f[" + str(parent_array[ind]) + "_parent] = X^T*f[" + str(ind) + "]")
                    print(f[:,parent_array[ind]])

    # add velocity damping (defaults to 0)
    for k in range(n):
        c[k] += self.robot.get_damping_by_id(k) * qd[k]

    return (c,f)

def test_rnea(self, q, qd, qdd = None, GRAVITY = -9.81):
    # forward pass
    (v,a,f) = self.test_rnea_fpass(q, qd, qdd, GRAVITY)
    # backward pass
    (c,f) = self.test_rnea_bpass(q, qd, f)

    return (c,v,a,f)

def test_minv_bpass(self, q):

    # allocate memory
    n = len(q)
    Minv = np.zeros((n,n))
    F = np.zeros((n,6,n))
    U = np.zeros((n,6))
    Dinv = np.zeros(n)

    # set initial IA to I
    IA = copy.deepcopy(self.robot.get_Imats_dict_by_id())

    max_bfs_levels = self.robot.get_max_bfs_level()
    
    # backward pass
    for bfs_level in range(max_bfs_levels,-1,-1):
        # do in parallel
        inds = self.robot.get_ids_by_bfs_level(bfs_level)
        for ind in inds:
            # Compute U, D
            S = self.robot.get_S_by_id(ind)
            U[ind,:] = np.matmul(IA[ind],S)
            Dinv[ind] = 1/np.matmul(S.transpose(),U[ind,:])
            if self.DEBUG_MODE:
                print("U[" + str(ind) + "]")
                print(U[ind,:])
                print("Dinv[" + str(ind) + "] = " + str(Dinv[ind]))
            # Update Minv
            Minv[ind,ind] = Dinv[ind]
        if self.DEBUG_MODE:
            print("Minv after Dinv setting before subtree update")
            print(Minv)
        # do in parallel with above
        for ind in inds:
            subtreeInds = self.robot.get_subtree_by_id(ind)
            S = self.robot.get_S_by_id(ind)
            for subInd in subtreeInds:
                Minv[ind,subInd] -= Dinv[ind] * np.matmul(S.transpose(),F[ind,:,subInd])
        if self.DEBUG_MODE:
            print("Minv after subtree update")
            print(Minv)
        # do in parallel with above
        for ind in inds:
            # update parent if applicable
            parent_ind = self.robot.get_parent_id(ind)
            Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
            subtreeInds = self.robot.get_subtree_by_id(ind)
            if parent_ind != -1:
                # update F
                for subInd in subtreeInds:
                    F[ind,:,subInd] += U[ind,:]*Minv[ind,subInd]
                    F[parent_ind,:,subInd] += np.matmul(np.transpose(Xmat),F[ind,:,subInd]) 
                # update IA
                Ia = IA[ind] - np.outer(U[ind,:],Dinv[ind]*U[ind,:])
                IaParent = np.matmul(np.transpose(Xmat),np.matmul(Ia,Xmat))
                IA[parent_ind] += IaParent
                if self.DEBUG_MODE:
                    print("F Temp += U*Minv [" + str(ind) + "]")
                    print(F[ind,:,:])
                    print("Ia[" + str(ind) + "]")
                    print(Ia)
                    print("F_parent = X^T F Temp[" + str(parent_ind) + "]")
                    print(F[parent_ind,:,:])
                    print("IaX[" + str(ind) + "]")
                    print(np.matmul(Ia,Xmat))
                    print("I_parent = X^T IaX[" + str(parent_ind) + "]")
                    print(IA[parent_ind])
    return (Minv, F, U, Dinv)

def test_minv_fpass(self, q, Minv, F, U, Dinv):
    n = len(q)
    n_bfs_levels = self.robot.get_max_bfs_level() + 1 # starts at 0
    
    # forward pass
    # CANNOT BE IN PARALLEL BY BFS_LEVEL BECAUSE OF THE i:
    for ind in range(n):
        parent_ind = self.robot.get_parent_id(ind)
        S = self.robot.get_S_by_id(ind)
        Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
        if parent_ind != -1:
            Minv[ind,ind:] -= Dinv[ind]*np.matmul(np.matmul(U[ind,:].transpose(),Xmat),F[parent_ind,:,ind:])
        F[ind,:,ind:] = np.outer(S,Minv[ind,ind:])
        if parent_ind != -1:
            F[ind,:,ind:] += np.matmul(Xmat,F[parent_ind,:,ind:])

    return Minv

def test_densify_Minv(self, Minv):
    Minv_dense = copy.deepcopy(Minv)
    n = self.robot.get_num_pos()
    for row in range(n):
        for col in range(n):
            if row > col:
                Minv_dense[row,col] = Minv[col,row]
    return Minv_dense

def test_minv(self, q, output_dense = True):
    # based on https://www.researchgate.net/publication/343098270_Analytical_Inverse_of_the_Joint_Space_Inertia_Matrix

    # backward pass
    (Minv, F, U, Dinv) = self.test_minv_bpass(q)

    # forward pass
    Minv = self.test_minv_fpass(q, Minv, F, U, Dinv)

    # fill in full matrix (currently only upper triangular)
    if output_dense:
        Minv = self.test_densify_Minv(Minv)

    return Minv


def test_rnea_grad_inner(self, q, qd, v, a, f, GRAVITY = -9.81):
    
    # allocate memory
    n = len(qd)
    max_bfs_levels = self.robot.get_max_bfs_level()
    n_bfs_levels = max_bfs_levels + 1 # starts at 0
    MxXv = np.zeros((6,n))
    MxXa = np.zeros((6,n))
    Iv = np.zeros((6,n))
    Mxv = np.zeros((6,n))
    Mxf = np.zeros((6,n))
    FxvI = np.zeros((6,6,n))
    dv_dq = np.zeros((6,n,n))
    dv_dqd = np.zeros((6,n,n))
    da_dq = np.zeros((6,n,n))
    da_dqd = np.zeros((6,n,n))
    df_dq = np.zeros((6,n,n))
    df_dqd = np.zeros((6,n,n))
    dc_dq = np.zeros((n,n))
    dc_dqd = np.zeros((n,n))

    print("dv,da needed Cols: " + str(self.robot.get_total_ancestor_count() + n))
    print("Possible Cols: " + str(n*n))
    print("df needed Cols: " + str(self.robot.get_total_ancestor_count() + self.robot.get_total_subtree_count()))
    print("Possible Cols: " + str(n*n))

    gravity_vec = np.zeros((6))
    gravity_vec[5] = -GRAVITY # a_base is gravity vec

    if self.DEBUG_MODE:
        print("q")
        print(q)
        print("qd")
        print(qd)
        print("v")
        print(v)
        print("a")
        print(a)
        print("f")
        print(f)
        for ind in range(n):
            print("X[" + str(ind) + "]")
            print(self.robot.get_Xmat_Func_by_id(ind)(q[ind]))
        for ind in range(n):
            print("I[" + str(ind) + "]")
            print(self.robot.get_Imat_by_id(ind))

    #
    # Main temp comps
    #
    # first compute temporary values by type of operation
    # in theory we can use part of FxvI temp mem for Xv and Xa initial comps
    # but for now we'll just have extra mem because easier in python
    Xv = np.zeros((6,n))
    Xa = np.zeros((6,n))
    for ind in range(n): # do in parallel
        parent_ind = self.robot.get_parent_id(ind)
        Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
        Imat = self.robot.get_Imat_by_id(ind)
        for row in range(6): # do in parallel with ind
            if parent_ind != -1:
                Xv[row,ind] = np.matmul(Xmat[row,:],v[:,parent_ind])
                Xa[row,ind] = np.matmul(Xmat[row,:],a[:,parent_ind])
            else:
                Xv[row,ind] = 0
                Xa[row,ind] = np.matmul(Xmat[row,:],gravity_vec)
            Iv[row,ind] = np.matmul(Imat[row,:],v[:,ind])

    if self.DEBUG_MODE:
        print("Iv")
        print(Iv)
        print("Xv")
        print(Xv)
        print("Xa")
        print(Xa)

    # then do the mx comps
    for ind in range(n): # do in parallel
        S = self.robot.get_S_by_id(ind)
        MxXv[:,ind] = self.mxS(S,Xv[:,ind])
        MxXa[:,ind] = self.mxS(S,Xa[:,ind])
        Mxv[:,ind] = self.mxS(S,v[:,ind])
        Mxf[:,ind] = self.mxS(S,f[:,ind])

    if self.DEBUG_MODE:
        print("MxXv")
        print(MxXv)
        print("MxXa")
        print(MxXa)
        print("Mxv")
        print(Mxv)
        print("Mxf")
        print(Mxf)

    #
    # FORWARD PASS
    #
    # then serial dv/du in bfs waves
    for bfs_level in range(n_bfs_levels):
        inds = self.robot.get_ids_by_bfs_level(bfs_level)                
        for ind in inds: # do in parallel
            Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
            S = self.robot.get_S_by_id(ind)
            parent_ind = self.robot.get_parent_id(ind)
            # ONLY NEED TO GO THROUGH ANCESTORS
            ancestor_inds = self.robot.get_ancestors_by_id(ind)
            for col in ancestor_inds: # do in parallel with ind
                for row in range(6): # do in parallel with ind
                    # dv/du = X dv_parent/du + {MxXv or S for col ind}
                    dv_dq[row,col,ind] = np.matmul(Xmat[row,:],dv_dq[:,col,parent_ind])
                    dv_dqd[row,col,ind] = np.matmul(Xmat[row,:],dv_dqd[:,col,parent_ind])
            # THEN ADD SELF
            for row in range(6): # do in parallel with ind
                if bfs_level != 0:
                    dv_dq[row,ind,ind] += MxXv[row,ind] # because lambda v needs a parent
                dv_dqd[row,ind,ind] += S[row] # all joints have an S
            if self.DEBUG_MODE:
                print("dv[" + str(ind) + "]_dq")
                print(dv_dq[:,:,ind])
                print("dv[" + str(ind) + "]_dqd")
                print(dv_dqd[:,:,ind])

    # then in parallel da/du = MxS(dv/du)*qd + {MxXa, Mxv}
    for ind in range(n): # do in parallel
        S = self.robot.get_S_by_id(ind)
        # NEED TO GO THROUGH ANCESTORS AND SELF
        col_inds = self.robot.get_ancestors_by_id(ind)
        col_inds.append(ind)
        for col in col_inds: # do in parallel with ind
            da_dq[:,col,ind] = self.mxS(S,dv_dq[:,col,ind],qd[ind])
            da_dqd[:,col,ind] = self.mxS(S,dv_dqd[:,col,ind],qd[ind])
            if col == ind:
                da_dq[:,col,ind] += MxXa[:,ind]
                da_dqd[:,col,ind] += Mxv[:,ind]
        if self.DEBUG_MODE:
            print("da[" + str(ind) + "]_dq part 1 = MxS(dv/du)*qd + {MxXa, Mxf}")
            print(da_dq[:,:,ind])
            print("da[" + str(ind) + "]_dqd part 1 = MxS(dv/du)*qd + {MxXa, Mxf}")
            print(da_dqd[:,:,ind])

    # then in serial update da/du += X*da_parent/du
    for bfs_level in range(1, n_bfs_levels): # SKIP BFS = 0 because no parent ----- !!!!!!!!!
        inds = self.robot.get_ids_by_bfs_level(bfs_level)           
        for ind in inds: # do in parallel
            parent_ind = self.robot.get_parent_id(ind)
            Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
            # NEED TO GO THROUGH ANCESTORS AND SELF
            col_inds = self.robot.get_ancestors_by_id(ind)
            col_inds.append(ind)
            for col in col_inds: # do in parallel with ind
                for row in range(6): # do in parallel with ind
                    da_dq[row,col,ind] += np.matmul(Xmat[row,:],da_dq[:,col,parent_ind])
                    da_dqd[row,col,ind] += np.matmul(Xmat[row,:],da_dqd[:,col,parent_ind])
            if self.DEBUG_MODE:
                print("da[" + str(ind) + "]_dq += X*da_parent/dq")
                print(da_dq[:,:,ind])
                print("da[" + str(ind) + "]_dqd += X*da_parent/dqd")
                print(da_dqd[:,:,ind])

    # then the df/du = fxdv/du*Iv and temp var Fxv*I
    for ind in range(n): # do in parallel
        Imat = self.robot.get_Imat_by_id(ind)
        # NEED TO GO THROUGH ANCESTORS AND SELF
        col_inds = self.robot.get_ancestors_by_id(ind)
        col_inds.append(ind)
        df_len = len(col_inds)
        col_inds.extend(list(range(6)))
        for i in range(len(col_inds)): # do in parallel with ind
            col = col_inds[i]
            if i < df_len:
                df_dq[:,col,ind] = self.fxv(dv_dq[:,col,ind],Iv[:,ind])
                df_dqd[:,col,ind] = self.fxv(dv_dqd[:,col,ind],Iv[:,ind])
            else:
                Imat = self.robot.get_Imat_by_id(ind)
                FxvI[:,col,ind] = self.fxv(v[:,ind],Imat[:,col])

        if self.DEBUG_MODE:
            print("df[" + str(ind) + "]_dq part 1 = fxdv/du*Iv")
            print(df_dq[:,:,ind])
            print("df[" + str(ind) + "]_dqd part 1 = fxdv/du*Iv")
            print(df_dqd[:,:,ind])
            print("fxvI[" + str(ind) + "]")
            print(FxvI[:,:,ind])

    # then in parallel compute df/du += Ia + FxvI*dv/du
    for ind in range(n): # do in parallel
        Imat = self.robot.get_Imat_by_id(ind)
        # NEED TO GO THROUGH ANCESTORS AND SELF
        col_inds = self.robot.get_ancestors_by_id(ind)
        col_inds.append(ind)
        for col in col_inds: # do in parallel with ind
            for row in range(6): # do in parallel with ind
                df_dq[row,col,ind] += np.matmul(Imat[row,:],da_dq[:,col,ind]) + \
                                      np.matmul(FxvI[row,:,ind],dv_dq[:,col,ind])
                df_dqd[row,col,ind] += np.matmul(Imat[row,:],da_dqd[:,col,ind]) + \
                                       np.matmul(FxvI[row,:,ind],dv_dqd[:,col,ind])
        if self.DEBUG_MODE:
            print("df[" + str(ind) + "]_dq += Ia + FxvI*dv/du")
            print(df_dq[:,:,ind])
            print("df[" + str(ind) + "]_dqd += Ia + FxvI*dv/du")
            print(df_dqd[:,:,ind])

    # and also at the same time compute the temp var -X^T * mxf
    # since all temps are done re-use one in practice
    Xmxf = np.zeros((6,n))
    for ind in range(n): # do in parallel
        XmatT = self.robot.get_Xmat_Func_by_id(ind)(q[ind]).transpose()
        for row in range(6): # do in parallel with inds
            Xmxf[row,ind] = -np.matmul(XmatT[row,:],Mxf[:,ind])
        if self.DEBUG_MODE:
            print("-X^T * mxf[" + str(ind) + "]")
            print(Xmxf[:,ind])

    # for debug in python save down fp df_du
    df_fp_dq = copy.deepcopy(df_dq)
    df_fp_dqd = copy.deepcopy(df_dqd)

    #
    # BACKWARD PASS
    #
    # update df serially (df_lambda/du += X^T * df/du + {Xmxf, 0})
    for bfs_level in range(max_bfs_levels,0,-1): # STOP AT 1 because updating parent and last is 0 ---- !!!!!
        inds = self.robot.get_ids_by_bfs_level(bfs_level)
        parent_inds = [self.robot.get_parent_id(ind) for ind in inds]
        if self.DEBUG_MODE:
            for ind in self.robot.get_unique_parent_ids(inds):
                print("df[" + str(ind) + "]_dq (parent) BEFORE UPDATE += X^T * df/du + {Xmxf, 0})")
                print(df_dq[:,:,ind])
                print("df[" + str(ind) + "]_dqd (parent) BEFORE UPDATE  += X^T * df/du + {Xmxf, 0})")
                print(df_dqd[:,:,ind])         
        for ind in inds: # do in parallel with ind
            parent_ind = self.robot.get_parent_id(ind)
            XmatT = self.robot.get_Xmat_Func_by_id(ind)(q[ind]).transpose()
            # NEED TO GO THROUGH ANCESTORS AND SUBTREE
            col_inds = self.robot.get_ancestors_by_id(ind)
            col_inds.extend(self.robot.get_subtree_by_id(ind))
            for col in col_inds: # do in parallel with ind  
                for row in range(6): # do in parallel with ind
                    df_dq[row,col,parent_ind] += np.matmul(XmatT[row,:],df_dq[:,col,ind])
                    df_dqd[row,col,parent_ind] += np.matmul(XmatT[row,:],df_dqd[:,col,ind])
                    if col == ind:
                        df_dq[row,col,parent_ind] += Xmxf[row,ind]
        if self.DEBUG_MODE:
            for ind in self.robot.get_unique_parent_ids(inds):
                print("df[" + str(ind) + "]_dq (parent) += X^T * df/du + {Xmxf, 0})")
                print(df_dq[:,:,ind])
                print("df[" + str(ind) + "]_dqd (parent) += X^T * df/du + {Xmxf, 0})")
                print(df_dqd[:,:,ind])

    # extract dc/du
    for ind in range(n): # do in parallel
        S = self.robot.get_S_by_id(ind)
        # NEED TO GO THROUGH ANCESTORS AND SUBTREE
        col_inds = self.robot.get_ancestors_by_id(ind)
        col_inds.extend(self.robot.get_subtree_by_id(ind))
        for col in col_inds: # do in parallel with ind  
            dc_dq[ind,col] = np.matmul(S.transpose(),df_dq[:,col,ind])
            dc_dqd[ind,col] = np.matmul(S.transpose(),df_dqd[:,col,ind]) + (self.robot.get_damping_by_id(ind) if ind == col else 0)

    return (dc_dq, dc_dqd, dv_dq, dv_dqd, da_dq, da_dqd, df_fp_dq, df_fp_dqd, df_dq, df_dqd)

def test_rnea_grad(self, q, qd, qdd = None, GRAVITY = -9.81):
    (c, v, a, f) = self.test_rnea(q, qd, qdd, GRAVITY)
    (dc_dq, dc_dqd, dv_dq, dv_dqd, da_dq, da_dqd, df_fp_dq, df_fp_dqd, df_dq, df_dqd) = self.test_rnea_grad_inner(q, qd, v, a, f, GRAVITY)
    dc_du = np.hstack((dc_dq,dc_dqd))
    return dc_du

def test_fd_grad(self, q, qd, u, GRAVITY = -9.81):
    n = self.robot.get_num_pos()
    (c, v, a, f) = self.test_rnea(q, qd, None, GRAVITY)
    Minv = self.test_minv(q, True)
    umc = u - c
    qdd = np.matmul(Minv,umc)
    (c, v, a, f) = self.test_rnea(q, qd, qdd, GRAVITY)
    dc_du = self.test_rnea_grad(q, qd, qdd, GRAVITY)
    df_du = -np.matmul(Minv,dc_du)
    if self.DEBUG_MODE:
        print("Minv")
        print(Minv)
        print("qdd")
        print(qdd)
        print("v")
        print(v)
        print("a")
        print(a)
        print("f")
        print(f)
        print("dc_dq")
        print(dc_du[:,:n])
        print("dc_dqd")
        print(dc_du[:,n:])
    return df_du

def mxS(self, S, vec, alpha = 1.0):
    if S[0] == 1:
        return self.mx0(vec,alpha)
    elif S[1] == 1:
        return self.mx1(vec,alpha)
    elif S[2] == 1:
        return self.mx2(vec,alpha)
    elif S[3] == 1:
        return self.mx3(vec,alpha)
    elif S[4] == 1:
        return self.mx4(vec,alpha)
    elif S[5] == 1:
        return self.mx5(vec,alpha)
    else:
        return np.zeros((6))

def mx0(self, vec, alpha = 1.0):
    vecX = np.zeros((6))
    try:
        vecX[1] = vec[2]*alpha
        vecX[2] = -vec[1]*alpha
        vecX[4] = vec[5]*alpha
        vecX[5] = -vec[4]*alpha
    except:
        vecX[1] = vec[0,2]*alpha
        vecX[2] = -vec[0,1]*alpha
        vecX[4] = vec[0,5]*alpha
        vecX[5] = -vec[0,4]*alpha
    return vecX

def mx1(self, vec, alpha = 1.0):
    vecX = np.zeros((6))
    try:
        vecX[0] = -vec[2]*alpha
        vecX[2] = vec[0]*alpha
        vecX[3] = -vec[5]*alpha
        vecX[5] = vec[3]*alpha
    except:
        vecX[0] = -vec[0,2]*alpha
        vecX[2] = vec[0,0]*alpha
        vecX[3] = -vec[0,5]*alpha
        vecX[5] = vec[0,3]*alpha
    return vecX

def mx2(self, vec, alpha = 1.0):
    vecX = np.zeros((6))
    try:
        vecX[0] = vec[1]*alpha
        vecX[1] = -vec[0]*alpha
        vecX[3] = vec[4]*alpha
        vecX[4] = -vec[3]*alpha
    except:
        vecX[0] = vec[0,1]*alpha
        vecX[1] = -vec[0,0]*alpha
        vecX[3] = vec[0,4]*alpha
        vecX[4] = -vec[0,3]*alpha
    return vecX

def mx3(self, vec, alpha = 1.0):
    vecX = np.zeros((6))
    try:
        vecX[4] = vec[2]*alpha
        vecX[5] = -vec[1]*alpha
    except:
        vecX[4] = vec[0,2]*alpha
        vecX[5] = -vec[0,1]*alpha
    return vecX

def mx4(self, vec, alpha = 1.0):
    vecX = np.zeros((6))
    try:
        vecX[3] = -vec[2]*alpha
        vecX[5] = vec[0]*alpha
    except:
        vecX[3] = -vec[0,2]*alpha
        vecX[5] = vec[0,0]*alpha
    return vecX

def mx5(self, vec, alpha = 1.0):
    vecX = np.zeros((6))
    try:
        vecX[3] = vec[1]*alpha
        vecX[4] = -vec[0]*alpha
    except:
        vecX[3] = vec[0,1]*alpha
        vecX[4] = -vec[0,0]*alpha
    return vecX

def mx(self, vec):
    return -self.fx(vec).transpose()

def fxS(self, S, vec, alpha = 1.0):
    return -self.mxS(S, vec, alpha)

def fx(self, vec):
    #   0  -v(2)  v(1)    0  -v(5)  v(4)
    # v(2)    0  -v(0)  v(5)    0  -v(3)
    #-v(1)  v(0)    0  -v(4)  v(3)    0
    #   0     0     0     0  -v(2)  v(1)
    #   0     0     0   v(2)    0  -v(0)
    #   0     0     0  -v(1)  v(0)    0
    result = np.zeros((6,6))
    result[0,1] = -vec[2]
    result[0,2] = vec[1]
    result[0,4] = -vec[5]
    result[0,5] = vec[4]

    result[1,0] = vec[2]
    result[1,2] = -vec[0]
    result[1,3] = vec[5]
    result[1,5] = -vec[3]

    result[2,0] = -vec[1]
    result[2,1] = vec[0]
    result[2,3] = -vec[4]
    result[2,4] = vec[3]

    result[3,4] = -vec[2]
    result[3,5] = vec[1]

    result[4,3] = vec[2]
    result[4,5] = -vec[0]

    result[5,3] = -vec[1]
    result[5,4] = vec[0]
    return result

def fxv(self, fxVec, timesVec):
    # Fx(fxVec)*timesVec
    #   0  -v(2)  v(1)    0  -v(5)  v(4)
    # v(2)    0  -v(0)  v(5)    0  -v(3)
    #-v(1)  v(0)    0  -v(4)  v(3)    0
    #   0     0     0     0  -v(2)  v(1)
    #   0     0     0   v(2)    0  -v(0)
    #   0     0     0  -v(1)  v(0)    0
    result = np.zeros((6))
    result[0] = -fxVec[2] * timesVec[1] + fxVec[1] * timesVec[2] - fxVec[5] * timesVec[4] + fxVec[4] * timesVec[5]
    result[1] =  fxVec[2] * timesVec[0] - fxVec[0] * timesVec[2] + fxVec[5] * timesVec[3] - fxVec[3] * timesVec[5]
    result[2] = -fxVec[1] * timesVec[0] + fxVec[0] * timesVec[1] - fxVec[4] * timesVec[3] + fxVec[3] * timesVec[4]
    result[3] =                                                     -fxVec[2] * timesVec[4] + fxVec[1] * timesVec[5]
    result[4] =                                                      fxVec[2] * timesVec[3] - fxVec[0] * timesVec[5]
    result[5] =                                                     -fxVec[1] * timesVec[3] + fxVec[0] * timesVec[4]
    return result

def mxv(self, fxVec, timesVec):
    # Fx(fxVec)*timesVec
    #   0  -v(2)  v(1)    0     0     0
    # v(2)    0  -v(0)    0     0     0  
    #-v(1)  v(0)    0     0     0     0
    #   0  -v(5)  v(4)    0  -v(2)  v(1)
    # v(5)    0  -v(3)   v(2)    0  -v(0)
    #-v(4)  v(3)    0   -v(1)  v(0)    0
    result = np.zeros((6))
    result[0] = -fxVec[2] * timesVec[1] + fxVec[1] * timesVec[2]
    result[1] =  fxVec[2] * timesVec[0] - fxVec[0] * timesVec[2]
    result[2] = -fxVec[1] * timesVec[0] + fxVec[0] * timesVec[1]
    result[3] = -fxVec[5] * timesVec[1] + fxVec[4] * timesVec[2] - fxVec[2] * timesVec[4] + fxVec[1] * timesVec[5]
    result[4] =  fxVec[5] * timesVec[0] - fxVec[3] * timesVec[2] + fxVec[2] * timesVec[3] - fxVec[0] * timesVec[5]
    result[5] = -fxVec[4] * timesVec[0] + fxVec[3] * timesVec[1] - fxVec[1] * timesVec[3] + fxVec[0] * timesVec[4]
    return result