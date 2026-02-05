import numpy as np
import gridCuda

def main():
    print("Testing gridCuda bindings....")

    # print some constants
    print(f"Number of joints: {gridCuda.NUM_JOINTS}")

    # generate random joint positions, velocities, and controls
    np.random.seed(0)
    q = np.random.normal(0, 1, gridCuda.NUM_JOINTS).astype(np.float32)
    qd = np.random.normal(0, 1, gridCuda.NUM_JOINTS).astype(np.float32)
    u = np.random.normal(0, 1, gridCuda.NUM_JOINTS).astype(np.float32)

    # create instance of module with default gravity 
    grid = gridCuda.GRidDataFloat(q, qd, u)

    # print the inputs
    print("\nJoint positions (q):")
    print(q)
    
    print("\nJoint velocities (qd):")
    print(qd)
    
    print("\Torque inputs (u):")
    print(u)

    # Compute and print end effector positions
    ee_pos = grid.get_end_effector_positions()
    print("\nEnd Effector Positions:")
    print(ee_pos.reshape(-1, 6))  # Reshape to make it more readable
    
    # Compute inverse dynamics
    c = grid.inverse_dynamics()
    print("\nInverse Dynamics (c):")
    print(c)
    
    # Compute mass matrix inverse
    Minv = grid.minv()
    print("\nMass Matrix Inverse (Minv):")
    print(Minv)
    
    # Compute joint accelerations
    qdd = grid.forward_dynamics()
    print("\nJoint Accelerations (qdd):")
    print(qdd)
    
    # Compute inverse dynamics gradient
    dc = grid.inverse_dynamics_gradient()
    print("\nInverse Dynamics Gradient dc_dq:")
    print(dc[0])  # First matrix is dc_dq
    print("\nInverse Dynamics Gradient dc_dqd:")
    print(dc[1])  # Second matrix is dc_dqd
    
    # Compute forward dynamics gradient
    df = grid.forward_dynamics_gradient()
    print("\nForward Dynamics Gradient df_dq:")
    print(df[0])  # First matrix is df_dq
    print("\nForward Dynamics Gradient df_dqd:")
    print(df[1])  # Second matrix is df_dqd


if __name__ == "__main__":
    main()

