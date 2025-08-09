We will analyze the flow for the spring mass system
- Data
    - For each sample, we get a trajectory. By default 50 such samples, 25 for train, 25 for test
    - Each trajectory is run for a total of 3 time units. Each time unit is divided into 10 smaller units
    - Hence, for each trajectory we have 30 (q,p,q_dot,p_dot) pairs
    - We can simulate for any radius (basically any value of p^2+q^2, that is any value for hamiltonian)

- Train
    - test data is validation data
    - we train using half trajectories, we validate using the remaining half
    - baseline predicts the derivatives at each state, hnn predicts the hamiltonian, then differentiates it
    - but HNN still gives 2 outputs
    - F1: used to generate the conservative (Hamiltonian) field, F2: used to generate the solenoidal (non-conservative) field

- Analyze

- Ideas
    - Need not have 2 outputs in most simple cases
    - Hamiltonian equation is wrong (factor of 2 missing)
    - currently we are predicting the trajectory
    - baseline is completely off, can skip it