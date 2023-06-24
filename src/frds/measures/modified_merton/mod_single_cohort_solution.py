from frds.measures.modified_merton.mod_single_cohort_computation import (
    mod_single_cohort_computation,
)


def mod_single_cohort_solution(b, param, N, Nsim2, E, sigE):
    fs = b[0]
    param[2] = b[1]

    # fmt: off
    Lt, Bt, Et, LH, BH, EH, sigEt, mFt, default, mdef, face, FH, Gt, mu, F, sigLt = mod_single_cohort_computation(fs, param, N, Nsim2)
    # fmt: on
    err = [E - Et, sigE - sigEt]

    return err
