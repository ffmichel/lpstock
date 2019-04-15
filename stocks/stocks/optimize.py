import copy

from scipy import optimize


def branch_and_bound(c, A_ub, b_ub, bounds):
    # linear program
    res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    # branch and bound:
    for idx in range(len(c)):
        new_bounds_left = copy.deepcopy(bounds)
        new_bounds_left[idx][0] = int(res.x[idx]) + 1
        res_left = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub,
                                    bounds=new_bounds_left)
        new_bounds_right = copy.deepcopy(bounds)
        new_bounds_right[idx][1] = int(res.x[idx])
        res_right = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub,
                                     bounds=new_bounds_right)
        if res_left.fun < res_right.fun:
            res = res_left
            bounds = copy.deepcopy(new_bounds_left)
            bounds[idx][1] = bounds[idx][0]
        else:
            res = res_right
            bounds = copy.deepcopy(new_bounds_right)
            bounds[idx][0] = bounds[idx][1]
    return res
