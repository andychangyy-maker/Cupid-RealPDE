"""Bounded Newton solve for exact historical SPS with fixed absolute intervals."""
import numpy as np


def derivatives(scale, rows, weights):
    a,b,c,d,e,f = rows.transpose(2,0,1)
    rel = np.sqrt(np.maximum((a*scale**2-2*b*scale+c).sum(1),1e-20))
    tke = np.sqrt(np.maximum((d*scale**4-2*e*scale**2+f).sum(1),1e-20))
    v, z = a*scale-b, 2*d*scale**3-2*e*scale
    w = weights/max(float(weights.sum()),1e-20)
    dr, dt = .25/(.5+rel)**2, .15/(.5+tke)**2
    gradient = (w[:,None]*(dr[:,None]*v/rel[:,None]+dt[:,None]*z/tke[:,None])).sum(0)
    diagonal = (w[:,None]*(dr[:,None]*a/rel[:,None]
                          +dt[:,None]*(6*d*scale**2-2*e)/tke[:,None])).sum(0)
    cr = w*(dr/rel**3+.5/((.5+rel)**3*rel**2))
    ct = w*(dt/tke**3+.3/((.5+tke)**3*tke**2))
    hessian = np.diag(diagonal)-(v.T*cr)@v-(z.T*ct)@z
    loss = float(np.sum(w*(.5*rel/(.5+rel)+.3*tke/(.5+tke))))
    return loss,gradient,hessian


def solve(rows, weights, initial):
    value=np.clip(np.asarray(initial,dtype=np.float64),.8,1.2)
    loss,gradient,hessian=derivatives(value,rows,weights)
    converged=False
    for _ in range(64):
        fixed=((value<=.8+1e-12)&(gradient>0))|((value>=1.2-1e-12)&(gradient<0))
        free=np.flatnonzero(~fixed)
        if not len(free) or np.max(np.abs(gradient[free]))<1e-7:
            converged=True
            break
        local=hessian[np.ix_(free,free)]
        damping=max(0.,1e-6-float(np.linalg.eigvalsh(local)[0]))
        direction=np.zeros(32)
        direction[free]=-np.linalg.solve(local+np.eye(len(free))*damping,gradient[free])
        accepted=False
        for backtrack in range(16):
            candidate=np.clip(value+direction*(.5**backtrack),.8,1.2)
            step=candidate-value
            new_loss,new_gradient,new_hessian=derivatives(candidate,rows,weights)
            if gradient@step<0 and new_loss<=loss+1e-4*(gradient@step):
                accepted=True
                break
        if not accepted: break
        change=loss-new_loss
        value,loss,gradient,hessian=candidate,new_loss,new_gradient,new_hessian
        if change<1e-11:
            converged=True
            break
    if loss>derivatives(np.ones(32),rows,weights)[0]:value=np.ones(32)
    return value,converged
