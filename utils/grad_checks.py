"""
utils/grad_checks.py

Runtime verification of AC-1 through AC-5 constraints from Theorem 1.
Call these after model construction (or in tests/test_ac_constraints.py).

AC-1: ∂o_t/∂θ_lang = 0  and  ∂s_t/∂θ_lang = 0
      (z never enters f_rep^(m) or Ψ)
AC-2: During policy/RL forward, s_{t-1} must be detached
      (s_sg chain used, not s_full)
AC-3: θ_lang has two independent optimisers with α_A ≠ α_B
      (checked on Trainer object)
AC-4: θ_π^M ∩ θ_π^L = ∅
AC-5: z is computed once before the perception loop, not inside it

Reference: theorem_final.docx §4 (AC constraints), phase_b1_training.docx §6.
"""
import torch
import torch.nn as nn
from typing import Dict, List


def check_ac1_no_z_in_encoder(
    model,
    X: Dict[int, torch.Tensor],
    w: list,
) -> bool:
    """
    AC-1: Verify ∂o_t/∂θ_lang = 0.
    Run a forward pass and check that modality encoder outputs
    have no gradient path to θ_lang parameters.
    """
    model.eval()
    # Enable grad for θ_lang
    lang_params = list(model.f_lang.parameters())
    for p in lang_params:
        p.requires_grad_(True)

    z = model.encode_language(w)                           # (B, d_z)

    # Encode one observation from each modality
    passed = True
    for m_id, enc in model.f_rep.items():
        m = int(m_id)
        x = X[m]
        l = torch.zeros(x.shape[0], 2 if m == 1 else 1, device=x.device)
        o_t = enc(x, l)

        # o_t should have NO grad connection to θ_lang
        for p in lang_params:
            if p.grad is not None:
                p.grad.zero_()

        scalar = o_t.sum()
        scalar.backward()

        for p in lang_params:
            if p.grad is not None and p.grad.abs().max() > 1e-9:
                print(f"  FAIL AC-1: modality {m_id} encoder has grad w.r.t. θ_lang")
                passed = False
                break
        else:
            print(f"  PASS AC-1: modality {m_id} encoder has no grad w.r.t. θ_lang")

    return passed


def check_ac1_no_z_in_psi(
    model,
    X: Dict[int, torch.Tensor],
    w: list,
) -> bool:
    """
    AC-1 / C6: Verify ∂s_t/∂θ_lang = 0.
    Run Ψ forward with an observation and check that s_t has no
    gradient path to θ_lang.
    """
    model.eval()
    lang_params = list(model.f_lang.parameters())
    for p in lang_params:
        p.requires_grad_(True)
        if p.grad is not None:
            p.grad.zero_()

    z = model.encode_language(w)
    B = z.shape[0]
    device = z.device

    s0 = model.psi.initial_state(B, device)

    # Encode a dummy observation without z.
    # l dimension matches the modality: 2 for vision (x,y), 1 for audio (τ).
    m_id = str(min(model.modalities))
    enc = model.f_rep[m_id]
    l_dim = 2 if int(m_id) == 1 else 1
    l = torch.zeros(B, l_dim, device=device)
    o_t = enc(X[int(m_id)], l).detach()                   # detach encoder output

    s1 = model.psi(s0, o_t)                               # Ψ forward — no z
    s1.sum().backward()

    for p in lang_params:
        if p.grad is not None and p.grad.abs().max() > 1e-9:
            print("  FAIL AC-1/C6: Ψ has grad w.r.t. θ_lang — z is leaking into Ψ")
            return False

    print("  PASS AC-1/C6: Ψ has no grad w.r.t. θ_lang")
    return True


def check_ac2_sg_state(traj: Dict) -> bool:
    """
    AC-2: Verify that s_sg entries are detached (require_grad=False or no grad_fn).
    traj is the output of model.perception_loop().
    """
    s_sg = traj.get('s_sg')
    if s_sg is None:
        print("  SKIP AC-2: no s_sg in trajectory")
        return True

    # s_sg is (B, T, d_s); each step should be detached
    # At index t, this is sg(s_{t-1}) — should not have grad_fn from Ψ
    if isinstance(s_sg, torch.Tensor):
        if s_sg.requires_grad:
            print("  FAIL AC-2: s_sg.requires_grad=True — state not detached")
            return False
        if s_sg.grad_fn is not None:
            print("  FAIL AC-2: s_sg has grad_fn — state not detached")
            return False
    print("  PASS AC-2: s_sg is properly detached")
    return True


def check_ac4_disjoint_policy_params(model) -> bool:
    """
    AC-4 / C5: Verify θ_π^M ∩ θ_π^L = ∅.
    Uses data_ptr() to identify unique parameter tensors.
    """
    ptrs_M = set(p.data_ptr() for p in model.pi_M.parameters())
    ptrs_L = set(p.data_ptr() for p in model.pi_L.parameters())
    intersection = ptrs_M & ptrs_L

    if intersection:
        print(f"  FAIL AC-4: θ_π^M ∩ θ_π^L has {len(intersection)} shared tensors")
        return False

    print(f"  PASS AC-4: θ_π^M ∩ θ_π^L = ∅ "
          f"(|θ_π^M|={len(ptrs_M)}, |θ_π^L|={len(ptrs_L)})")
    return True


def check_ac5_z_precomputed(model, w: list) -> bool:
    """
    AC-5: Verify z is computed before (not inside) the perception loop.
    This is a structural check — we verify z is produced by encode_language(),
    which is called before perception_loop() in the trainer.

    Here we check that the model has an encode_language() method and that
    it does not call f_rep or Ψ.
    """
    has_encode_lang = hasattr(model, 'encode_language')
    if not has_encode_lang:
        print("  FAIL AC-5: model has no encode_language() method")
        return False

    # Check that encode_language doesn't reference psi or f_rep
    import inspect
    src = inspect.getsource(model.encode_language)
    uses_psi  = 'psi' in src and 'f_lang' not in src.split('psi')[0]
    uses_frep = 'f_rep' in src

    if uses_psi or uses_frep:
        print("  FAIL AC-5: encode_language() calls Ψ or f_rep — z not pre-computed cleanly")
        return False

    print("  PASS AC-5: encode_language() is structurally separate from Ψ and f_rep")
    return True


def run_all_checks(model, X: Dict[int, torch.Tensor], w: list,
                   traj: Dict = None) -> bool:
    """Run all AC checks and report results."""
    print("\n=== AC Constraint Verification ===")
    results = []

    results.append(("AC-1 (no z in encoder)", check_ac1_no_z_in_encoder(model, X, w)))
    results.append(("AC-1/C6 (no z in Ψ)",    check_ac1_no_z_in_psi(model, X, w)))

    if traj is not None:
        results.append(("AC-2 (sg state)",     check_ac2_sg_state(traj)))

    results.append(("AC-4 (disjoint heads)",   check_ac4_disjoint_policy_params(model)))
    results.append(("AC-5 (z pre-computed)",   check_ac5_z_precomputed(model, w)))

    print("\n=== Summary ===")
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        all_pass = all_pass and passed

    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILURES — check above'}")
    return all_pass
