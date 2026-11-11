"""Fixed credit capture; HOST synthetic checks and one parent-only CUDA matrix."""
import copy
import importlib.util
import types
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[2] / "examples" / "diagnose_mujoco_tyx_credit.py"
_SPEC = importlib.util.spec_from_file_location("credit_diagnostic", _PATH)
credit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(credit)

def test_gae_independent_two_step():
    result = credit.gae_components(np.array([[1.],[2.]]), np.array([[.5],[.6]]),
                                  np.zeros((2,1),bool), np.array([.7]), gamma=.9, lam=.8)
    delta1 = 2 + .9*.7 - .6
    delta0 = 1 + .9*.6 - .5
    expected = np.array([[delta0+.9*.8*delta1],[delta1]])
    np.testing.assert_allclose(result["gae"], expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(result["reward_gae"]+result["critic_gae"], expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(result["normalized"], (expected-expected.mean())/(expected.std(ddof=1)+1e-8))

def test_gae_done_blocks_bootstrap():
    result=credit.gae_components(np.array([[1.],[2.]]),np.zeros((2,1)),
                                np.ones((2,1),bool),np.array([999.]))
    np.testing.assert_array_equal(result["gae"],[[1.],[2.]])

@pytest.mark.parametrize("bad", ("shape","nonfinite","mask","bootstrap","short"))
def test_gae_refuses_bad_schema(bad):
    r=np.ones((3,2)); v=np.zeros((3,2)); d=np.zeros((3,2),bool); last=np.zeros(2)
    if bad=="shape": v=v[:2]
    elif bad=="nonfinite": r[0,0]=np.nan
    elif bad=="mask": d=d.astype(float)
    elif bad=="bootstrap": last=np.zeros(3)
    else: r=r[:0]; v=v[:0]; d=d[:0]
    with pytest.raises(ValueError):
        credit.gae_components(r,v,d,last)

def fields():
    return {"observations":np.arange(24,dtype=np.float32).reshape(6,4),
            "actions":np.arange(12,dtype=np.float32).reshape(6,2)/10,
            "log_prob":np.arange(6,dtype=np.float32).reshape(6,1)}

def test_exact_batch_bijection():
    original=fields(); order=np.array([4,0,5,2,1,3])
    batch={key:value[order] for key,value in original.items()}
    np.testing.assert_array_equal(credit.bind_batch(original,batch),order)

@pytest.mark.parametrize("bad", ("missing","duplicate","changed","dtype"))
def test_batch_refuses_ambiguous_or_corrupt(bad):
    original=fields(); batch={k:v.copy() for k,v in original.items()}
    if bad=="missing": batch={k:v[:-1] for k,v in batch.items()}
    elif bad=="duplicate":
        for value in original.values(): value[1]=value[0]
        batch={k:v.copy() for k,v in original.items()}
    elif bad=="changed": batch["actions"][1,0]+=.1
    else: batch["actions"]=batch["actions"].astype(np.float64)
    with pytest.raises(ValueError):
        credit.bind_batch(original,batch)

@pytest.mark.parametrize("inputs,expected", (
    ((False,1,1,1,1),"A_BINDING_FAILURE"),
    ((True,0,1,1,2),"B_NO_MEAN_CREDIT"),
    ((True,1,2,1,3),"C_CRITIC_DOMINANT"),
    ((True,1,1,2,3),"D_ACTOR_CLIPPING_ACTIVE"),
    ((True,1,1,2,.5),"E_INCONCLUSIVE"),
))
def test_fixed_classification(inputs,expected):
    assert credit.classify(*inputs)==expected

@pytest.mark.parametrize("failure", (False,True))
def test_method_observer_restores_class_binding(failure):
    class Owner:
        def call(self,x): return x+1
    owner=Owner()
    with pytest.raises(RuntimeError) if failure else credit.nullcontext():
        with credit.observe_method(owner,"call",lambda original: lambda x: original(x)*2):
            assert owner.call(3)==8
            if failure: raise RuntimeError("injected")
    assert "call" not in vars(owner) and owner.call(3)==4

@pytest.mark.parametrize("kind", ("success","binder-failure","duplicate","missing"))
def test_single_pass_final_adam_capture_and_restoration(kind):
    optimizer={"state":{0:{"step":np.array(128,dtype=np.float32)}}, "param_groups":[]}
    def records(*args,**kwargs):
        if kind!="missing":
            yield "final_policy_and_adam",{"optimizer":optimizer}
        if kind=="duplicate":
            yield "final_policy_and_adam",{"optimizer":optimizer}
    module=types.SimpleNamespace(read_records=records)
    def binder(*args,**kwargs):
        stream=module.read_records("x",{})
        try:
            list(stream)
            if kind=="binder-failure": raise RuntimeError("binder failed")
            return {"bound":True}
        finally: stream.close()
    module.read_policy_pair=binder
    if kind=="success":
        pair, result=credit.extract_pair(module,Path("."),{},"J",14011)
        assert pair=={"bound":True}
        result["state"][0]["step"][...]=0
        assert optimizer["state"][0]["step"]==128
    else:
        with pytest.raises((ValueError,RuntimeError)):
            credit.extract_pair(module,Path("."),{},"J",14011)
    assert module.read_records is records

@pytest.mark.parametrize("failure", ("none","before-backward","mutation","returns"))
def test_actual_torch_backward_stops_before_adam_and_removes_hooks(failure):
    torch=pytest.importorskip("torch")
    actor=torch.nn.Linear(2,1); critic=torch.nn.Linear(2,1)
    optimizer=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=.001)
    algorithm=types.SimpleNamespace(actor=actor,critic=critic,optimizer=optimizer)
    def update():
        if failure=="before-backward": raise RuntimeError("before backward")
        optimizer.zero_grad()
        (actor(torch.ones(3,2)).sum()+critic(torch.ones(3,2)).sum()).backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(),1.)
        torch.nn.utils.clip_grad_norm_(critic.parameters(),1.)
        if failure=="mutation":
            with torch.no_grad(): actor.bias.add_(1.)
        if failure=="returns": return {}
        optimizer.step()
        raise AssertionError("optimizer returned after expected sentinel")
    algorithm.update=update
    before=credit.learner_digest(algorithm)
    captured={}
    if failure=="none":
        result=credit.backward_without_step(algorithm,captured,torch)
        assert result["optimizer_calls_blocked"]==1
        assert before==credit.learner_digest(algorithm)
        assert captured["preclip"] and captured["postclip"]
    else:
        with pytest.raises((RuntimeError,ValueError)):
            credit.backward_without_step(algorithm,captured,torch)
    for module in (actor,critic):
        assert all(not p._backward_hooks for p in module.parameters())
    assert not optimizer._optimizer_step_pre_hooks

@pytest.mark.cuda
def test_parent_only_fixed_three_actual_credit_capture(tmp_path):
    """Parent runs this once OR the CLI once; never both on the same declaration."""
    root=Path(__file__).resolve().parents[2]
    result=credit.run_fixed(root/"artifacts"/"p65-parity-n4a-rob1d-col-reducer-20261111-01",
                            tmp_path/"fixed-credit")
    assert result["status"]=="PASS_FIXED_CREDIT_DIAGNOSTIC_ONLY"
    assert result["accepted_world_steps"]==768 and result["optimizer_steps"]==0
    assert [(r["task"],r["seed"]) for r in result["runs"]]==[("J",14011),("R",14211),("S",14111)]
