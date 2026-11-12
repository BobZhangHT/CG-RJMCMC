import numpy as np
import pytest
from scripts import small_sample_validation as sn

def layouts():
    person=np.repeat(np.arange(21),4)
    trial=np.tile([1,2,3,4],21)
    z=np.zeros(84);c=np.tile([.5,-.5,.5,-.5],21)
    return person,trial,z,c

def test_cyclic_layout_and_independent_outcomes():
    template=layouts();train,test=sn.expand_layouts(template,template)
    assert np.array_equal(train[1][train[0]==0],train[1][train[0]==21])
    sn.STATE['protocol']={'cells':sn.CELLS}
    seed=sn.seed('formal','SN1',352,'DATA')
    a,b,truth=sn.simulate('SN1',seed,train,test)
    aa,bb,_=sn.simulate('SN1',seed,train,test)
    assert np.array_equal(a[-1],aa[-1]);assert np.array_equal(b[-1],bb[-1])
    assert not np.array_equal(a[-1][a[0]==0],a[-1][a[0]==21])
    assert np.all(truth[:32]==.04) and np.all(truth[32:]==.14)
    assert seed!=sn.seed('formal','SN1',353,'DATA')

def test_original_seed_values_and_design_validation():
    assert sn.seed('formal','SN0',352,'DATA')==sn.formal.deterministic_seed(sn.NAMESPACE,'formal|SN0|R000352|DATA')
    invalid=tuple(a[:4] for a in layouts())
    with pytest.raises(ValueError):sn.expand_layouts(invalid,invalid)


def test_self_contained_smoke_design_and_formal_requirement(tmp_path):
    train,test=sn.synthetic_smoke_design()
    assert set(train[0])==set(range(35))
    assert np.array_equal(train[1],test[1])
    assert not np.array_equal(train[3],test[3])
    for trial in range(1,65):
        assert set(train[3][train[1]==trial])=={-.5,.5}
    with pytest.raises(SystemExit) as exc:sn.main(['--full','--output-root',str(tmp_path)])
    assert exc.value.code==2


def test_duplicate_cache_identity_rejected():
    record=dict(cell='SN0',rep=1,methods=[{'method':'CG-RJMCMC'}])
    with pytest.raises(ValueError,match='duplicate N35 cell'):sn.validate_record_identities([record,dict(record)])
    with pytest.raises(ValueError,match='duplicate N35 method'):sn.validate_record_identities([dict(record,methods=[{'method':'CG-RJMCMC'},{'method':'CG-RJMCMC'}])])
