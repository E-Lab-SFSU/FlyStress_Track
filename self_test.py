from sleep_analysis.fly_detection import FlyDetection
from sleep_analysis.multi_fly_tracker import PerWellMultiFlyTracker

def d(x,y,a=40,n=1): return FlyDetection('A1',x,y,x,y,a,30,n)
t=PerWellMultiFlyTracker(['A1'],flies_per_well=3)
r=t.update_all({'A1':[d(10,10),d(30,10),d(50,10)]},0.0)
assert [x.observation_status for x in r]==['DETECTED','DETECTED','DETECTED']
r=t.update_all({'A1':[d(11,10),d(31,10),d(51,10)]},1.0)
assert [x.observation_status for x in r]==['','','']
r=t.update_all({'A1':[d(12,10),d(41,10,85,2)]},2.0)
assert sum(x.observation_status=='OVERLAP' for x in r)==2
r=t.update_all({'A1':[d(13,10)]},3.0)
assert sum(x.observation_status=='UNKNOWN' for x in r)==2
print('FlyStress three-fly self-test passed.')
