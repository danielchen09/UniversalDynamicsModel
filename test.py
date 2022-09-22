from dm_control import suite
from dm_control.suite import acrobot
from dm_control.suite import swimmer
from dm_control.suite import common
from dm_control.rl import control
from dm_control.utils import io as resources
import xml.etree.ElementTree as ET

tree = ET.parse('xmls/swimmer6.xml')
root = tree.getroot()
for option in root.iter('option'):
    for flag in option.iter('flag'):
        flag.set('frictionloss', 'enable')
for joint in root.iter('joint'):
    joint.attrib.pop('frictionloss', None)
    joint.attrib.pop('damping', None)
physics = acrobot.Physics.from_xml_string(ET.tostring(root, encoding='utf8').decode(), common.ASSETS)
task = acrobot.Balance(sparse=False, random=None)
env = control.Environment(physics, task, time_limit=10)
print(env.physics.named.data.sensordata)
print(env.physics.data.energy)
breakpoint()