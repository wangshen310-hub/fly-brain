"""Same actual input-time-series audit using the explicit plastic brain."""
import audit_motor_sensory_inputs as audit
from flysnake.research.teacher_afferent import PlasticAfferentBrain

audit.SpikingConnectome=PlasticAfferentBrain

if __name__=='__main__':audit.main()
