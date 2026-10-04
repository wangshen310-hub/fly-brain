"""Fresh inference-only replay for the explicitly identified afferent model."""
import verify_teacher_spiking as verifier
from flysnake.research.teacher_afferent import PlasticAfferentBrain

verifier.SpikingConnectome=PlasticAfferentBrain

if __name__=='__main__':verifier.main()
