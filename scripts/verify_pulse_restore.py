"""Independent native reload tool for the pulse-balanced experiment.

Reuse the retained verification orchestration with the NEW brain, game and
provenance. Old experiments and verifier files are unchanged.
"""
from pathlib import Path
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import train_pulse_balanced as pulse
import verify_actor_restore as verifier
from flysnake.research.pulse_balance import PulseBalancedBrain,LEARNING_VERSION

verifier.NativeActorCreditBrain=PulseBalancedBrain
verifier.sources=pulse.sources

if __name__=='__main__':
    verifier.main()
    # The underlying tool validates all native replay data before returning.
    parser_output=Path(sys.argv[sys.argv.index('--output')+1])
    result=json.loads(parser_output.read_text(encoding='utf-8'))
    result.update(protocol=LEARNING_VERSION,
                  verification_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  retained_verifier_source_sha256=hashlib.sha256(Path(verifier.__file__).read_bytes()).hexdigest())
    from flysnake.lab.experiment import atomic_json
    atomic_json(parser_output,result)
