from tqdm import tqdm

from gerador.april.fs import get_process_model_files
from gerador.april.generation.anomaly import *
from gerador.april.generation.drift import *
from gerador.april.generation.utils import generate_for_process_model

anomalies = [
    AttributeAnomaly(max_events=3, max_attributes=2),
    EarlyAnomaly(max_distance=5, max_sequence_size=2),
    InsertAnomaly(max_inserts=2),
    LateAnomaly(max_distance=5, max_sequence_size=2),
    ReworkAnomaly(max_distance=5, max_sequence_size=3),
    SkipSequenceAnomaly(max_sequence_size=2)
]

drifts = [
    GradualDrift(max_sequence_size=2),
    IncrementalDrif(max_sequence_size=2),
    RecurringDrift(max_sequence_size=2),
    SuddenDrift(max_sequence_size=3)
]

process_models = [m for m in get_process_model_files() if 'testing' not in m and 'paper' not in m]

for process_model in tqdm(process_models, desc='Generate'):
    #generate_for_process_model(process_model, size=100, anomalies=anomalies, drifts=drifts, anomaly_p=0.20, num_attr=[1, 2, 3, 4], seed=1337)
    generate_for_process_model(process_model, size=1000, anomalies=anomalies, drifts=drifts, anomaly_p=0, num_attr=[1], seed=1337)
