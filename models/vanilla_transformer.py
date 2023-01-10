# Mostly derived from https://huggingface.co/blog/time-series-transformers

from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction

train_dataset = [] # to be synthesized soon

lags_sequence = list(range(50))
num_horizons = 50
config = TimeSeriesTransformerConfig(
    prediction_length=num_horizons,
    context_length=num_horizons*3, # context length
    lags_sequence=lags_sequence,
    num_time_features=0, # no time covariates/features yet
    num_static_categorical_features=0, # no categorical features, though we can think of putting
                                       # something that describes the whole timeseries
    encoder_layers=4, 
    decoder_layers=4,
)

model = TimeSeriesTransformerForPrediction(config)