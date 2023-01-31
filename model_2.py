
from src.models import Wav2Vec2ForSpeechClassification
import torch
import torch.nn as nn
from transformers import AutoConfig

label_list = ['negative', 'positive']
class wav2vec2(nn.Module):
    def __init__(self, model_args, args):
        super(wav2vec2, self).__init__()
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=2,
            label2id={label: i for i, label in enumerate(label_list)},
            id2label={i: label for i, label in enumerate(label_list)},
            finetuning_task="wav2vec2_clf",
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        feature_extractor = Wav2Vec2ForSpeechClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        model = feature_extractor.wav2vec2
        self.feature_extractor = model.feature_extractor
        self.feature_projection = model.feature_projection
        self.encoder = model.encoder
        self.encoder.layers = self.encoder.layers[:5]
        
        self.proj1 = nn.Linear(1024, 1024)
        self.proj2 = nn.Linear(1024, args.num_classes_s+1)
        self.proj2 = nn.Linear(1024, 256)
        self.proj3 = nn.Linear(256, 128)
        self.proj4 = nn.Linear(128, args.num_classes_s+1)
        self.dropout = nn.Dropout(p=0.05, inplace=False)

    def forward(self, x):

        feature = self.feature_extractor(x)
        feature = feature.mean(dim=-1)
        feature = self.feature_projection(feature)
        a, b = feature
        feature = a
        feature = feature.unsqueeze(0)
        feature = self.encoder(feature)
        feature = feature['last_hidden_state']
        feature = feature.squeeze(0)
        feature = feature.unsqueeze(dim=1)
        
        return feature