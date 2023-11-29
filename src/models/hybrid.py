from typing import Optional

import pandas as pd
import numpy as np

from recommenders.models.deeprec.deeprec_utils import prepare_hparams
from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF

from src.models import LightGCNCustom, NLPSimilarity

class HybridRecommendation:
    def __init__(self, train, test, dfp, *, lightgcn_config=dict(), nlp_config=dict()):
        self.train = train
        self.test = test
        self.dfp = dfp
        self.lightgcn = self._prepare_lightgcn(**lightgcn_config)
        self.nlp = self._prepare_nlp(**nlp_config)

    def _prepare_lightgcn(self, cf_seed=None, **kwargs):
        hparams = prepare_hparams(
            model_type='lightgcn',
            **kwargs,
        )
        dataloader = ImplicitCF(train=self.train, test=self.test, seed=cf_seed)
        return LightGCNCustom(data=dataloader, hparams=hparams)

    def _prepare_nlp(self, filter_window: Optional[str] = None, **kwargs):
        train = self.train
        if filter_window:
            t = train['timestamp'].max()
            offset = pd.tseries.frequencies.to_offset(filter_window)

            train = train[train['timestamp'] > (t - offset)]
        
        return NLPSimilarity(train, self.dfp, **kwargs)

    def fit_epoch(self):
        self.lightgcn.fit_epoch()
        if self.nlp.embeddings is None: self.nlp.fit()

    def fit(self):
        self.lightgcn.fit()
        self.nlp.fit()

    def _merge_apply(self, row, top_k=5):
        common = row[row['itemID'].duplicated(keep='first')]
        common['prediction'] = np.nan
        common['rec'] = 'both'
        notcommon = row.drop_duplicates('itemID', keep=False).sort_index()
    
        return pd.concat((common, notcommon)).head(top_k).set_index('itemID')[['prediction', 'rec']]

    def merge_recommendations(self, nlp_recs, gnn_recs, top_k=5, **kwargs):
        nlp_recs['rec'] = 'nlp'
        gnn_recs['rec'] = 'gnn'

        all_recs = pd.concat((nlp_recs, gnn_recs))
        return all_recs.groupby('userID').apply(self._merge_apply, top_k=top_k, **kwargs).reset_index()
    
    def recommend_k_items(
        self, to_users, top_k=5, sort_top_k=True, remove_seen=True, use_id=False, recommend_from=None, **kwargs,
    ):
        nlp_recs = self.nlp.recommend_k_items(to_users, top_k=top_k, remove_seen=remove_seen, recommend_from=recommend_from)

        gnn_users = pd.DataFrame({'userID': to_users})
        gnn_recs = self.lightgcn.recommend_k_items(gnn_users, sort_top_k=True, top_k=top_k, remove_seen=remove_seen, recommend_from=recommend_from)

        return self.merge_recommendations(nlp_recs, gnn_recs, **kwargs)

class WeightedHybridRecommendation(HybridRecommendation):
    def __init__(self, train, test, dfp, w, *_, **kwargs):
        super().__init__(train, test, dfp, **kwargs)
        self.w = w

    def _merge_apply(self, row, top_k=5):
        row[row['rec'] == 'gnn']['prediction'] *= self.w
        return row.sort_values('prediction').head(top_k).set_index('itemID')[['prediction', 'rec']]
