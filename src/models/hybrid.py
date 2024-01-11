from typing import Optional, Union, Callable

import pandas as pd
import numpy as np

from recommenders.models.deeprec.deeprec_utils import prepare_hparams
from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF

from src.models import LightGCNCustom, NLPSimilarity

def interweave(df, by):
    # https://stackoverflow.com/a/77678823/4505998
    return df.iloc[np.argsort(df.groupby(by).cumcount())]

def merge_apply_prioritize_common(group, top_k=5):
    """
    First, it gets the common recommendations, then interweaves the
    remaining ones
    """
    # This method of merging and applying changes the ranking
    # and the ndcg decreases significantly
    common = group[group['itemID'].duplicated(keep='first')]
    common['prediction'] = np.nan
    common['rec'] = 'both'

    # TODO: sort_index should merge the two arrays grabbing one of each kind
    notcommon = interweave(group.drop_duplicates('itemID', keep=False), 'rec')

    return pd.concat((common, notcommon)).head(top_k).set_index('itemID')[['prediction', 'rec']]

def merge_apply_naive(group, top_k=5):
    """
    Interweaves the two predictors and then removes the duplicate recommendations
    """
    interweaved = interweave(group, 'rec')
    interweaved['rec'] = interweaved['rec'].where(~interweaved.duplicated('itemID', keep=False), 'both')
    interweaved = interweaved.drop_duplicates('itemID', keep='first')

    return interweaved.head(top_k).set_index('itemID')[['prediction', 'rec']]

def merge_apply_avg(group, top_k=5):
    """
    Averages the ranking of the two predictors
    """
    group['hyb_score'] = group.groupby('rec').cumcount()

    common = pd.DataFrame(index=group[group['itemID'].duplicated(keep='first')]['itemID'], columns=['hyb_score', 'prediction', 'rec'])
    if not common.empty:
        common['hyb_score'] = group.groupby('itemID')['hyb_score'].mean()
        common['prediction'] = common['hyb_score']
        common['rec'] = 'both'
    
    notcommon = group.drop_duplicates('itemID', keep=False).set_index('itemID')

    # Always ascending (i.e: the first element will have a score of 0)
    both = pd.concat((common, notcommon)).sort_values('hyb_score')
    both['prediction'] = both['hyb_score']

    return both.head(top_k)[['prediction', 'rec']]

__STR_2_F_DICT = {
    'avg': merge_apply_avg,
    'naive': merge_apply_naive,
    'prioritize': merge_apply_prioritize_common,
}
def strToFunc(string: str):
    return __STR_2_F_DICT[string]

class HybridRecommendation:
    def __init__(self, train, test, dfp, *, seed=None, merge_func: Union[Callable, str] = None, lightgcn_config=dict(), nlp_config=dict()):
        self.train = train
        self.test = test
        self.dfp = dfp
        self.seed = seed
        self.lightgcn = self._prepare_lightgcn(**lightgcn_config)
        self.nlp = self._prepare_nlp(**nlp_config)

        if merge_func is None:
            self._merge_apply = merge_apply_avg
        elif isinstance(merge_func, str):
            self._merge_apply = strToFunc(merge_func)
        else:
            self._merge_apply = merge_func

    def _prepare_lightgcn(self, cf_seed=None, **kwargs):
        hparams = prepare_hparams(
            model_type='lightgcn',
            **kwargs,
        )
        dataloader = ImplicitCF(train=self.train, test=self.test, seed=cf_seed or self.seed)
        return LightGCNCustom(data=dataloader, hparams=hparams, seed=self.seed)

    def _prepare_nlp(self, **kwargs):
        return NLPSimilarity(self.train, self.dfp, **kwargs)

    def fit_epoch(self):
        self.lightgcn.fit_epoch()
        if self.nlp.embeddings is None: self.nlp.fit()

    def fit(self):
        self.lightgcn.fit()
        self.nlp.fit()

    def merge_recommendations(self, nlp_recs, gnn_recs, top_k=5, **kwargs):
        nlp_recs['rec'] = 'nlp'
        gnn_recs['rec'] = 'gnn'

        # Merge the recommendations for each user
        # The sort_index inderleaves the two dataframes
        all_recs = pd.concat((nlp_recs, gnn_recs))

        return all_recs.groupby('userID').apply(self._merge_apply, top_k=top_k, **kwargs).reset_index()
    
    def recommend_k_items(
        self, to_users, top_k=5, sort_top_k=True, remove_seen=True, use_id=False, recommend_from=None, **kwargs,
    ):
        self.nlp_recs = self.nlp.recommend_k_items(to_users, top_k=top_k, remove_seen=remove_seen, recommend_from=recommend_from)

        gnn_users = pd.DataFrame({'userID': to_users})
        self.gnn_recs = self.lightgcn.recommend_k_items(gnn_users, sort_top_k=True, top_k=top_k, remove_seen=remove_seen, recommend_from=recommend_from)

        return self.merge_recommendations(self.nlp_recs, self.gnn_recs, **kwargs)

class WeightedHybridRecommendation(HybridRecommendation):
    def __init__(self, train, test, dfp, w, *_, **kwargs):
        super().__init__(train, test, dfp, **kwargs)
        self.w = w

    def _merge_apply(self, row, top_k=5):
        row[row['rec'] == 'gnn']['prediction'] *= self.w
        return row.sort_values('prediction').head(top_k).set_index('itemID')[['prediction', 'rec']]
